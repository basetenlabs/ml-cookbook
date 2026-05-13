# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import glob
import json
import math
import os
import random
import shutil

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from huggingface_hub import snapshot_download
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, get_cosine_schedule_with_warmup


def resolve_model_path(path_or_repo_id: str) -> str:
    """Return a local directory path for either a local checkpoint or a HF repo id.

    The downstream code does `shutil.copytree(MODEL_PATH, ...)` and reads
    `config.json` directly off `MODEL_PATH`, so we must hand it a real
    filesystem path. If the user passed a HF repo id, materialize it via
    `snapshot_download` (cached, so this is a no-op on repeat runs).
    """
    if os.path.isdir(path_or_repo_id):
        return path_or_repo_id
    return snapshot_download(repo_id=path_or_repo_id)


def compute_loss(model, batch, speaker_embedding, sub_talker_loss_weight: float = 0.3):
    """Forward pass + composite TTS loss used by both train and eval.

    `speaker_embedding` is the (1, hidden) constant produced once before
    training by `compute_target_speaker_embedding`. It's broadcast into
    position 6 of the codec-embedding row on every step. The speaker encoder
    itself is frozen (single-speaker recipe with one shared ref_audio), so
    recomputing it per micro-batch would be pure waste.
    """
    input_ids = batch['input_ids']
    codec_ids = batch['codec_ids']
    text_embedding_mask = batch['text_embedding_mask']
    codec_embedding_mask = batch['codec_embedding_mask']
    attention_mask = batch['attention_mask']
    codec_0_labels = batch['codec_0_labels']
    codec_mask = batch['codec_mask']

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]

    input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
    input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    batch_size = input_codec_embedding.shape[0]
    input_codec_embedding[:, 6, :] = speaker_embedding.to(
        device=input_codec_embedding.device, dtype=input_codec_embedding.dtype
    ).expand(batch_size, -1)

    input_embeddings = input_text_embedding + input_codec_embedding

    for i in range(1, 16):
        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
        input_embeddings = input_embeddings + codec_i_embedding

    outputs = model.talker(
        inputs_embeds=input_embeddings[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        labels=codec_0_labels[:, 1:],
        output_hidden_states=True,
    )

    # `Qwen3TTSTalker.forward` returns `hidden_states=(outputs.hidden_states,
    # codec_ids)` (a 2-tuple), so `[0]` unwraps the inner tuple-of-layers
    # produced by the underlying causal LM and `[-1]` selects the last
    # layer's `(batch, seq, hidden)` tensor — the talker's final hidden
    # state at every position.
    hidden_states = outputs.hidden_states[0][-1]
    # `codec_mask[:, :-1]` (not `codec_mask[:, 1:]`) is the upstream-correct
    # slice — see QwenLM/Qwen3-TTS commit 022e286 ("fix finetuning bug").
    # The sub-talker is the depth head that predicts channels 1..15 of
    # frame t from the talker's frame-t hidden state, so the hidden states
    # passed in must be those AT codec positions, not those one step before.
    talker_hidden_states = hidden_states[codec_mask[:, :-1]]
    talker_codec_ids = codec_ids[codec_mask]

    _sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(
        talker_codec_ids, talker_hidden_states
    )

    loss = outputs.loss + sub_talker_loss_weight * sub_talker_loss
    return loss


@torch.no_grad()
def compute_target_speaker_embedding(model, ref_audio_path, sample_dataset):
    """Run the (frozen) speaker encoder on `ref_audio_path` once.

    This is the vector we write into the codec embedding table at checkpoint
    save time AND broadcast into the codec embedding column at every training
    step. Because the encoder is frozen, computing it once up front is
    equivalent to averaging across the dataset — and is faithful by
    construction when every row in the JSONL shares the same `ref_audio`.

    `sample_dataset` is only used for its audio-loading helpers (so we don't
    duplicate librosa/mel-spectrogram code here).
    """
    wav, sr = sample_dataset._load_audio_to_np(ref_audio_path)
    ref_mel = sample_dataset.extract_mels(audio=wav, sr=sr)
    ref_mel = ref_mel.to(model.device).to(model.dtype)
    return model.speaker_encoder(ref_mel)


@torch.no_grad()
def evaluate(model, eval_dataloader, speaker_embedding, sub_talker_loss_weight):
    """Run the eval forward pass over `eval_dataloader` and return mean loss.

    Returns NaN if the dataloader is empty so callers can detect "no eval
    happened" without special-casing zero-batch averages.
    """
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in eval_dataloader:
        loss = compute_loss(model, batch, speaker_embedding, sub_talker_loss_weight)
        total_loss += loss.item()
        total_batches += 1
    if was_training:
        model.train()
    if total_batches == 0:
        return float("nan")
    return total_loss / total_batches


def save_checkpoint(
    output_dir: str,
    model,
    accelerator: Accelerator,
    model_path: str,
    target_speaker_embedding: torch.Tensor,
    speaker_name: str,
    training_args: dict,
):
    """Persist a self-contained Qwen3-TTS checkpoint to `output_dir`.

    Only rank-0 actually writes; callers should already be gated on
    `accelerator.is_main_process`. We assume `accelerator.wait_for_everyone()`
    has been called immediately before so other ranks aren't racing into the
    next epoch.
    """
    # Clone the base model's auxiliary files (text tokenizer, configs,
    # processor, speech_tokenizer/) so each checkpoint is self-contained
    # and loadable via `Qwen3TTSModel.from_pretrained(<checkpoint_dir>)`.
    #
    # `speech_tokenizer/` (~682 MB) is the wav<->codes codec. It is never
    # trained (audio_codes are precomputed by prepare.py), but
    # `Qwen3TTSModel.from_pretrained` only fetches it from the HF Hub when
    # the path argument is a repo id; for a local checkpoint dir it expects
    # `speech_tokenizer/config.json` on disk and raises OSError otherwise.
    # So we copy it through.
    #
    # README.md / .gitattributes are skipped — docs / git metadata not
    # needed at inference.
    shutil.copytree(
        model_path,
        output_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("README.md", ".gitattributes"),
    )

    # Defensive cleanup: if the base model ever ships sharded weights, the
    # copytree above will bring over `model-00001-of-N.safetensors` plus
    # `model.safetensors.index.json`, and our subsequent `save_file` of a
    # consolidated `model.safetensors` would leave the index pointing at
    # stale shards. Wipe any prior weight files before writing our own.
    for stale in glob.glob(os.path.join(output_dir, "model*.safetensors*")):
        os.remove(stale)
    for stale in glob.glob(os.path.join(output_dir, "pytorch_model*.bin*")):
        os.remove(stale)

    input_config_file = os.path.join(model_path, "config.json")
    output_config_file = os.path.join(output_dir, "config.json")
    with open(input_config_file, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config_dict["tts_model_type"] = "custom_voice"
    talker_config = config_dict.get("talker_config", {})
    talker_config["spk_id"] = {speaker_name: 3000}
    talker_config["spk_is_dialect"] = {speaker_name: False}
    config_dict["talker_config"] = talker_config
    with open(output_config_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "training_args.json"), "w", encoding="utf-8") as f:
        json.dump(training_args, f, indent=2, ensure_ascii=False)

    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

    # Speaker encoder is frozen and not trained; drop its weights from the
    # checkpoint to save a few hundred MB. Inference re-loads them from the
    # base model dir copied above.
    drop_prefix = "speaker_encoder"
    for k in [k for k in state_dict.keys() if k.startswith(drop_prefix)]:
        del state_dict[k]

    # Bake the target speaker embedding into the codec embedding table at
    # the configured `spk_id` (3000). At inference, the talker reads this
    # row instead of running the speaker encoder.
    weight = state_dict["talker.model.codec_embedding.weight"]
    state_dict["talker.model.codec_embedding.weight"][3000] = (
        target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
    )

    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help=(
            "Micro-batches accumulated per optimizer step. Effective batch "
            "size = batch_size * gradient_accumulation_steps. Smaller values "
            "give more (noisier) updates per epoch, which tends to help "
            "small-data SFT."
        ),
    )
    # Defaults below intentionally mirror run.sh / README. TTS fine-tuning is
    # sensitive to LR — values much above ~1e-5 tend to degrade speaker
    # quality on small corpora, so we lean conservative.
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help=(
            "Fraction of total optimizer steps spent linearly warming the LR "
            "from 0 to --lr before cosine decay to 0. Set to 0.0 to disable "
            "the schedule entirely (constant LR)."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=12)
    parser.add_argument(
        "--sub_talker_loss_weight",
        type=float,
        default=0.3,
        help=(
            "Weight on the sub-talker (depth head) cross-entropy added to the "
            "main talker loss. 0.3 is the upstream default; lower it (e.g. "
            "0.1) if the depth head over-fits before the main loss settles, "
            "raise it (e.g. 0.5) if codec channels 1..15 sound noisy."
        ),
    )
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument(
        "--eval_split",
        type=int,
        default=0,
        help=(
            "Hold out this many rows from --train_jsonl as a validation set "
            "(chosen pseudo-randomly with --eval_seed). Eval loss is printed "
            "at the end of every epoch. 0 disables eval."
        ),
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=42,
        help=(
            "RNG seed for the deterministic train/eval split. Same seed + "
            "same JSONL = same split across runs, so eval loss is comparable."
        ),
    )
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=5,
        help=(
            "Write a checkpoint every N epochs. The final epoch is always "
            "saved regardless of N so short runs (num_epochs < N) still "
            "produce one checkpoint."
        ),
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="How often (in micro-batches) to print + log train loss / LR.",
    )
    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=args.output_model_path,
    )
    # Tensorboard logs land under `<output_model_path>/sft_12hz/`. Drop the
    # full argparse namespace as the run's hparams so each TB run is
    # self-describing.
    accelerator.init_trackers("sft_12hz", config=vars(args))

    MODEL_PATH = resolve_model_path(args.init_model_path)

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    # Freeze the speaker encoder: single-speaker recipe runs it once before
    # training (see compute_target_speaker_embedding below) and reuses the
    # cached embedding on every step. Setting requires_grad=False keeps its
    # params out of the optimizer entirely.
    qwen3tts.speaker_encoder.requires_grad_(False)

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]

    # Hold out a deterministic eval slice. We shuffle indices with a seeded
    # RNG (rather than just slicing the head/tail) because the input JSONL is
    # often produced by a sorted pipeline (prepare.py sorts by file_name), and
    # grabbing a contiguous block would skew eval toward one end of the
    # corpus.
    eval_data = []
    if args.eval_split > 0:
        if args.eval_split >= len(train_data):
            raise ValueError(
                f"--eval_split={args.eval_split} must be smaller than the "
                f"dataset ({len(train_data)} rows)."
            )
        rng = random.Random(args.eval_seed)
        indices = list(range(len(train_data)))
        rng.shuffle(indices)
        eval_idx = set(indices[: args.eval_split])
        eval_data = [train_data[i] for i in sorted(eval_idx)]
        train_data = [
            row for i, row in enumerate(train_data) if i not in eval_idx
        ]
        accelerator.print(
            f"Held out {len(eval_data)} eval samples (seed={args.eval_seed}); "
            f"training on {len(train_data)}."
        )

    # `skip_ref_mel=True` because the speaker embedding is precomputed once
    # (see below) and broadcast into every micro-batch — there's no reason
    # to re-extract the mel spectrogram per __getitem__.
    dataset = TTSDataset(train_data, qwen3tts.processor, config, skip_ref_mel=True)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    eval_dataloader = None
    if eval_data:
        eval_dataset = TTSDataset(eval_data, qwen3tts.processor, config, skip_ref_mel=True)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=eval_dataset.collate_fn,
        )

    optimizer = AdamW(
        [p for p in qwen3tts.model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    # Cosine LR schedule with linear warmup. With small datasets a flat LR
    # tends to over-shoot early (gradient noise on a partially-adapted speaker
    # head) and to over-fit late; warmup + cosine decay smooths both ends.
    # `optimizer_steps_per_epoch` mirrors how often Accelerator actually steps
    # the optimizer (once every `gradient_accumulation_steps` micro-batches),
    # so the cosine reaches its minimum at the end of training rather than
    # mid-run.
    optimizer_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    total_optimizer_steps = optimizer_steps_per_epoch * args.num_epochs
    warmup_steps = int(args.warmup_ratio * total_optimizer_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader, scheduler
    )
    if eval_dataloader is not None:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    accelerator.print(
        f"Training plan: {args.num_epochs} epochs x "
        f"{optimizer_steps_per_epoch} opt steps/epoch = "
        f"{total_optimizer_steps} total steps "
        f"(warmup={warmup_steps}, peak_lr={args.lr})"
    )

    # Pick the reference audio used for the saved speaker embedding. The
    # single-speaker recipe writes the same `ref_audio` on every row of the
    # JSONL, so train_data[0] is canonical; warn if that invariant is broken.
    ref_audio_path = train_data[0]['ref_audio']
    distinct_refs = {row['ref_audio'] for row in train_data}
    if len(distinct_refs) > 1:
        accelerator.print(
            f"[warn] Found {len(distinct_refs)} distinct ref_audio values in "
            f"{args.train_jsonl}; saving the embedding for the first row's "
            f"ref_audio: {ref_audio_path}"
        )
    # Run the (frozen) speaker encoder exactly once on the shared ref_audio.
    # The resulting (1, hidden) tensor is what we (a) inject into the codec
    # embedding column at every training step and (b) bake into the codec
    # embedding table at save time.
    target_speaker_embedding = compute_target_speaker_embedding(
        model, ref_audio_path, dataset
    )

    training_args_snapshot = vars(args)

    best_eval_loss = float("inf")
    global_step = 0
    model.train()

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                loss = compute_loss(
                    model, batch, target_speaker_embedding, args.sub_talker_loss_weight
                )
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                # `scheduler` was passed through `accelerator.prepare`, so it
                # only advances on real optimizer steps (i.e. once per
                # `gradient_accumulation_steps` micro-batches). Calling step()
                # unconditionally here is the canonical accelerate pattern.
                scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1

            if step % args.log_every_n_steps == 0:
                current_lr = scheduler.get_last_lr()[0]
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | "
                    f"Loss: {loss.item():.4f} | LR: {current_lr:.2e}"
                )
                accelerator.log(
                    {"train/loss": loss.item(), "train/lr": current_lr, "train/epoch": epoch},
                    step=global_step,
                )

        if eval_dataloader is not None:
            eval_loss = evaluate(
                model, eval_dataloader, target_speaker_embedding, args.sub_talker_loss_weight
            )
            accelerator.print(
                f"Epoch {epoch} | Eval Loss: {eval_loss:.4f} "
                f"(n={len(eval_data)})"
            )
            accelerator.log(
                {"eval/loss": eval_loss, "eval/epoch": epoch},
                step=global_step,
            )

        is_periodic_save = (epoch + 1) % args.save_every_n_epochs == 0
        is_final_epoch = epoch == args.num_epochs - 1
        should_save = is_periodic_save or is_final_epoch

        # Sync all ranks before rank-0 starts the slow copytree + state_dict
        # gather. Without this, other ranks happily proceed into the next
        # epoch while rank-0 is mid-save, which races on shared filesystems
        # and on shutdown.
        if should_save:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
                save_checkpoint(
                    output_dir=output_dir,
                    model=model,
                    accelerator=accelerator,
                    model_path=MODEL_PATH,
                    target_speaker_embedding=target_speaker_embedding,
                    speaker_name=args.speaker_name,
                    training_args=training_args_snapshot,
                )
            accelerator.wait_for_everyone()

        # Best-checkpoint tracking. Persisted to `<output_model_path>/best/`
        # so callers can always grab the lowest-eval-loss snapshot without
        # having to parse training logs. Only updated when eval is enabled
        # and the current epoch beats the running minimum.
        if eval_dataloader is not None and eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                best_dir = os.path.join(args.output_model_path, "best")
                save_checkpoint(
                    output_dir=best_dir,
                    model=model,
                    accelerator=accelerator,
                    model_path=MODEL_PATH,
                    target_speaker_embedding=target_speaker_embedding,
                    speaker_name=args.speaker_name,
                    training_args={
                        **training_args_snapshot,
                        "best_epoch": epoch,
                        "best_eval_loss": best_eval_loss,
                    },
                )
                accelerator.print(
                    f"[best] Updated {best_dir} (epoch {epoch}, eval_loss={best_eval_loss:.4f})"
                )
            accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    train()
