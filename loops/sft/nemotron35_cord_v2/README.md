# Nemotron 3.5 CORD-v2 vision LoRA SFT

Train receipt-image extraction through the **native Baseten Loops SDK**, then
score a saved checkpoint and the unadapted base model on the official test split.
This example has its own pinned environment; it does not use the parent
directory's Tinker compatibility shim. No local GPU, container or model-weight
download is needed. CPU PyTorch is used only for image resizing.

## Recipe

| Setting | Value |
| --- | --- |
| Managed model | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` |
| Dataset | `naver-clova-ix/cord-v2`, commit `7f0115a4b758a71d6473b8d085751692da2fef98` |
| Official splits | 800 train, 100 validation, 100 test |
| LoRA | Rank 64; alpha and target modules are managed, not client overrides |
| Optimizer | Adam: betas 0.9/0.95, epsilon 1e-8, weight decay 0.01, clip norm 1.0 |
| Learning rate | Constant 2e-4; no warmup or decay |
| Global batch / seed | 8 / 42, client-side shuffle each epoch |
| Training context | 8192; reject overlong complete sequences, never truncate |
| Full training | 400 updates = four complete epochs |
| Validation | Every 100 updates, all 100 validation receipts, forward-only/token-weighted CE |
| Checkpoints | Update 400 only; a shorter smoke saves at its final update |

The user prompt is `Describe this image.`. Targets are CORD `gt_parse` objects
serialized as reverse-sorted `<s_field>value</s_field>` tags, with `<sep/>` between
list elements. Only the assistant answer and turn terminator contribute to loss;
the image, user prompt and template-injected thinking prefix are masked. Labels
are explicitly shifted for next-token prediction. Training never downloads the
test split; generation evaluation is a separate operation.

## Prerequisites and cost

- Python 3.13 and [uv](https://docs.astral.sh/uv/).
- A Baseten API key for a workspace with this **vision SFT model enabled** and
  sufficient training and standalone-sampler capacity. Model availability is
  workspace-dependent; preflight checks access without allocating a trainer.
- A Weights & Biases account/key, or `WANDB_MODE=offline` for local logging.
- Internet access to Hugging Face for the public tokenizer, processor metadata
  and dataset. No manually prepared tokenizer directory is required; setup
  downloads and caches the metadata automatically. `HF_TOKEN` is optional for
  higher download rate limits.
- Roughly 2.1 GB for public train/validation parquet, plus prepared images and
  the later test download. Leave several GB free.

**Training and generation allocate billable remote GPUs.** The historical
trainer used eight B200s and each standalone sampler used four B200s; these are
observations, not placement guarantees. Cold provisioning can take several
minutes. `replicas=1` means one managed trainer, not one GPU.

The Python client must remain running and connected to submit updates. A laptop
sleep or `kill -9` can interrupt the loop and bypass cleanup. For a long run,
use a stable workstation/VM. Closing a Python client alone does not stop billing.

## 1. Install and configure

From the cookbook checkout:

```bash
cd loops/sft/nemotron35_cord_v2
uv sync --locked
export BASETEN_API_KEY="your-baseten-api-key"
export WANDB_API_KEY="your-wandb-api-key"
export WANDB_PROJECT="loops-cord-v2"
```

Optionally set `BASETEN_TEAM` and `WANDB_ENTITY` if your accounts require them.
All run names, custom metrics, plots and artifacts start with `baseten_`.
`--env-file /path/to/keys` is available for train/preflight/evaluation/cleanup;
only explicitly supplied files are read, and shell values take precedence.
Never commit keys or generated runtime directories.

The standalone environment retains the cookbook's `certifi>=2026.10.19`
security requirement. Use the checked-in lockfile rather than reusing an older
environment that bypasses this minimum.

Run the offline unit tests; these require neither model assets nor API keys:

```bash
uv run python -m unittest discover -s tests -v
```

## 2. Fetch the public tokenizer and processor metadata (no GPUs)

Run:

```bash
uv run python prepare_assets.py
```

The default source is the public model repository:
[`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/tree/main).
No `--tokenizer-dir`, manual file copy, or HF login is needed for normal setup.

This downloads **only** `tokenizer.json`, `tokenizer_config.json`, the chat
template when present, and `preprocessor_config.json`—not model weights. Setup
creates the ignored `assets/tokenizer/` cache itself; it is not a directory you
must supply or include in a PR. It resolves the repository to an immutable
commit, records that commit and SHA256s in `assets/manifest.json`, and validates
the processor geometry and template without executing remote code.
To rerun from a particular HF snapshot, pass its recorded 40-character commit:

```bash
uv run python prepare_assets.py --revision YOUR_RECORDED_HF_COMMIT
```

Assets are cached, fingerprinted and ignored by Git. Repeating setup verifies
the cache instead of replacing it. To change assets/revision, use a fresh recipe
copy; never overwrite assets underneath already-tokenized data.

<details>
<summary>Optional: offline or historical metadata import</summary>

This fallback is not required for normal Hugging Face setup. To import an
existing authorized metadata snapshot instead of downloading it:

```bash
uv run python prepare_assets.py \
  --tokenizer-dir /path/to/authorized/tokenizer \
  --processor-config /path/to/authorized/preprocessor_config.json
```

The optional local tokenizer directory must contain `tokenizer.json` and
`tokenizer_config.json`, plus `chat_template.jinja` when the template lives in
that file. Do not copy an arbitrary model's tokenizer or a Qwen processor.

For strict asset replay of the historical audited run, add
`--require-reference-fingerprints` to the asset command. The four expected
hashes are in `reference_fingerprints.json`; mismatches stop before allocation.
The historical tokenizer came from the supported BF16 model, but its processor
configuration came from a prior NVIDIA 09/21 cache. Identity with the managed
BF16 server processor was **not verified**. Different authorized assets are a
new adaptation, not an exact replay of the reported score.

</details>

## 3. Prepare and preflight data (no GPUs)

```bash
uv run python prepare_data.py
uv run python preflight.py --offline
uv run python preflight.py
```

Preparation downloads the pinned public train/validation splits, resizes images,
tokenizes every complete example, verifies counts and split disjointness, and
publishes the cache only after all 900 rows pass. Repeating it verifies and
reuses the existing cache. `CORD_HF_CACHE=/path/to/existing/hf_cache` can reuse
downloaded parquet without copying it; new prepared rows still stay local.

Any image/token/hash/count failure stops before allocating GPUs. There is no
silent truncation or dropped-row fallback. An 8K audit is not proof of managed
processor parity: run the smoke before committing to a full training run.

## 4. Smoke, then fresh full training

Use a new timestamp for each rerun; existing output directories are never
overwritten. These names also determine the exact output locations:

```bash
STAMP=$(date -u +%Y%m%d_%H%M%S)
SMOKE="baseten_cord_v2_smoke_${STAMP}"
TRAIN="baseten_cord_v2_full_${STAMP}"
EVAL="baseten_cord_v2_step400_test_f1_${STAMP}"
BASELINE="baseten_cord_v2_baseline_test_f1_${STAMP}"

uv run python train.py --steps 3 --name "$SMOKE"
```

Confirm `outputs/$SMOKE/completion.json` says `complete: true`, three finite
updates completed, final validation covered all 100 receipts, both checkpoint
paths exist, and its trainer has become inactive. The longest-receipt probe is
forward-only, not an optimizer update. A smoke saves at update 3.

Then run the full recipe:

```bash
uv run python train.py --name "$TRAIN"
```

Every invocation **freshly initializes from the base model**. The full run does
not resume the smoke. Unset `LOOPS_REUSE_FROM_RUN_ID` and
`LOOPS_REUSE_FROM_SESSION_ID`; the script rejects them to avoid attaching to an
unrelated trainer. There is no resume implementation. A rerun means a new
trainer and 400 new updates, not reuse of the previous checkpoint.

The pinned SDK exposes rank but not alpha or target modules; the historical
rank64 run used alpha32 and included `lm_head`.
MTP training, optimizer precision, topology and microbatch placement are managed
by Baseten rather than controlled by this recipe.

## 5. Checkpoint F1 and matched baseline

After a verified **400-update** full run (not a smoke):

```bash
uv run python evaluate_f1.py \
  --training-output "outputs/$TRAIN" --name "$EVAL"

uv run python evaluate_f1.py --baseline \
  --training-output "outputs/$TRAIN" \
  --compare-output "outputs/$EVAL" --name "$BASELINE"
```

The checkpoint evaluator provisions a new standalone sampler pinned to the
saved immutable `sampler_weights` URI. The baseline provisions a separate base
model sampler **without an adapter**. Neither command restarts the trainer.
Each covers all 100 official test receipts before reporting a complete score.
The baseline validates identical input fingerprints and generation settings,
recomputes the checkpoint score, then reports the F1 delta in percentage points.

Generation uses temperature 0, top_p 1, seed42, thinking enabled through the
template, 65536 inference context and a 32768-token output budget. The echoed
expanded prompt tokens must match exactly. These are inference limits, not
training context. The raw SDK does not expose reasoning-effort or MTP/speculative
decoding controls; no vLLM-equivalence claim is made.

The metric is **micro exact receipt field-path/value F1**, not official CORD
tree-edit/fuzzy accuracy, tool-call F1 or general OCR accuracy. Field values are
case-sensitive with outer whitespace stripped; list order is ignored and
duplicates count as a multiset. Malformed output counts as an empty prediction;
input/output truncations and malformed-output counts are reported separately.

The shared prompt deliberately stays `Describe this image.`. Base-model prose
does not satisfy the trained tag/JSON schema, so the comparison includes
**format adherence as well as extraction**. Do not interpret the delta as an
isolated OCR improvement. Changing the prompt/schema requires a separately
labeled experiment for both models, not overwriting this comparison.

## Artifacts, cleanup and expected results

Each `outputs/baseten_*` directory records configuration, data/asset audits,
remote resource IDs, W&B URL, metrics/plots or all rollouts, completion audits,
and deactivation requests. Training stores `checkpoints.json` and the backend
checkpoint registry. Evaluation saves `rollouts.jsonl`, `test_snapshot.json`,
`test_audit.json`, `completion_audit.json`, and baseline `comparison.json`.
Training/test F1 summaries and loss curves are logged to W&B; baseline metrics
have distinct keys so they do not overwrite the finetuned score.

Normal completion, Python exceptions and Ctrl-C request deactivation of **only
the resource created by that invocation**. `--keep-active` intentionally keeps
the trainer billable; it is not the default. If the client was forcibly killed
or cleanup failed, inspect the resource record and explicitly clean it up:

```bash
uv run python cleanup.py --training-output "outputs/$TRAIN" --yes
uv run python cleanup.py --evaluation-output "outputs/$EVAL" --yes
uv run python cleanup.py --evaluation-output "outputs/$BASELINE" --yes
```

Verify `INACTIVE` in Baseten; a recorded deactivation request alone is not proof
that billing has stopped. If the resource ID was not written during a hard kill,
identify the matching uniquely named resource in the dashboard before stopping
it. Do not deactivate unrelated jobs.

The prior audited 400-update run, using the historical fingerprints, achieved:

| Result | Observed value |
| --- | --- |
| Final train CE | 0.00387465 |
| Validation CE at 100 / 200 / 300 / 400 | 0.02424193 / 0.02304767 / 0.02760747 / 0.02592035 |
| Step400 test F1 / precision / recall | 87.84975% / 87.61468% / 88.08609% |
| Test malformed / truncated outputs | 0 / 0, all 100 receipts |
| Matched base strict structured F1 | 0%; all 100 outputs failed the target schema |

The base result is **not 0% OCR accuracy**. These are historical observations,
not guarantees for this cleaned recipe, a backend update or different assets.
Seed42 controls client ordering and requested sampling seed; managed training
and inference are not promised bitwise deterministic. No new paid GPU rerun
is implied by local tests passing.

## Sources

- [Baseten Loops](https://docs.baseten.co/loops/overview), [vision inputs](https://docs.baseten.co/loops/vision), [SDK](https://docs.baseten.co/reference/sdk/loops/service-client)
- [CORD-v2 dataset](https://huggingface.co/datasets/naver-clova-ix/cord-v2), [CORD source](https://github.com/clovaai/cord)
