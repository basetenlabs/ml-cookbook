# Nemotron 3.5 ASR Fine-Tuning with NeMo

This example fine-tunes NVIDIA's [Nemotron 3.5 ASR streaming](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) speech recognition model using the NVIDIA NeMo framework on Baseten.

Nemotron 3.5 ASR is a 600M-parameter, multilingual (40 language-locales), real-time streaming model built on a Cache-Aware FastConformer-RNNT architecture with prompt-based language conditioning. This recipe does a full fine-tune from the base `.nemo` checkpoint using NeMo's `speech_to_text_finetune.py`, following NVIDIA's [official fine-tuning recipe](https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr). It uses the small public AN4 corpus as a stand-in so it runs end-to-end quickly.

The job runs on the `nvcr.io/nvidia/nemo:26.02` container — the last consolidated NeMo Framework image that bundles the ASR/TTS collection with a pre-tested CUDA/numba/lightning/kaldialign stack (the newer `26.06` image is Megatron/LLM-only, and the collection now lives in [`NVIDIA-NeMo/Speech`](https://github.com/NVIDIA-NeMo/Speech)). `run.sh` uses the container's bundled NeMo when it has the streaming-prompt recipe, and otherwise clones it and prepends it to `PYTHONPATH` so the recipe code runs against the container's already-installed, correctly CUDA-linked deps. The base checkpoint is delivered read-only via the [Baseten Delivery Network (BDN)](https://docs.baseten.co/training/concepts/storage) — declared as a `WeightsSource` in `config.py` — so it's on local disk before training starts and never costs billed GPU time to download.

**Resources:** 1 node, 1x H100 GPU

## Prerequisites

1. [Create a Baseten account](https://baseten.co/signup) if you don't already have one.
2. Add a Hugging Face access token as the Baseten secret `hf_access_token` (used to download the gated base checkpoint). See [secrets](https://app.baseten.co/settings/secrets).
3. Install the Truss CLI:
   ```bash
   # pip
   pip install -U truss
   # or uv
   uv add truss
   ```

## Getting Started

Initialize the example, navigate into the directory, and push the training job:

```bash
truss train init --examples nemotron-asr-nemo
cd nemotron-asr-nemo
truss train push config.py
```

## Using Your Own Data

The example trains on AN4 purely as a smoke test. To fine-tune on your own language, domain, or accent, replace `prepare_data.py` with your own manifest builder that emits NeMo JSON-lines manifests:

```json
{"audio_filepath": "/abs/path/clip.wav", "duration": 4.27, "text": "Reference transcript.", "lang": "en-US", "target_lang": "en-US"}
```

Two details matter most:

- **Every clip needs a `target_lang` tag** matching a locale the model recognizes (e.g. `en-US`, `es-ES`, `el-GR`, `bg-BG`). This drives the model's prompt-based language conditioning and is unforgiving of mismatched labels.
- **Match the base model's text style** — punctuated, properly-cased transcripts, since that's what the model produces.

For larger datasets, point the trainer at tarred NeMo/Lhotse shards and drive training with a fixed step budget (`trainer.max_steps`) rather than epochs. When specializing on a few languages in the multilingual model, blend in a slice of the other languages ("replay") to avoid eroding them.

For real, **frozen** datasets (e.g. tarred shards on Hugging Face, S3, or GCS), mount them through BDN the same way as the base checkpoint rather than downloading them in `run.sh` — add another `WeightsSource` in `config.py` and read from the mount path:

```python
weights=[
    WeightsSource(source=f"hf://{INIT_MODEL}", mount_location=INIT_MODEL_MOUNT, auth_secret_name="hf_access_token"),
    WeightsSource(source="s3://my-bucket/asr-shards", mount_location="/app/data"),
]
```

The AN4 smoke test stays as an in-container download because it's tiny and needs on-the-fly preprocessing (`.sph`→`.wav` + manifest building), which a read-only BDN mount can't do in place.

## Evaluate

Evaluate at your deployment latency on held-out data using NeMo's streaming inference script. The lowest-latency setting (`att_context_size=[56,0]`, 80ms chunk, 0ms look-ahead) is the most demanding, honest condition:

```bash
python ${NEMO_DIR}/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
    model_path=<path-to-finetuned.nemo> \
    dataset_manifest=<path-to-test_manifest.json> \
    target_lang=auto \
    att_context_size="[56,3]" \
    decoder_type=rnnt \
    pad_and_drop_preencoded=true \
    batch_size=8 \
    strip_lang_tags=false
```

The same checkpoint covers the whole latency/accuracy spectrum — pick the operating point at inference time via `att_context_size`, no retraining required.
