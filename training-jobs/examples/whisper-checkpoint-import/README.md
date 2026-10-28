# Import and deploy a fine-tuned Whisper checkpoint

Bring an existing Hugging Face Whisper checkpoint into Baseten Training, then deploy
it from the Training dashboard. The import runs on CPU and takes no optimizer steps.
You do not need access to any private Baseten repository or a local Hugging Face token.

## Requirements

- A Baseten workspace with Training access and a team that will own the deployment.
- Python 3, the [Baseten CLI](https://docs.baseten.co/reference/cli/overview) with
  `baseten train` support, and [uv](https://docs.astral.sh/uv/getting-started/installation/).
  The CLI uses uv to run its Python Training tooling.
- A complete Hugging Face checkpoint matching Whisper **large-v2**, **large-v3**, or
  **large-v3-turbo**, with full safetensors weights and tokenizer files. Merge LoRA
  adapters into the base model before importing. Small, medium, distilled, and
  checkpoints with changed architecture or vocabulary are rejected.
- A compatible serving template available to your workspace. Successful import does
  not guarantee that every family, GPU, or serving mode is available.

## 1. Get the example and authenticate

```bash
git clone https://github.com/basetenlabs/ml-cookbook.git
cd ml-cookbook/training-jobs/examples/whisper-checkpoint-import
baseten auth login --with-api-key --profile whisper-import --no-switch
```

Enter an API key for the destination workspace when prompted. The named profile
keeps this example's commands pointed at that workspace without changing your default.
This directory is standalone: you may copy it out of the repository.

## 2. Import your checkpoint

For public weights:

```bash
bash import_whisper_checkpoint.sh \
  --profile whisper-import \
  --team <team-name> \
  --repo openai/whisper-large-v3
```

For private or gated weights, save a read token in the destination workspace's
[Secrets settings](https://app.baseten.co/settings/secrets), or reuse an existing
secret that can read the repository. Pass its **name**, never its token value:

```bash
bash import_whisper_checkpoint.sh \
  --profile whisper-import \
  --team <team-name> \
  --repo <your-hf-org/your-whisper-checkpoint> \
  --revision <full-commit-sha> \
  --hf-secret-name hf_access_token
```

The job receives that workspace secret as `HF_TOKEN`. It resolves and downloads the
repository inside Baseten; no local `HF_TOKEN` is read or forwarded. Secret discovery
is not automatic: choose an existing secret with access to your source repository.

Use a full commit SHA for repeatable imports and the default job name. The public
large-v3 example defaults to commit `06f233fe06e710322aca913c1bc4249a0d71fce1`.
Omitting `--revision` for another repository selects `main`, resolved to its latest
commit when the job starts, and requires an explicit `--job-name`.
To import a branch or tag, supply `--job-name` explicitly: the SHA is resolved inside
the job, so it is unavailable for naming at submission time. The job then downloads
every file from that resolved commit and records it in its provenance.

Run `bash import_whisper_checkpoint.sh --help` for all options. `--project` defaults
to the HF model ID; `--job-name` defaults to `<HF-model-id>-<sha6>` with slashes
replaced by hyphens. For example, `example/whisper` at commit `abcdef...` creates
project `example/whisper` and job `example-whisper-abcdef`. Both names are customizable:

```bash
bash import_whisper_checkpoint.sh \
  --profile whisper-import \
  --repo <your-hf-org/your-whisper-checkpoint> \
  --hf-secret-name hf_access_token \
  --revision <full-commit-sha> \
  --project my-whisper-imports \
  --job-name candidate-1
```

`--checkpoint-name` defaults to `checkpoint-0`.
Without `--profile` or `--team`, the CLI uses its current workspace and default team.
The selected team must contain the HF secret and will own the checkpoint and deployment.

The import uses 4 CPU cores and 16 GiB of memory, downloads the selected weights and
tokenizer, and saves a Training checkpoint. Weights are copied unchanged. The
`config.json` base-model identifier is normalized to the detected Whisper family;
`import_provenance.json` records the original HF repository and immutable commit.
Source repository names and revisions appear in job logs and checkpoint metadata.
No model repository is created or uploaded to Hugging Face.

## 3. Follow logs, then configure deployment in the UI

1. The script tails Training logs in your terminal and waits until checkpoint sync
   is **COMPLETE**. Sync can continue after the import process exits.
2. When prompted, open the job in the [Training dashboard](https://app.baseten.co/training/).
3. Select **Deploy** on the checkpoint and choose an available Batch or Real-time
   template, a distinct test model name, and supported hardware.
4. Enter the model ID and deployment ID in the terminal. The script tails deployment
   logs until ready. Then send an audio request to verify transcription.

The script does not launch deployment itself. Press Enter at the model-ID prompt,
or pass `--import-only`, to finish after checkpoint sync. In a noninteractive shell
it prints the resume command instead of prompting. Ctrl-C detaches log monitoring
without stopping the cloud job or deployment. Import plus checkpoint sync has a
one-hour timeout. Deployment monitoring gets a separate hour after you enter the IDs;
time spent configuring deployment in the UI does not count against it.

Resume monitoring without importing again:

```bash
bash import_whisper_checkpoint.sh --profile whisper-import --monitor-job <job-id>
```

Or attach directly to a deployment configured in the UI:

```bash
bash import_whisper_checkpoint.sh --profile whisper-import \
  --monitor-job <job-id> --model-id <model-id> --deployment-id <deployment-id>
```

Include the same `--checkpoint-name` when resuming a non-default checkpoint.

The platform prepares a TensorRT engine from the imported checkpoint. This example
does not run a separate engine-build workflow or deploy automatically. If no compatible
template appears, contact Baseten with the model family and serving mode you need.
Do not modify a checkpoint after it syncs.

Start with a separate test model. Check streaming partials, final transcripts,
top-N output if used, latency, and scaling cost before migrating an existing service.
This example does not recreate a custom Chain, add per-language checkpoint routing,
or configure diarization. Deactivate test deployments when finished.

See [serving checkpoints](https://docs.baseten.co/training/deployment) for deployment
details.

## Local checks

```bash
uv run --no-project --python 3.12 --with truss==0.18.16 \
  --with-requirements requirements.txt python -m unittest discover -s . -p 'test_*.py'
bash -n import_whisper_checkpoint.sh
```

These checks use mocked HF downloads and do not submit a job or deploy a model.
The runtime's `certifi` minimum matches the cookbook's root dependency floor.
