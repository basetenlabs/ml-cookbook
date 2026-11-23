# Speech data handling: inference and training

[Baseten Security Practices](https://www.baseten.co/security-practices/#zero-data-retention)
is the source of truth for Zero Data Retention (ZDR) and data retention. This
page explains how the cookbook examples relate to that policy; it does not
certify any deployment or extend the policy to a whole application.

## Inference requests

- Baseten's ZDR commitment covers inference inputs and outputs on Model APIs and
  Dedicated Inference, subject to the policy's customer-controlled factors.
  For speech, payloads include audio, text prompts, reference clips and their
  transcripts, generated audio, and interim and final transcripts.
- The Qwen3-ASR **Batch** example calls the synchronous transcription endpoint;
  it does not use Baseten **Async Inference**. The latter is elective and can
  queue model inputs for up to 72 hours. The policy says it does not store
  model outputs.
- Keep payloads out of application logs, traces, error messages, and exports.
  Use public or synthetic samples for smoke tests. The example clients report
  status or counts by default; `PRINT_TRANSCRIPT=1` sends speech content to
  stdout and `OUTPUT_WAV=path.wav` saves synthesized speech locally. Client
  output, URL-hosted inputs, and third-party integrations have their own
  retention and access controls.
- Before using sensitive audio, verify the selected server image, deployment
  settings, and dependencies with synthetic inputs on success, error, and
  disconnect. Check logs, trace exports, and any files or caches written by
  the runtime. A passing client-side check alone does not verify server behavior.

## Training jobs and Train-to-Serve

The inference ZDR commitment is not a requirement to discard training
artifacts. Training Jobs recipes intentionally use checkpoints and may use
caches to reuse data. Set a retention and access policy for each copy, including:

| Artifact | What to check |
| --- | --- |
| Job upload and source data | If a recipe bundles data (for example, OmniVoice's local dataset mode), audio and transcripts are part of the uploaded job context. Keep private datasets out of Git and review archive exclusions and storage access before submission. |
| Training cache and preprocessing | Prepared manifests, clips, or tokens can be cached depending on the recipe and mount. Review the [cache scope and cleanup controls](https://docs.baseten.co/training/concepts/cache). |
| Checkpoints and deployed weights | A [checkpoint](https://docs.baseten.co/training/concepts/checkpoints) without an expiry stays until deleted; expiry does not remove an ever-deployed checkpoint. [Job and project deletion](https://docs.baseten.co/training/management) do not remove deployed checkpoints. Account for any deployed or cached weight copies separately. Treat a fine-tuned voice checkpoint, including any stored speaker embedding, as a sensitive derived artifact when the source voice is sensitive. |
| External tracking and exports | Only enable destinations approved for the workload. Hugging Face exports, hosted tracking, and downloaded artifacts have independent access and deletion controls. |

The policy's 30-day post-subscription retrieval window is not a per-job or
per-checkpoint TTL.
