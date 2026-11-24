# Shared review rules

These rules are shared across Baseten's voice, speech-to-text and model repos; the source of truth is `basetenlabs/model-registry/.greptile/rules.md`, and this copy is adapted for ml-cookbook. Repo-specific rules can be added in their own `.greptile/` folders. For each finding, quote the offending line, say which rule it breaks, and suggest the fix.

**How these apply here:** ml-cookbook holds public training and inference examples, not production serving code. Apply the rules to new or changed lines in a PR; do not ask authors to fix untouched existing examples. Where a rule says "ships to customers", read it as "an example readers copy and run".

## 1. New Hugging Face items go under `baseten/`, not `baseten-admin/`

Baseten is migrating its Hugging Face repos from the `baseten-admin` namespace to `baseten`. Everything new goes under `baseten/`.

**Flag:** any new or changed reference to a Hugging Face repo in the `baseten-admin` namespace: `huggingface.co/baseten-admin/...` URLs, `repo_id="baseten-admin/..."`, `baseten-admin/<repo>` strings in Python, YAML (`config.yaml`, `model_cache`, presets), JSON, shell scripts and docs. This includes code, scripts or workflows that create, upload or push a repo, model, engine or dataset to `baseten-admin`.

**Why:** `baseten-admin` is the old namespace. Many `baseten-admin/...` repos are private or have been deleted, and references to them have broken deploys with 404/401 errors at model load (for example a `snapshot_download` of a Whisper TensorRT engine).

**Ask the author to:**
- Publish any new repo, model, engine or dataset under `baseten/<repo>`, and reference it there.
- Reference an existing Baseten repo by its `baseten/<repo>` path. Hugging Face redirects renamed repos, so check that the `baseten/` path resolves; if the repo has not been moved yet, move it to `baseten/` (or ask the owner to) rather than adding a new `baseten-admin/` reference.
- Point at the upstream public repo instead when the weights are not Baseten-built.
- For push targets in examples and docs where the reader pushes their own fine-tune (`hub_model_id`, `HUB_MODEL_ID`, output repos), use a placeholder such as `<your-hf-username>/<repo>`.

```python
# Bad: new reference or upload target in the old namespace
repo_id = "baseten-admin/whisper-large-v3-trt"
# Good
repo_id = "baseten/whisper-large-v3-trt"
# Good: example where the reader pushes their own model
hub_model_id = "<your-hf-username>/my-finetune"
```

**Not a finding:** existing `baseten-admin/` references on lines the PR does not change, the name of the Hugging Face *account* whose token is used for downloads, and code that detects or skips the old namespace (for example `repo.startswith("baseten-admin/")`).

**Also flag:** a new reference to any Hugging Face repo that customer deployments download from, when nothing in the PR shows the repo is public (or gated with terms accepted). Ask the author to confirm its visibility, because a private repo works on the author's machine and fails for customers.

## 2. No model inputs or outputs in logs (Zero Data Retention)

Baseten's policy is to never log, record or store model inputs or outputs, and not to keep them after real-time processing: https://www.baseten.co/security-practices/#zero-data-retention

**Treat as model input or output:** anything in the request body or response. That includes audio (raw bytes or base64), transcripts and other text, prompts, model parameters, caller-supplied URLs (including pre-signed or signed URLs, their query strings, paths, file names and userinfo), and output text or audio.

In voice and speech pipelines, call content also includes: STT transcripts (partial, final, dropped or "held" segments), STT keyword or hint lists sent by the client, LLM messages, prompts and generated text (including truncated slices like `text[:60]` and filtered or rejected output), text sent to TTS (sentences, lines about to be spoken or skipped), tool or function-call names with their arguments, and request or response bodies to and from model providers (STT, LLM, TTS). A slice, `repr()` or `!r` of any of these is still content.

**Flag:**
- Logging, `print`, f-strings, `repr()` or `str()` of request or response objects, or of fields taken from them, at any level that reaches customer-visible or retained logs (INFO, WARNING, ERROR, and `print`).
- Log prefixes or context fields built from request content, such as a prefix containing the audio URL or its file name.
- Exceptions whose message embeds the payload, and logging raw exception text from code that handles payloads (an HTTP error message often contains the URL; an engine error can quote the prompt or transcript). Check every path that logs the error, including the final or non-streaming path, not just the one the PR touched.
- Provider error handling that logs or raises the response body (`body[:500]`, `resp.text`) or a whole event when an expected field is missing (`event.get("error") or event`). Log the status code, body length or the event's key names instead.
- `logger.exception` or traceback capture that includes local variables holding payloads. With loguru, a sink added without `diagnose=False` prints every frame's variables on `logger.exception`.
- Metrics labels, span attributes, trace events or error-tracker context carrying request content.
- DEBUG logging of content, unless it is clearly gated and approved for debugging. Call it out either way so a reviewer decides.

**Good:** log sizes, counts, durations, sample rates, server-generated request IDs, hashes, status codes and exception class names. A past incident logged `Received whisper_input: audio=...` at INFO with the full base64 audio. It was fixed by redacting the audio to `<N bytes>` and moving the line to DEBUG behind `logger.isEnabledFor(logging.DEBUG)`.

```python
# Bad
logger.info(f"Received whisper_input: {whisper_input}")
logger.warning(f"Probe failed for {audio_url}: {exc}")
logger.info(f"STT: {frame.text}")
logger.info(f"backend: not saying {line[:40]!r}")
logger.info(f"tool call: {name}({', '.join(args)})")
# Good
logger.info("Received request audio=<%d bytes> language=%s", len(audio_b64), lang_code)
logger.warning("Probe failed: %s status=%s", type(exc).__name__, getattr(exc.response, "status_code", None))
```

## 3. Guard server-side fetches of caller-supplied URLs

**Flag:** code that downloads, probes or hands to a subprocess (for example `ffprobe` or `ffmpeg`) a URL that came from the request, without checking it as described under **Good**. This applies only to server-side code (for example a Truss `model.py`); client scripts that pass a URL to a deployed model are not a finding. Also flag `follow_redirects=True` on such a URL without re-checking each hop, and `verify=False` or other disabled TLS verification.

**Why:** a caller-controlled URL can point at internal addresses or cloud metadata endpoints, either directly or through a redirect or a DNS answer that changes between check and connect. Every path that touches the URL needs the same check, including duration probes and fallbacks.

**Good:** allow only `http`/`https`, reject private, loopback, link-local and metadata addresses, connect to the address that was checked, cap or re-check redirects, and fail the request (or fall back to a safe estimate) when the check refuses.

```python
# Bad
async with httpx.AsyncClient(follow_redirects=True) as client:
    r = await client.get(request.audio_url)
# Good
url = check_url(request.audio_url)  # your check: raises on disallowed scheme or address
async with httpx.AsyncClient(follow_redirects=False) as client:
    r = await client.get(url)
```

## 4. No secrets in code, config or output

**Flag:** hard-coded credentials or tokens (Hugging Face `hf_...` tokens, Baseten API keys, cloud keys, registry passwords, webhook URLs with embedded secrets), secrets in `config.yaml` values, test fixtures, notebooks or docs, and code that logs, prints or echoes a secret (including `set -x` in a shell step that handles one).

**Good:** read secrets from the platform's secret store or environment (Truss `secrets:` plus `secrets["hf_access_token"]`, GitHub Actions `${{ secrets.X }}`), pass them as `token=...`, and keep placeholders such as `<HF_TOKEN>` in docs.

```python
# Bad
snapshot_download(repo_id, token="hf_abc123...")
# Good
snapshot_download(repo_id, token=os.environ["HF_TOKEN"])
```

## 5. Pin what runs in CI and what ships

**Flag:**
- Third-party GitHub Actions referenced by tag or branch (`@v4`, `@main`) in a job that has `id-token: write` or uses secrets. Actions must be pinned to a full commit SHA. `chainguard-actions/*` is this repo's approved hardened source; flag other forks or copies of upstream actions.
- Images, model weights or dependencies that ship to customers referenced by a moving tag or branch: `:latest`, an unpinned Hugging Face `revision`, or an unpinned package that the production image pins.
- `trust_remote_code=True` without a pinned `revision` (a commit SHA), because the remote code can change under a deployed model.
- A workflow that can push over an existing production image tag.

**Why:** a moved tag can run new code with the job's credentials, and an unpinned artifact can change what customers run without a code change.

**Good:** `chainguard-actions/*` actions pinned to a full commit SHA, with the version in a comment when known; images pinned by digest or an immutable version tag; model revisions pinned to a commit SHA; test dependencies at the version production runs. Reusable workflows from the same repo may use `@main` when a comment says why.

```yaml
# Bad
- uses: actions/checkout@v4
# Good
- uses: chainguard-actions/actions-checkout@<40-char-commit-sha>  # v4.x.y
```

## 6. Use placeholders for customer names in examples, tests and docs

**Flag:** real customer or prospect names, customer org IDs, customer model or deployment IDs, or customer-provided data (audio, transcripts) added to comments, docs, examples, test fixtures, sample payloads or log messages.

**Why:** these files get copied into public examples, shared with other customers and indexed by tools. Customer identity and data should stay in systems built to hold it.

**Good:** neutral placeholders (`acme`, `customer-a`, `<org_id>`) and synthetic or openly licensed test audio. Do not flag code paths that deliberately encode a customer label by design (for example private artifact naming that hashes the identifier); ask only that tests and docs for them use a placeholder.
