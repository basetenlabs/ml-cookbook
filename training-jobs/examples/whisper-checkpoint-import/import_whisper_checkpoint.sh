#!/usr/bin/env bash
# Submit the standalone Whisper checkpoint import example.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: import_whisper_checkpoint.sh [options]
  --repo REPO              Hugging Face repository (default: openai/whisper-large-v3)
  --revision REF           Full commit SHA, or branch/tag with an explicit --job-name
  --hf-secret-name NAME    Existing Baseten workspace secret holding an HF read token
  --profile NAME           Baseten CLI profile for the destination workspace
  --team NAME              Owning team (default: workspace's default team)
  --project NAME           Training project (default: HF model ID)
  --job-name NAME          Training job (default: <HF-model-id>-<sha6>)
  --checkpoint-name NAME   Checkpoint directory (default: checkpoint-0)
  --import-only            Finish after checkpoint sync; skip the deployment-log prompt
  --monitor-job ID         Follow an already-submitted import instead of submitting again
  --model-id ID            Model whose deployment logs to follow after import
  --deployment-id ID       Deployment whose logs to follow after import
  --help                   Show this help
EOF
}

JOB_ENV=()
PUSH_ARGS=()
MONITOR_ARGS=()
PROFILE_ARGS=()
MONITOR_JOB=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --import-only) MONITOR_ARGS+=(--import-only); shift ;;
        --repo|--revision|--hf-secret-name|--profile|--team|--project|--job-name|--checkpoint-name|--monitor-job|--model-id|--deployment-id)
            if [ "$#" -lt 2 ] || [ -z "$2" ] || [[ "$2" == --* ]]; then
                echo "Error: $1 requires a value." >&2
                exit 1
            fi
            case "$1" in
                --repo) JOB_ENV+=("SOURCE_REPO=$2") ;;
                --revision) JOB_ENV+=("SOURCE_REVISION=$2") ;;
                --hf-secret-name) JOB_ENV+=("HF_SECRET_NAME=$2") ;;
                --profile|--team) PUSH_ARGS+=("$1" "$2") ;;
                --project) JOB_ENV+=("TRAINING_PROJECT_NAME=$2") ;;
                --job-name) JOB_ENV+=("TRAINING_JOB_NAME=$2") ;;
                --checkpoint-name) JOB_ENV+=("CHECKPOINT_NAME=$2"); MONITOR_ARGS+=(--checkpoint-name "$2") ;;
                --monitor-job) MONITOR_JOB="$2" ;;
                --model-id|--deployment-id) MONITOR_ARGS+=("$1" "$2") ;;
            esac
            if [ "$1" = --profile ]; then PROFILE_ARGS=(--profile "$2"); fi
            shift 2
            ;;
        *) echo "Error: unknown option $1." >&2; usage >&2; exit 1 ;;
    esac
done

if ! command -v baseten >/dev/null 2>&1; then
    echo "Error: install the Baseten CLI with Training support; see README.md." >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: install Python 3 to monitor the import; see README.md." >&2
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "$MONITOR_JOB" ]; then
    exec python3 "${SCRIPT_DIR}/monitor.py" --job-id "$MONITOR_JOB" \
        ${PROFILE_ARGS[@]+"${PROFILE_ARGS[@]}"} ${MONITOR_ARGS[@]+"${MONITOR_ARGS[@]}"}
fi
OUTPUT=$(mktemp)
trap 'rm -f "$OUTPUT"' EXIT
env ${JOB_ENV[@]+"${JOB_ENV[@]}"} baseten train push \
    --config "${SCRIPT_DIR}/config.py" ${PUSH_ARGS[@]+"${PUSH_ARGS[@]}"} | tee "$OUTPUT"
python3 "${SCRIPT_DIR}/monitor.py" --submission-output "$OUTPUT" \
    ${PROFILE_ARGS[@]+"${PROFILE_ARGS[@]}"} ${MONITOR_ARGS[@]+"${MONITOR_ARGS[@]}"}
