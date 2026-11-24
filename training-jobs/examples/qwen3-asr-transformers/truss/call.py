"""Transcribe audio with a deployed Qwen3-ASR checkpoint via /v1/audio/transcriptions.

Usage:
    BASETEN_API_KEY=... MODEL_ID=... DEPLOYMENT_ID=... python call.py
    AUDIO_PATH=clip.wav python call.py        # local file
    AUDIO_URL=https://... python call.py      # remote file (downloaded first)
    PRINT_TRANSCRIPT=1 python call.py        # opt in to writing text to stdout
    LANGUAGE=es python call.py                # force a language; omit to auto-detect
"""

import os
import urllib.request
from http.client import HTTPException
from pathlib import Path
from urllib.parse import urlsplit

from openai import APIError, OpenAI

SERVED_MODEL_NAME = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_AUDIO_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
)


def audio_file() -> Path | tuple[str, bytes]:
    if path_value := os.getenv("AUDIO_PATH"):
        path = Path(path_value)
        if not path.is_file():
            raise SystemExit("AUDIO_PATH must point to an existing audio file.")
        return path
    url = os.getenv("AUDIO_URL", DEFAULT_AUDIO_URL)
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            raise ValueError("Expected an HTTP(S) URL.")
        suffix = Path(parsed.path).suffix or ".wav"
    except ValueError:
        raise SystemExit("AUDIO_URL must be a valid HTTP(S) URL.") from None
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return f"audio{suffix}", response.read()
    except (OSError, HTTPException, ValueError):
        raise SystemExit("Audio download failed.") from None


def main() -> None:
    model_id = os.environ["MODEL_ID"]
    deployment_id = os.environ.get("DEPLOYMENT_ID")
    # Target a specific deployment, or the production environment by default.
    route = (
        f"deployment/{deployment_id}" if deployment_id else "environments/production"
    )
    optional = {}
    if language := os.getenv("LANGUAGE"):
        optional["language"] = language  # ISO 639-1 code, e.g. "en", "es", "zh"
    try:
        with OpenAI(
            api_key=os.environ["BASETEN_API_KEY"],
            base_url=f"https://model-{model_id}.api.baseten.co/{route}/sync/v1",
        ) as client:
            transcription = client.audio.transcriptions.create(
                model=SERVED_MODEL_NAME, file=audio_file(), **optional
            )
    except APIError as exc:
        status = getattr(exc, "status_code", None)
        raise SystemExit(
            f"Transcription failed (HTTP {status})."
            if status
            else "Transcription request failed."
        ) from None

    print("Transcription completed.")
    if os.getenv("PRINT_TRANSCRIPT", "").strip().lower() in {"1", "true"}:
        print(transcription.text)


if __name__ == "__main__":
    main()
