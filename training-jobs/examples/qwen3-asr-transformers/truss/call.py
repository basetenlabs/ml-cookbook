"""Transcribe audio with a deployed Qwen3-ASR checkpoint via /v1/audio/transcriptions.

Usage:
    BASETEN_API_KEY=... MODEL_ID=... DEPLOYMENT_ID=... python call.py
    AUDIO_PATH=clip.wav python call.py        # local file
    AUDIO_URL=https://... python call.py      # remote file (downloaded first)
    LANGUAGE=es python call.py                # force a language; omit to auto-detect
"""

import os
import tempfile
import urllib.request
from pathlib import Path

from openai import OpenAI

SERVED_MODEL_NAME = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_AUDIO_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
)


def audio_file() -> Path:
    if path_value := os.getenv("AUDIO_PATH"):
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"AUDIO_PATH does not exist: {path}")
        return path
    url = os.getenv("AUDIO_URL", DEFAULT_AUDIO_URL)
    suffix = Path(url.split("?")[0]).suffix or ".wav"
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    with urllib.request.urlopen(url) as response:
        handle.write(response.read())
    handle.close()
    return Path(handle.name)


def main() -> None:
    model_id = os.environ["MODEL_ID"]
    deployment_id = os.environ.get("DEPLOYMENT_ID")
    # Target a specific deployment, or the production environment by default.
    route = (
        f"deployment/{deployment_id}" if deployment_id else "environments/production"
    )
    client = OpenAI(
        api_key=os.environ["BASETEN_API_KEY"],
        base_url=f"https://model-{model_id}.api.baseten.co/{route}/sync/v1",
    )

    optional = {}
    if language := os.getenv("LANGUAGE"):
        optional["language"] = language  # ISO 639-1 code, e.g. "en", "es", "zh"
    with audio_file().open("rb") as handle:
        transcription = client.audio.transcriptions.create(
            model=SERVED_MODEL_NAME, file=handle, **optional
        )
    print(transcription.text)


if __name__ == "__main__":
    main()
