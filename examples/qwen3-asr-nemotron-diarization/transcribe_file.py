# /// script
# requires-python = ">=3.11"
# dependencies = ["requests"]
# ///
"""Transcribe one recording with speaker labels on the Nemotron 3 + Qwen3-ASR batch preset.

Usage: uv run --python 3.11 transcribe_file.py https://example.com/meeting.wav [language]
Reads BASETEN_API_KEY and BASETEN_MODEL_ID from the environment.
"""

import os
import sys

import requests


def main(audio_url: str, language: str) -> None:
    model_id = os.environ["BASETEN_MODEL_ID"]
    url = f"https://model-{model_id}.api.baseten.co/environments/production/predict"
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {os.environ['BASETEN_API_KEY']}"},
        json={
            "transcription_input": {
                "audio": {"url": audio_url},
                "latency": "offline",
                "language": language,
            }
        },
        timeout=300,
    )
    response.raise_for_status()
    result = response.json()
    print("language:", result["language"]["used"])
    for segment in result["segments"]:
        span = f"{segment['start']:7.2f}-{segment['end']:7.2f}"
        print(f"[{segment['speaker']} {span}] {segment['text']}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else "auto")
