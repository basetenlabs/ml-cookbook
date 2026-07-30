"""Minimal client for a fine-tuned OmniVoice deployment on Baseten.

OmniVoice is a zero-shot voice-clone model: the voice identity at inference
comes from a short reference clip, not a speaker name baked into the weights.
Fine-tuning adapts the model toward your speaker/domain, but the most stable
way to reproduce the target voice is still voice cloning with a reference clip
from your training corpus (see the OmniVoice README). Text-only requests fall
back to "auto voice", which picks a random voice each call.

For the cloned voice to match your fine-tuned speaker, set REF_AUDIO_PATH /
REF_TEXT below to a clip (+ transcript) from your training corpus. When they
are unset, this client falls back to pulling one LJ Speech clip via the HF
datasets-server so it runs out of the box against the demo fine-tune.

Usage:
    export BASETEN_API_KEY=...
    # Edit MODEL_ID and DEPLOYMENT_ID below, then:
    python truss/call.py
"""

import base64
import os
from pathlib import Path

import httpx

BASETEN_API_KEY = os.getenv("BASETEN_API_KEY")
MODEL_ID = "..."
DEPLOYMENT_ID = "..."

# Reference voice clip. Set REF_AUDIO_PATH to a wav from your training corpus
# and REF_TEXT to its transcript; leave both as None to fall back to an
# LJ Speech clip fetched from the HF datasets-server.
REF_AUDIO_PATH = None  # e.g. "datasets/my_speaker/utt_0001.wav"
REF_TEXT = None  # required only when REF_AUDIO_PATH is set

# Which LJ Speech row to use as the reference when auto-fetching.
DATASET_REPO = "SeanSleat/lj_speech"
DATASET_CONFIG = "main"
DATASET_SPLIT = "train"
REF_INDEX = 0

API_BASE = (
    f"https://model-{MODEL_ID}.api.baseten.co/deployment/{DEPLOYMENT_ID}/sync"
)


def load_reference() -> tuple[bytes, str]:
    """Return (wav_bytes, transcript) for the reference voice clip."""
    if REF_AUDIO_PATH:
        if not REF_TEXT:
            raise RuntimeError("Set REF_TEXT to the transcript of REF_AUDIO_PATH.")
        path = Path(REF_AUDIO_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Reference audio not found: {path}")
        return path.read_bytes(), REF_TEXT

    # Fall back to one clip + transcript from the LJ Speech datasets-server.
    rows_url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={DATASET_REPO}&config={DATASET_CONFIG}"
        f"&split={DATASET_SPLIT}&offset={REF_INDEX}&length=1"
    )
    with httpx.Client(timeout=60.0) as client:
        row = client.get(rows_url).raise_for_status().json()["rows"][0]["row"]
        transcript = row.get("normalized_text") or row["text"]

        cache_path = Path(__file__).parent / "lj_reference.wav"
        if not cache_path.exists():
            audio_src = row["audio"][0]["src"]
            cache_path.write_bytes(client.get(audio_src).raise_for_status().content)
            print(f"Cached reference clip {row['id']} -> {cache_path}")
        return cache_path.read_bytes(), transcript


def main() -> None:
    if not BASETEN_API_KEY:
        raise RuntimeError("Set BASETEN_API_KEY in the environment.")
    if MODEL_ID == "..." or DEPLOYMENT_ID == "...":
        raise RuntimeError("Edit MODEL_ID and DEPLOYMENT_ID in call.py.")

    ref_audio, ref_text = load_reference()
    ref_audio_b64 = base64.b64encode(ref_audio).decode("utf-8")

    text = "Wow! Isn't fine-tuning this model amazing?"
    output_path = "output.wav"

    response = httpx.post(
        f"{API_BASE}/v1/audio/speech",
        headers={
            "Authorization": f"Api-Key {BASETEN_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "input": text,
            "language": "English",
            "response_format": "wav",
            "ref_audio": f"data:audio/wav;base64,{ref_audio_b64}",
            "ref_text": ref_text,
        },
        timeout=300.0,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Speech request failed ({response.status_code}): {response.text}")

    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Wrote {output_path} ({len(response.content):,} bytes)")


if __name__ == "__main__":
    main()
