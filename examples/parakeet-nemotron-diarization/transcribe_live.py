# /// script
# requires-python = ">=3.11"
# dependencies = ["websockets==15.0.1"]
# ///
"""Stream a 16 kHz mono PCM16 WAV to the Parakeet + Nemotron 3 Diarization preset.

Usage: uv run --python 3.11 transcribe_live.py meeting.wav [max_speakers]
Reads BASETEN_API_KEY and BASETEN_MODEL_ID from the environment.
"""

import asyncio
import base64
import json
import os
import sys
import wave

from websockets.asyncio.client import connect

SAMPLE_RATE = 16_000
FRAME_BYTES = SAMPLE_RATE // 10 * 2  # 100 ms of PCM16.


def load_audio(path: str) -> bytes:
    with wave.open(path, "rb") as audio:
        shape = (audio.getframerate(), audio.getnchannels(), audio.getsampwidth())
        if shape != (SAMPLE_RATE, 1, 2):
            raise ValueError("Use an uncompressed 16 kHz mono PCM16 WAV file.")
        return audio.readframes(audio.getnframes())


async def transcribe(path: str, max_speakers: int) -> None:
    pcm = load_audio(path)
    model_id = os.environ["BASETEN_MODEL_ID"]
    url = f"wss://model-{model_id}.api.baseten.co/environments/production/websocket"
    headers = {"Authorization": f"Bearer {os.environ['BASETEN_API_KEY']}"}
    # A deployment scaled to zero holds the handshake open until a replica is ready.
    async with connect(
        url, additional_headers=headers, open_timeout=600, max_size=None
    ) as ws:
        await ws.send(json.dumps({"session_id": "demo", "max_speakers": max_speakers}))

        async def send_audio() -> None:
            loop = asyncio.get_running_loop()
            start = loop.time()
            for offset in range(0, len(pcm), FRAME_BYTES):
                chunk = pcm[offset : offset + FRAME_BYTES]
                frame = {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
                await ws.send(json.dumps(frame))
                next_send = start + (offset + len(chunk)) / (SAMPLE_RATE * 2)
                await asyncio.sleep(max(0.0, next_send - loop.time()))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        sender = asyncio.create_task(send_audio())
        printed = 0
        async for message in ws:
            frame = json.loads(message)
            if frame.get("type") == "error":
                raise RuntimeError(f"Server error: {frame['error']}")
            segments = frame.get("segments", [])
            for turn in segments[printed:]:
                flag = " (overlap)" if turn.get("overlap") else ""
                span = f"{turn['start']:6.2f}-{turn['end']:6.2f}"
                print(f"[{turn['speaker']} {span}]{flag} {turn['text']}", flush=True)
            printed = max(printed, len(segments))
            for tail in frame.get("partial", []):
                print(f"    … {tail['speaker']}: {tail['text']}", flush=True)
            if frame.get("is_final"):
                print(
                    f"\n{frame['num_speakers']} speaker(s), "
                    f"{frame['processed_s']:.1f} s processed"
                )
                break
        await sender


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    asyncio.run(transcribe(sys.argv[1], int(sys.argv[2]) if len(sys.argv) == 3 else 8))
