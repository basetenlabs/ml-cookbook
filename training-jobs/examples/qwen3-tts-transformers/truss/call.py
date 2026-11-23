import os
from contextlib import nullcontext
from pathlib import Path

import requests

MODEL_ID = "..."
DEPLOYMENT_ID = "..."


def main() -> None:
    if MODEL_ID == "..." or DEPLOYMENT_ID == "...":
        raise SystemExit("Edit MODEL_ID and DEPLOYMENT_ID in call.py.")

    output_path = os.getenv("OUTPUT_WAV")
    try:
        with requests.post(
            f"https://model-{MODEL_ID}.api.baseten.co/deployment/{DEPLOYMENT_ID}/sync/v1/audio/speech",
            headers={"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"},
            json={
                "input": "Wow! Isn't fine-tuning this model amazing?",
                "voice": "ft_speaker",
                "task_type": "CustomVoice",
                "language": "English",
                "stream": True,
            },
            stream=True,
            timeout=300,
        ) as response:
            if response.status_code != 200:
                raise SystemExit(
                    f"Speech request failed (HTTP {response.status_code})."
                )
            total_bytes = 0
            with (
                Path(output_path).open("xb") if output_path else nullcontext() as output
            ):
                for chunk in response.iter_content(chunk_size=4096):
                    if output is not None:
                        output.write(chunk)
                    total_bytes += len(chunk)
    except requests.RequestException:
        raise SystemExit("Speech request failed.") from None

    print(
        f"Received {total_bytes} bytes of audio."
        + (" WAV saved locally." if output_path else "")
    )


if __name__ == "__main__":
    main()
