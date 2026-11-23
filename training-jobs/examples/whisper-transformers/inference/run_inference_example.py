import os
from pathlib import Path

from openai import APIError, OpenAI


def main() -> None:
    base_url = (
        f"https://model-{os.environ['BASETEN_MODEL_ID']}.api.baseten.co"
        f"/deployment/{os.environ['BASETEN_DEPLOYMENT_ID']}/sync/v1"
    )
    audio_path = Path(
        os.getenv("AUDIO_PATH", Path(__file__).with_name("speech-94649.mp3"))
    )
    try:
        with OpenAI(api_key=os.environ["BASETEN_API_KEY"], base_url=base_url) as client:
            with audio_path.open("rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model=os.environ["SERVED_MODEL_NAME"], file=audio_file
                )
    except APIError as exc:
        status = getattr(exc, "status_code", None)
        raise SystemExit(
            f"Transcription failed (HTTP {status})."
            if status
            else "Transcription request failed."
        ) from None

    print("Transcription completed.")
    if os.getenv("PRINT_TRANSCRIPT") == "1":
        print(transcription.text)


if __name__ == "__main__":
    main()
