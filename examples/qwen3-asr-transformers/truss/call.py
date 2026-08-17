import base64
import mimetypes
import os
from pathlib import Path

from openai import OpenAI


MODEL_ID = "..."
DEPLOYMENT_ID = "..."
SERVED_MODEL_NAME = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_AUDIO_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
)


def audio_url() -> str:
    if path_value := os.getenv("AUDIO_PATH"):
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"AUDIO_PATH does not exist: {path}")

        mime_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    return os.getenv("AUDIO_URL", DEFAULT_AUDIO_URL)


def parse_content(content: str) -> dict[str, str]:
    prefix = "language "
    separator = "<asr_text>"
    if content.startswith(prefix) and separator in content:
        language, text = content[len(prefix) :].split(separator, maxsplit=1)
        return {"language": language, "text": text}
    return {"raw_output": content}


def main() -> None:
    client = OpenAI(
        api_key=os.environ["BASETEN_API_KEY"],
        base_url=(
            f"https://model-{MODEL_ID}.api.baseten.co/deployment/"
            f"{DEPLOYMENT_ID}/sync/v1"
        ),
    )

    response = client.chat.completions.create(
        model=SERVED_MODEL_NAME,
        stream=False,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": audio_url()},
                    }
                ],
            }
        ],
    )
    content = response.choices[0].message.content or ""
    print(parse_content(content))


if __name__ == "__main__":
    main()
