import os
import requests

client = requests.Session()

BASETEN_API_KEY = os.getenv('BASETEN_API_KEY')
MODEL_ID = "..."
DEPLOYMENT_ID = "..."

resp = client.post(
    f"https://model-{MODEL_ID}.api.baseten.co/deployment/{DEPLOYMENT_ID}/sync/v1/audio/speech",
    headers={
        "Authorization": f"Api-Key {os.getenv('BASETEN_API_KEY')}",
        "Content-Type": "application/json",
    },
    json={
        "input": (
            "Wow! Isn't fine-tuning this model amazing?"
        ),
        "voice": "ft_speaker",
        "task_type": "CustomVoice",
        "language": "English",
        "stream": True,
    },
    stream=True,
)

if not resp.ok:
    print(f"Status: {resp.status_code}")
    print(f"Headers: {dict(resp.headers)}")
    print(f"Body: {resp.text}")
    resp.raise_for_status()

total_bytes = 0
with open("output.wav", "wb") as f:
    for chunk in resp.iter_content(chunk_size=4096):
        if chunk:
            f.write(chunk)
            total_bytes += len(chunk)

print(f"WAV file written to output.wav ({total_bytes} bytes)")