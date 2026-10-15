from openai import OpenAI


## TODO: update these with your deployment details
API_KEY = ""
BASE_URL = "https://model-4q9l8ejq.api.baseten.co/development/sync/v1"
MODEL_NAME = "baseten-admin/whisper-larger-v3-turbo-minirun3"


print("Loading audio file and sending to model for transcription...")
audio_file = open("speech-94649.mp3", "rb")

print("Sending request to model...")
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)
transcription = client.audio.transcriptions.create(model=MODEL_NAME, file=audio_file)
print(transcription)
