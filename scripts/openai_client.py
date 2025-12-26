# /// script
# dependencies = [
#     "openai>=2.14.0",
# ]
# ///

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

transcription = client.audio.transcriptions.create(
    model="omniASR_CTC_300M_v2", file=open("harvard.wav", "rb"), language="en"
)

print(transcription)
