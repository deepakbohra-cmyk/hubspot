import json

def transcription():
    with open("data/data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

