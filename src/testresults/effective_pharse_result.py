import json

def effective_phrase_result():
    with open("data/call_analysis.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data