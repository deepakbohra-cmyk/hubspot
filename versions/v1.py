# isme phrases nikale h 
import json
from src.transcribe.transcription import transcription
from src.analytics.text_analytics import extract_effective_phrases, extract_repeated_words

def main():
    calls = transcription()
    result = []

    for idx, call in enumerate(calls):
        text = call.get("transcription", "").strip()

        if not text:
            # Skip empty transcripts
            print(f"Call {idx} is empty. Skipping...")
            continue


        # Extract effective phrases
        phrases = extract_effective_phrases(text)

        # Extract repeated words
        repeated_words = extract_repeated_words(text, topn=5)

        # Store everything for this call
        result.append({
            "call_id": call.get("id", idx),
            "transcription": text,
            "effective_phrases": phrases,
            "repeated_words": repeated_words
        })


    with open("call_analysis.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print("✅ All call analyses saved to 'call_analysis.json'")

if __name__ == "__main__":
    main()
