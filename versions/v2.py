import os
import re
import json
from collections import Counter
import google.generativeai as genai

genai.configure(api_key="AIzaSyC5nxTcb0BGKNgLo1DnNDj2yuQsVoMlQLc")

def load_calls(row_numbers, data_file="data.json"):
    texts = []
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for row in row_numbers:
        for d in data:
            if d["did_number"] == row:
                texts.append(d["transcription"])
    return texts


# Load good & bad calls from JSON
good_calls = load_calls([918062463239, 918069245483])
bad_calls  = load_calls([918062463218, 918062757075])

def clean(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())

def get_frequent_words(text_list, top_n=30):
    all_words = []
    for t in text_list:
        all_words.extend(clean(t).split())
    return Counter(all_words).most_common(top_n)

good_words = get_frequent_words(good_calls)
bad_words = get_frequent_words(bad_calls)


def extract_phrases(text_list, n=3):
    phrases = Counter()
    for t in text_list:
        words = clean(t).split()
        for i in range(len(words)-n+1):
            phrases[" ".join(words[i:i+n])] += 1
    return phrases.most_common(50)

good_phrases = extract_phrases(good_calls)
bad_phrases = extract_phrases(bad_calls)


# Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")

def analyze_new_call(new_call_text):
    
    print("Extracting features...")
    new_call_words = get_frequent_words([new_call_text])
    new_call_phrases = extract_phrases([new_call_text])

    prompt = f"""
You are analyzing call quality.

GOOD CALL WORDS:
{good_words}

BAD CALL WORDS:
{bad_words}

GOOD CALL PHRASES:
{good_phrases}

BAD CALL PHRASES:
{bad_phrases}

NEW CALL WORDS:
{new_call_words}

NEW CALL PHRASES:
{new_call_phrases}

Based on these, tell:
1. What this call did well
2. What it needs to improve
3. Missing effective phrases
4. Tone and structure feedback
"""

    print("Generating feedback...")
    response = model.generate_content(prompt)
    return response.text


if __name__ == "__main__":
    new_call_text = "Hello.... your long call transcription ..."
    feedback = analyze_new_call(new_call_text)

    print("\n------ FEEDBACK ------\n")
    print(feedback)
