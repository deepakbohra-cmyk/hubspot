import os
import re
import json
from collections import Counter
import google.generativeai as genai
from nltk.corpus import stopwords
import nltk
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Add custom stopwords
hindi_stopwords = {"हां","हाँ","नहीं","मत","क्यों","क्या","कब","कैसे","ठीक","थोड़ा","ज़्यादा","कम","आप"}
hinglish_stopwords = {"haan","nahi","thik","sir","please","ok","ji","haanji"}

stop_words |= hindi_stopwords | hinglish_stopwords


# ------------------------------------------------------------------------------------
# LOAD GOOD & BAD CALLS FROM JSON
# ------------------------------------------------------------------------------------
def load_calls(did_numbers, data_file="data/data.json"):
    texts = []
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for d in data:
        if d["did_number"] in did_numbers:
            texts.append(d["transcription"])
    return texts


good_calls = load_calls([918062463239, 918069245483])
bad_calls  = load_calls([918062463218, 918062757075])


# ------------------------------------------------------------------------------------
# TEXT PROCESSING
# ------------------------------------------------------------------------------------
def clean(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())

def get_frequent_words(text_list, top_n=30):
    words = []
    for t in text_list:
        words.extend(clean(t).split())
    return Counter(words).most_common(top_n)

def extract_phrases(text_list, n=3):
    phrases = Counter()
    for t in text_list:
        words = clean(t).split()
        for i in range(len(words) - n + 1):
            phrases[" ".join(words[i:i+n])] += 1
    return phrases.most_common(50)


good_words = get_frequent_words(good_calls)
bad_words = get_frequent_words(bad_calls)

good_phrases = extract_phrases(good_calls)
bad_phrases = extract_phrases(bad_calls)



# ------------------------------------------------------------------------------------
# GEMINI MODEL
# ------------------------------------------------------------------------------------
model = genai.GenerativeModel("gemini-2.5-flash")


# ------------------------------------------------------------------------------------
# STEP 1 — TRANSCRIBE AUDIO USING GEMINI
# ------------------------------------------------------------------------------------
def transcribe_audio(audio_path):
    print("Transcribing audio using Gemini...")

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    result = model.generate_content(
        contents=[
            {"mime_type": "audio/wav", "data": audio_data}
        ],
        generation_config={"temperature": 0.1}
    )

    return result.text



# ------------------------------------------------------------------------------------
# STEP 2 — CLASSIFY CALL AS GOOD / BAD
# ------------------------------------------------------------------------------------
def classify_call(transcription):
    prompt = f"""
You are an expert call quality evaluator.

GOOD CALL WORDS: {good_words}
BAD CALL WORDS: {bad_words}
GOOD PHRASES: {good_phrases}
BAD PHRASES: {bad_phrases}

NEW CALL TRANSCRIPTION:
{transcription}

Tell only:
1. Is this a GOOD CALL or BAD CALL?
2. Why? (short reasoning)
    """

    result = model.generate_content(prompt)
    return result.text



# ------------------------------------------------------------------------------------
# STEP 3 — ANALYSIS (your previous logic)
# ------------------------------------------------------------------------------------
def analyze_call(transcription):
    new_call_words = get_frequent_words([transcription])
    new_call_phrases = extract_phrases([transcription])

    prompt = f"""
GOOD CALL WORDS: {good_words}
BAD CALL WORDS: {bad_words}
GOOD PHRASES: {good_phrases}
BAD PHRASES: {bad_phrases}

NEW CALL WORDS: {new_call_words}
NEW CALL PHRASES: {new_call_phrases}

Give:
1. What this call did well
2. What needs improvement
3. Missing effective phrases
4. Tone and structure feedback
"""

    result = model.generate_content(prompt)
    return result.text



# ------------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    AUDIO_PATH = "audio1.mp3"   # <-- PUT YOUR FILE HERE

    transcription = transcribe_audio(AUDIO_PATH)

    print("\n------ TRANSCRIPTION ------\n")
    print(transcription)

    print("\n------ CALL TYPE ------\n")
    print(classify_call(transcription))

    print("\n------ DETAILED ANALYSIS ------\n")
    print(analyze_call(transcription))
