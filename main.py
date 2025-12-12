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

nltk.download('stopwords')

stop_words = set(stopwords.words("english"))

hindi_stopwords = {
    "हां", "हाँ", "नहीं", "मत", "क्यों", "क्या", "कब", "कैसे",
    "ठीक", "थोड़ा", "ज़्यादा", "कम", "आप", "हम", "वे", "यह", "वह",
    "मेरी", "मेरा", "हमारा", "तुम", "तुम्हारा", "उसका"
}

hinglish_stopwords = {
    "haan", "han", "nahi", "nai", "achha", "acha", "thik", "theek", "sir",
    "madam", "please", "plz", "pls", "bol", "bata", "batau", "kar", "karo",
    "krunga", "krungi", "karunga", "karungi", "mera", "meri", "mere",
    "aap", "tum", "main", "mai", "mujhe", "mujhko", "tera", "teri",
    "haanji", "ji", "haan ji", "ok", "okay", "thoda"
}

stop_words |= hindi_stopwords | hinglish_stopwords

def load_calls(row_numbers, data_file="data/data.json"):
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
    new_call_text = "Hello. Hello. yes actually wo jo aap keh rahe hain na isme bank store aur A plus content isme included nahi hai. So hamara 36000 hai. Hmm. Haan ji. Actually kya hota hai jo hamara A plus content hota hai and the brand store, isme jo used hota hai wo banners aur info graphics hote hain aur per banner aur info graphics ki costing agar jo main baat karu banane ki to 250 to 300 hoti hai. Us according iski costing hoti hai. Jaise ki agar jo hamari ek A plus listing me hume 5 se 6 banner used hote hain jo ki isme lagaye jaate hain. Haan ji. Matlab A plus content to wahi ho gaya na jaise jo hum normal listing ko open karte hain description mein ja ke jo extra 4 5 photos I don't know kya hoti hai. Yes, yes, yes. Yes. Toh ye A plus ho gaya aur brand store toh wahi ho gaya jo hum brand store per click karte hain toh hume banners vagera usme show hote hain. Item show hoti hai. Hmm hmm. Yes. Yes. Woh hamare 10 to 15. Haan. Yes. 10 to 15 banners usme lagte hain. Toh wo uske charges hote hain. Okay, theek hai. Toh jo 36 wala jo aapka plan hai. Isme sari services include hongi but ye dono isme include nahi hai iske alag se charges hai na. Ji ji ji ji. Yes. Theek hai. Aur advertisement pe jo bhi spend karna hai wo alag hota hai na isme. Wo totally aapke upar hai. Aap kitna budget leke chalna chahte hain isme. Haan ji haan ji. Par ye matlab ki pura jo plan hai ye 6 months ke liye ho jayega na. Yes. So. Chaliye. isme hum wo baaki hum pe depend karta hai ki hum isme A plus aur kitna add on karana chahte hain. Abhi hamara jo discussion hua tha wo 5 ke liye hua tha. But if in case Nitin sir ka koi plan change hota hai aur product leke aate hain toh wo uske upar depend karta hai. Haan wo to amount toh aapne daal hi diya. Usko to hum calculate kar hi denge. Ji. Ji ji. Theek hai na? Isme confusion dena. Toh wahi mujhe. Koi baat nahi. No issues. It's very nice ki aapne ek baar clear kar liya. Because main sirf, wo to aise hi hoti rehti toh kya fayda tha. Bilkul bilkul bilkul. Ke kaam toh hume karna hi hai na ab. Ye cheeze toh clear karke hume chalni padengi. Chaliye koi baat nahi. Ab hum theek hai. Main ek baar sir ko bhi ye dikha deta hu hai na. Okay. Hello. Okay okay okay okay. Okay. Thank you so much."
    feedback = analyze_new_call(new_call_text)

    print("\n------ FEEDBACK ------\n")
    print(feedback)
