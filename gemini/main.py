import os
import re
from collections import Counter
import google.generativeai as genai

genai.configure(api_key="AIzaSyC5nxTcb0BGKNgLo1DnNDj2yuQsVoMlQLc")

def load_folder(folder_path):
    texts = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r") as f:
                texts.append(f.read())
    return texts

good_calls = load_folder("good_calls")
bad_calls  = load_folder("bad_calls")

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

    print("Generating feedback from Gemini...")
    response = model.generate_content(prompt)
    return response.text


if __name__ == "__main__":
    new_call_text =  "Hello.\nI think I'm talking to a third party called Amazon.\nAh.\nMai ye bol raha hu sir. Meesho pe to mera koi work hi nahi hai sell karne ka. Mera kuch hai hi nahi Meesho pe.\nMeesho pe footwear ki baat kar rahe ho aap?\nHa footwear ki. Kyunki mere main kaise brand pe work kar raha hu. Main local work nahi kar raha hu. Mera brand 1500 ki ek jodi hai toh woh Meesho pe kahin se kahin tak ki nahi bikegi.\nEk kaam karo. Kitna kitna kya hai aapka product footwear hai na?\nHa footwear hai.\nAapka ticket size kya hai?\nKya bol rahe ho sorry?\nTicket size kya hai apki? Ticket size product size.\nProduct size hi meri 600 se start hai.\nAur mai sell ha sell karunga usko 1500 minimum.\nNahi nahi kis type ke footwear hai aapke?\nYe sab like aap search karoge Eridani karke brand hai. Metro karke brand hai. Mochi karke brand hai.\nNahi nahi nahi nahi matlab mera kehna hai ki sports sandals sandals sandals ladies footwear sandals.\nTo aap ek global add kar lo na. Amazon global pe hum karate hai aapko register. Hamara New York me bhi office hai. Vaha se hum aapka operate karenge. Yaha se bhi kar sakte hai hum.\nAmazon global add kar lo ek. US aapko isme kar dete hai Amazon.in.com aur aapka Flipkart kar lete hai.\nTeen platform aap karo. Isme aapko isliye mai keh raha hu global se aapka paanch orders bhi aa gaya ek mahine ka. To aapka kaam ho gaya fir.\nMatlab mai bata raha hu ki paanch orders bhi galti se ho gaya aur Black Friday sales wagera chal raha hai to jitna aap pure mahine me Amazon.in se kamaoge toh woh paanch order match kar dega. Kyunki waha ki ticket size waha ki price range sab kuch alag hogi.\nAchcha sir.\nTheek hai.\nToh woh aap karo. Ek global me add kara deta hu aapka Amazon.com. US me hum usko sell karenge. Dispatching yahi se hoga hamara self-ship se hoga. Jo third party vendors ke through hum self-ship karenge to usme hum kar lete hai.\nAchcha to usme mera ye charges chala jayega parcel me third party vendors se?\nThird party se uska courier charges wagera listing me hi rehta hai. Aapka koi charge nahi lagta hai. Sab pay karta hai buyer.\nTheek hai.\nTo agar buyer ha.\nBuyer agar buyer.\nAgar buyer paisa dega to.\nAur dekho aur.\nAur uska.\nEk aur jo.\nEk aur cheez. Listing me hi price rehti hai hum add karke rakhte hai na. Listing me hi price add karke rakhte hai hum saari costing.\nToh aapne ye toh bataya nahi jo global ka add kar lu main. Aur iska. Kya bolte hai? Global ka add karu aur kitna bata rahe ho aap?\nHello.\nYa.\nYeah.\nHa abhi kar lijiye." 
    feedback = analyze_new_call(new_call_text)
    print("\n------ Feedback ------\n")
    print(feedback)
