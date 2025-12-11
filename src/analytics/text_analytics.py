from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import nltk


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

def extract_effective_phrases(text):
    text_lower = text.lower()
    sentences = [s.strip() for s in sent_tokenize(text_lower) if s.strip()]

    if not sentences:
        return []

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)

    # Automatically adjust number of clusters
    num_clusters = min(4, len(sentences))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    effective_phrases = []
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]

        # Pick sentence with highest "semantic strength"
        norms = np.linalg.norm(cluster_embeddings, axis=1)
        top_index = cluster_indices[np.argmax(norms)]
        effective_phrases.append(sentences[top_index])

    return effective_phrases

def extract_repeated_words(text, topn=5):
    # Extract words
    words = re.findall(r'\b\w+\b', text.lower())

    # Filter stopwords (English + Hindi + Hinglish)
    filtered = [w for w in words if w not in stop_words]

    freq = Counter(filtered)
    return freq.most_common(topn)