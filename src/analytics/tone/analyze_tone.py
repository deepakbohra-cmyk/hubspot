
def analyze_tone(audio_path):
    prosody = extract_prosody(audio_path)
    voice_quality = extract_voice_quality(audio_path)
    emotion = detect_emotion(audio_path)

    result = {
        "avg_pitch": prosody["avg_pitch"],
        "energy": prosody["energy"],
        "tempo": prosody["tempo"],
        "voice_confidence": "high" if voice_quality["jitter"] < 0.02 else "low",
        "emotion": emotion["emotion"],
        "jitter": voice_quality["jitter"],
        "shimmer": voice_quality["shimmer"]
    }

    # Simple rule-based tone summary:
    tone = ""

    if result["emotion"] == "happy":
        tone = "Positive / Cheerful"
    elif result["emotion"] == "angry":
        tone = "Frustrated / Irritated"
    elif result["emotion"] == "sad":
        tone = "Low-energy / Unhappy"
    elif result["emotion"] == "neutral":
        tone = "Calm / Professional"

    if result["voice_confidence"] == "low":
        tone += " + Uncertain/Low Confidence"

    result["tone_summary"] = tone

    return result
