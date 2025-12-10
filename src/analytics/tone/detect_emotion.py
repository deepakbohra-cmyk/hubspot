import torchaudio
from speechbrain.inference.interfaces import foreign_class

def detect_emotion(audio_path):

    classifier = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2",
        pymodule_file="custom_interface.py",
        classname="CustomEmotionRecognition"
    )

    out_prob, emotion = classifier.classify_file(audio_path)

    return {
        "emotion": emotion,
        "probabilities": out_prob
    }
