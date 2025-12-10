import librosa
import numpy as np

def extract_prosody(audio_path):
    y, sr = librosa.load(audio_path)

    # Pitch (F0)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = pitches[magnitudes > np.median(magnitudes)]
    avg_pitch = np.mean(pitch) if len(pitch) > 0 else 0

    # Energy / Volume
    energy = np.mean(librosa.feature.rms(y=y))

    # Speech Speed (Tempo)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    return {
        "avg_pitch": float(avg_pitch),
        "energy": float(energy),
        "tempo": float(tempo)
    }
