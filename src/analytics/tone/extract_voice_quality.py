import parselmouth
from parselmouth.praat import call

def extract_voice_quality(audio_path):
    snd = parselmouth.Sound(audio_path)

    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    avg_f0 = call(pitch, "Get mean", 0, 0, "Hertz")

    # Voice tremble / stability
    pointProcess = call(snd, "To PointProcess (periodic, cc)", 75, 600)
    jitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = call([snd, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    return {
        "avg_f0": avg_f0,
        "jitter": jitter,
        "shimmer": shimmer
    }
