import librosa
import numpy as np

def calculate_audio_scores(audio_path, frame_count, fps):
    y, sr = librosa.load(audio_path, sr=None)
    frame_duration = int(sr / fps)
    audio_scores = []

    for i in range(frame_count):
        start = i * frame_duration
        end = start + frame_duration
        frame_audio = y[start:end]

        rms = librosa.feature.rms(y=frame_audio).mean()
        audio_scores.append(rms)
    
    audio_scores = np.array(audio_scores)
    audio_scores = (audio_scores - np.min(audio_scores)) / (np.max(audio_scores) - np.min(audio_scores) + 1e-8)
    return audio_scores.tolist()
