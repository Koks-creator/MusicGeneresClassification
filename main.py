from math import ceil
import os
from typing import Tuple
from collections import Counter
import tensorflow as tf
import numpy as np
import librosa


SAMPLE_RATE = 22050
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
MODEL_PATH = r"models/RNN_Audio.h5"
MODEL_NAME = os.path.split(MODEL_PATH)[-1]
MODELS_CONFIG = {
    "CNN_Audio.h5": {
        "NumSegments": 10,
        "Classes": ['country', 'blues', 'metal', 'disco', 'jazz', 'reggae', 'hiphop', 'rock', 'pop', 'classical']
    },
    "CNN_Audio1Segment60.h5": {
        "NumSegments": 1,
        "Classes": ['country', 'blues', 'metal', 'disco', 'jazz', 'reggae', 'hiphop', 'rock', 'pop', 'classical']
    },
    "RNN_Audio.h5": {
        "NumSegments": 10,
        "Classes": ['jazz', 'metal', 'pop', 'blues', 'rock', 'reggae', 'classical', 'hiphop', 'disco', 'country']
    },
}


def process_song(song_path: str, num_mfcc: int, n_fft: int, hop_length: int, sample_rate: int, samples_per_track: int,
                 num_segments: int) -> list:
    mffc_list = []

    samples_per_segment = int(samples_per_track / num_segments)
    num_mfcc_vectors_per_segment = ceil(samples_per_segment / hop_length)

    signal, sample_rate = librosa.load(path=song_path, sr=sample_rate)
    for d in range(num_segments):
        start = samples_per_segment * d
        end = start + samples_per_segment

        mfcc = librosa.feature.mfcc(
            y=signal[start:end],
            sr=sample_rate,
            n_mfcc=num_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        mfcc = mfcc.T

        if len(mfcc) == num_mfcc_vectors_per_segment:
            mffc_list.append(mfcc)

    return mffc_list


def analyze_results(predictions: np.array, classes_list: list) -> Tuple[str, dict]:
    predicted_classes = []

    for prediction in predictions:
        predicted_classes.append(classes_list[np.argmax(prediction)])

    counter = Counter(predicted_classes)

    return counter.most_common(1)[0][0], counter


song_path = r"music/blues.00069.wav"
result = process_song(song_path, NUM_MFCC, N_FFT, HOP_LENGTH, SAMPLE_RATE, SAMPLES_PER_TRACK,
                      MODELS_CONFIG[MODEL_NAME]["NumSegments"])

model = tf.keras.models.load_model(MODEL_PATH)
predictions = model.predict(np.array(result))

print(analyze_results(predictions, MODELS_CONFIG[MODEL_NAME]["Classes"]))