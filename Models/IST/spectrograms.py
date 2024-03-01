from platform import win32_is_iot
import librosa
import random
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from numba import njit, guvectorize

@njit
def sample(audio_path, duration=5.0, sr=44100):
    total_duration = librosa.get_duration(path=audio_path)
    y, _ = librosa.load(audio_path, sr=sr, duration=total_duration)

    if total_duration < duration:
        pad_length = int((duration - total_duration) * sr)
        y = np.pad(y, (0, pad_length), mode='constant')

    start = random.uniform(0, max(0, total_duration - duration))
    y = y[int(start * sr):int((start + duration) * sr)]

    return y, sr


def time_x_freq_y(spec):
    spec = spec.T

    return spec


def cqt_spectrogram_path(audio_path: str, duration: float = 5.0):
    y, sr = sample(audio_path, duration)

    return cqt_spectrogram(y, sr)


def cqt_spectrogram(y, sr, filename: str = "", categories: str = "", normalise_spec=True, flip_axis=True, hop=512):
    cqt = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr, hop_length=512)), ref=np.max)

    if normalise_spec: cqt, sr = normalise(cqt, sr)
    if flip_axis: cqt = time_x_freq_y(cqt)

    return cqt, sr


def mel_spectrogram_path(audio_path: str, duration:float = 5.0):
    y, sr = sample(audio_path, duration)

    mel = librosa.feature.melspectrogram(y=y, sr=sr)

    D = librosa.power_to_db(mel, ref=np.max)

    return D, sr


def mel_spectrogram(y, sr, normalise_spec=True, flip_axis=True, fft=2048, hop=512, win=2048, mel=128):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop, win_length=win, n_mels=mel, n_fft=fft)

    # Create a spectrogram
    D = librosa.power_to_db(mel, ref=np.max)

    if normalise_spec: D, sr = normalise(D, sr)
    if flip_axis: D = time_x_freq_y(D)

    return D, sr


def mfcc_spectrogram_path(audio_path: str, duration:float = 5.0, mfcc_bins:int = 13, mel=128):
    y, sr = sample(audio_path, duration)

    return mfcc_spectrogram(y, sr, mfcc_bins, mel)


def mfcc_spectrogram(y, sr, mfcc_bins:int = 13, mel=128, fft=2048, hop=512, win=2048, normalise_spec=True, flip_axis=True):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=mfcc_bins, n_mels=mel, n_fft=fft, hop_length=hop, win_length=win)

    if normalise_spec: mfccs, sr = normalise(mfccs, sr)
    if flip_axis: mfccs = time_x_freq_y(mfccs)

    return mfccs, sr


def spectrogram_path(audio_path: str, duration:float = 5.0):
    y, sr = sample(audio_path, duration)
    
    return spectrogram(y, sr)


def spectrogram(y, sr, normalise_spec=True, flip_axis=True, hop=512, win=2048, fft=2048):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop, win_length=win, n_fft=fft)), ref=np.max)

    if normalise_spec: D, sr = normalise(D, sr)
    if flip_axis: D = time_x_freq_y(D)

    return D, sr


def load_from_path(audio_path, sr=16000):
    y, sr = librosa.load(audio_path, sr=sr)
    return y, sr


def normalise(spectrogram, sr):
    spectrogram_min = np.min(spectrogram)
    spectrogram_max = np.max(spectrogram)

    # Min-Max scaling
    normalized_spectrogram = (spectrogram - spectrogram_min) / (spectrogram_max - spectrogram_min)

    return normalized_spectrogram, sr
