import numpy as np
import librosa

SAMPLE_RATE = 22050
DURATION_SECONDS = 4
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
FIXED_LENGTH = SAMPLE_RATE * DURATION_SECONDS

def load_and_fix_length(file_path_or_buffer) -> np.ndarray:
    y, _ = librosa.load(file_path_or_buffer, sr=SAMPLE_RATE, mono=True)
    if len(y) > FIXED_LENGTH:
        y = y[:FIXED_LENGTH]
    else:
        y = np.pad(y, (0, FIXED_LENGTH - len(y)))
    return y

def extract_log_mel_spectrogram(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    return log_mel[..., np.newaxis].astype(np.float32)

def audio_file_to_features(file_path_or_buffer) -> np.ndarray:
    y = load_and_fix_length(file_path_or_buffer)
    return extract_log_mel_spectrogram(y)  