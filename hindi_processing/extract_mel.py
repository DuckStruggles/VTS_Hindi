import os
import numpy as np
import librosa

AUDIO_FOLDER = "hindi_dataset/audio"
MEL_FOLDER   = "hindi_dataset/mel"
SAMPLE_RATE  = 22050   # HiFi-GAN expects 22050 Hz
N_MELS       = 80
N_FFT        = 1024
HOP_LENGTH   = 256
WIN_LENGTH   = 1024

os.makedirs(MEL_FOLDER, exist_ok=True)

files = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith(".wav")]

if not files:
    raise FileNotFoundError(f"No .wav files found in {AUDIO_FOLDER}")

print(f"Extracting mel spectrograms from {len(files)} audio files...")

for file in sorted(files):
    path     = os.path.join(AUDIO_FOLDER, file)
    save_path = os.path.join(MEL_FOLDER, file.replace(".wav", ".npy"))

    # load and resample to 22050 Hz (HiFi-GAN's expected sample rate)
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)

    # compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y          = audio,
        sr         = SAMPLE_RATE,
        n_fft      = N_FFT,
        hop_length = HOP_LENGTH,
        win_length = WIN_LENGTH,
        n_mels     = N_MELS,
        fmin       = 0.0,
        fmax       = 8000.0,
    )

    # log scale — same as HiFi-GAN training
    mel = np.log(np.maximum(mel, 1e-5)).astype(np.float32)  # (N_MELS, T)

    np.save(save_path, mel)
    print(f"  {file}: mel shape {mel.shape} -> {save_path}")

print(f"\nDone. Mel spectrograms saved to {MEL_FOLDER}/")
print("Now run: python hindi_processing/train_mel_projector.py")