import os
import numpy as np
import librosa

units_dir = "hindi_dataset/units"
audio_dir = "hindi_dataset/audio"

pred_unit_dir = "pred_unit/sample/sample"
pred_mel_dir = "pred_mel/sample/"
spk_dir = "spk_emb/sample/sample"

os.makedirs(pred_unit_dir, exist_ok=True)
os.makedirs(pred_mel_dir, exist_ok=True)
os.makedirs(spk_dir, exist_ok=True)

# HiFi-GAN compatible parameters
SR = 22050
N_MELS = 80
N_FFT = 1024
HOP = 256
WIN = 1024
FMIN = 0
FMAX = 8000

for file in os.listdir(units_dir):

    if not file.endswith(".units"):
        continue

    name = file.replace(".units", "")

    units_path = os.path.join(units_dir, file)
    audio_path = os.path.join(audio_dir, name + ".wav")

    if not os.path.exists(audio_path):
        continue

    print("Preparing", name)

    # copy predicted units
    units = open(units_path).read().strip()

    with open(f"{pred_unit_dir}/{name}.txt", "w") as f:
        f.write(units)

    # load audio
    audio, sr = librosa.load(audio_path, sr=SR)

    # compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=1.0
    )

    # convert to log mel (important for HiFi-GAN)
    mel = np.log(np.clip(mel, 1e-5, None))

    # save mel
    np.save(f"{pred_mel_dir}/{name}.npy", mel)

    # simple speaker embedding (mean mel vector)
    spk_emb = np.mean(mel, axis=1)
    np.save(f"{spk_dir}/{name}.npy", spk_emb)

print("Done preparing vocoder inputs")