import os
import numpy as np
import librosa
import pickle

print("Script started")

units_dir = "hindi_dataset/units"
audio_dir = "hindi_dataset/audio"

SAMPLE_RATE = 16000
MEL_BINS = 80
VOCAB_SIZE = 200

unit_mels = {i: [] for i in range(VOCAB_SIZE)}

for file in os.listdir(units_dir):

    if not file.endswith(".units"):
        continue

    name = file.replace(".units","")

    units_path = os.path.join(units_dir, file)
    audio_path = os.path.join(audio_dir, name + ".wav")

    if not os.path.exists(audio_path):
        continue

    print("Processing", name)

    with open(units_path) as f:
        units = list(map(int, f.read().split()))

    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=MEL_BINS
    )

    mel = mel.T

    length = min(len(units), len(mel))

    for i in range(length):
        unit_mels[units[i]].append(mel[i])


embeddings = np.zeros((VOCAB_SIZE, MEL_BINS))

for unit in unit_mels:

    if len(unit_mels[unit]) > 0:
        embeddings[unit] = np.mean(unit_mels[unit], axis=0)

print("Saving unit embeddings...")

with open("models/unit_embeddings.pkl","wb") as f:
    pickle.dump(embeddings,f)

print("Done.")