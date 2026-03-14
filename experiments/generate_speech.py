import numpy as np
import librosa
import soundfile as sf
import os
import sys
import pickle

# ----------------------------
# configuration
# ----------------------------

SAMPLE_RATE = 16000
MEL_BINS = 80
VOCAB_SIZE = 200

units_folder = "hindi_dataset/units"
output_folder = "outputs"

os.makedirs(output_folder, exist_ok=True)

# ----------------------------
# get clip name from terminal
# ----------------------------

if len(sys.argv) < 2:
    print("Usage: python generate_speech.py clip_name")
    sys.exit()

clip = sys.argv[1]

units_path = os.path.join(units_folder, clip + ".units")

print("Loading units:", units_path)

# ----------------------------
# load unit tokens
# ----------------------------

with open(units_path) as f:
    units = list(map(int, f.read().split()))

units = np.array(units)

print("Total units:", len(units))

# ----------------------------
# load learned embeddings
# ----------------------------

print("Loading learned unit embeddings...")

with open("models/unit_embeddings.pkl", "rb") as f:
    embedding_table = pickle.load(f)

# ----------------------------
# units → mel spectrogram
# ----------------------------

print("Creating mel spectrogram...")

mel = embedding_table[units]

mel = mel.T

# ----------------------------
# mel → waveform
# ----------------------------

print("Generating audio...")

audio = librosa.feature.inverse.mel_to_audio(
    mel,
    sr=SAMPLE_RATE,
    n_iter=32
)

# ----------------------------
# save output
# ----------------------------

output_file = os.path.join(output_folder, clip + "_generated.wav")

sf.write(
    output_file,
    audio,
    SAMPLE_RATE
)

print("Audio saved:", output_file)
print("Hindi speech generated successfully.")