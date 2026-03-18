import sys
import os
import pickle
import json
import numpy as np
import torch
from scipy.io.wavfile import write
from types import SimpleNamespace

# ── resolve project root regardless of where the script is called from ──────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from vocoder.hifigan.models import Generator

# ── configuration ────────────────────────────────────────────────────────────
PRED_UNIT_DIR   = os.path.join(ROOT, "pred_unit",  "sample")
SPK_EMB_DIR     = os.path.join(ROOT, "spk_emb",    "sample")
EMBEDDINGS_PATH = os.path.join(ROOT, "models",     "unit_embeddings.pkl")
CONFIG_PATH     = os.path.join(ROOT, "vocoder",    "hifigan", "config.json")
CKPT_PATH       = os.path.join(ROOT, "vocoder",    "hifigan", "generator_v1")
OUTPUT_DIR      = os.path.join(ROOT, "outputs")

SAMPLE_RATE = 22050
N_MELS      = 80
VOCAB_SIZE  = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── argument parsing ──────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python generate_speech.py <clip_name>")
    print("Example: python generate_speech.py clip1")
    sys.exit(1)

clip = sys.argv[1]

# ── load predicted unit IDs ──────────────────────────────────────────────────
unit_path = os.path.join(PRED_UNIT_DIR, clip + ".txt")

if not os.path.exists(unit_path):
    print(f"ERROR: predicted unit file not found: {unit_path}")
    sys.exit(1)

with open(unit_path) as f:
    content = f.read().strip()

unit_ids = np.array(list(map(int, content.split())), dtype=np.int64)
print(f"Loaded {len(unit_ids)} predicted units for '{clip}'")

# ── load unit embedding table ─────────────────────────────────────────────────
print("Loading unit embeddings…")
with open(EMBEDDINGS_PATH, "rb") as f:
    embedding_table = pickle.load(f)

VOCAB_SIZE_ACTUAL, EMBED_DIM = embedding_table.shape
unit_ids = np.clip(unit_ids, 0, VOCAB_SIZE_ACTUAL - 1)

# ── load speaker embedding ────────────────────────────────────────────────────
spk_path = os.path.join(SPK_EMB_DIR, clip + ".npy")

if os.path.exists(spk_path):
    spk_emb = np.load(spk_path).astype(np.float32).flatten()
    print(f"Speaker embedding loaded: shape {spk_emb.shape}")
else:
    print(f"Warning: no speaker embedding found at {spk_path}, using zeros")
    spk_emb = np.zeros(EMBED_DIM, dtype=np.float32)

# ── units → mel spectrogram ───────────────────────────────────────────────────
# FIX: this is the correct conversion order.
#   step 1 – look up the embedding for every predicted unit token
#   step 2 – add speaker identity
#   step 3 – project embedding dim → N_MELS
#   step 4 – log-scale so values are in the range HiFi-GAN was trained on

print("Converting units → mel spectrogram…")

embeddings = embedding_table[unit_ids]          # (T, EMBED_DIM)

# add speaker embedding as a global per-frame offset
spk = np.zeros(EMBED_DIM, dtype=np.float32)
n   = min(len(spk_emb), EMBED_DIM)
spk[:n] = spk_emb[:n]
embeddings = embeddings + spk[None, :]          # (T, EMBED_DIM)

# project to mel dimension (same fixed weights as prepare_vocoder_inputs.py)
rng    = np.random.default_rng(42)
W_proj = rng.standard_normal((EMBED_DIM, N_MELS)).astype(np.float32) * 0.1
b_proj = np.zeros(N_MELS, dtype=np.float32)

mel = embeddings @ W_proj + b_proj              # (T, N_MELS)
mel = np.log(np.maximum(mel, 1e-5))             # log-scale
mel = mel.T.astype(np.float32)                  # (N_MELS, T)  ← HiFi-GAN expects this

print(f"Mel spectrogram shape: {mel.shape}")

# ── load HiFi-GAN ─────────────────────────────────────────────────────────────
# FIX: use HiFi-GAN, not librosa Griffin-Lim
print("Loading HiFi-GAN vocoder…")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CONFIG_PATH) as f:
    config = SimpleNamespace(**json.load(f))

model = Generator(config).to(device)
ckpt  = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["generator"])
model.eval()

# ── mel → waveform ────────────────────────────────────────────────────────────
print("Synthesising waveform with HiFi-GAN…")

mel_tensor = torch.FloatTensor(mel).unsqueeze(0).to(device)  # (1, N_MELS, T)

with torch.no_grad():
    audio = model(mel_tensor)

audio = audio.squeeze().cpu().numpy()
audio = audio / (np.max(np.abs(audio)) + 1e-8)  # normalise to [-1, 1]

# ── save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, f"{clip}_generated.wav")
write(out_path, SAMPLE_RATE, (audio * 32767).astype(np.int16))

print(f"Saved: {out_path}")
print("Done.")