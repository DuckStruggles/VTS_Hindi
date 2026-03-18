"""
train_mel_projector.py
----------------------
Trains a small MelProjector network that learns to map
HuBERT embeddings (768-dim) → mel spectrogram frames (80-dim).

Inputs  : hindi_dataset/hubert_features/*.npy   (T, 768)
Targets : hindi_dataset/mel/*.npy               (80, T)

Output  : models/mel_projector.pth

Run AFTER extract_mel.py and BEFORE prepare_vocoder_inputs.py.
"""

import os
import numpy as np
import torch
import torch.nn as nn

# ── config ────────────────────────────────────────────────────────────────────
HUBERT_FOLDER = "hindi_dataset/hubert_features"
MEL_FOLDER    = "hindi_dataset/mel"
MODEL_SAVE    = "models/mel_projector.pth"
EPOCHS        = 100
LR            = 1e-3
EMBED_DIM     = 768   # HuBERT base output dim
N_MELS        = 80
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training MelProjector on: {DEVICE}")

# ── model ─────────────────────────────────────────────────────────────────────
class MelProjector(nn.Module):
    def __init__(self, in_dim: int = 768, out_dim: int = 80):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)   # (T, N_MELS)

# ── load data ─────────────────────────────────────────────────────────────────
hubert_files = sorted([f for f in os.listdir(HUBERT_FOLDER) if f.endswith(".npy")])

if not hubert_files:
    raise FileNotFoundError(f"No .npy files in {HUBERT_FOLDER}. Run extract_hubert_features.py first.")

pairs = []   # list of (hubert_features, mel_frames) tensors

print(f"Loading {len(hubert_files)} clips...")

for file in hubert_files:
    name     = file.replace(".npy", "")
    h_path   = os.path.join(HUBERT_FOLDER, file)
    mel_path = os.path.join(MEL_FOLDER, name + ".npy")

    if not os.path.exists(mel_path):
        print(f"  [skip] {name} — no mel file found")
        continue

    hubert = np.load(h_path).astype(np.float32)   # (T_h, 768)  at 50 Hz
    mel    = np.load(mel_path).astype(np.float32)  # (80, T_m)   at ~86 Hz

    # HuBERT is at 50 Hz, mel is at 22050/256 ≈ 86 Hz
    # Downsample mel to match HuBERT length by taking every other frame
    # More precisely: interpolate mel to HuBERT length
    T_h = hubert.shape[0]
    T_m = mel.shape[1]

    if T_h == 0 or T_m == 0:
        print(f"  [skip] {name} — empty array")
        continue

    # resample mel time axis to match HuBERT frames using linear interpolation
    mel_resampled = np.zeros((N_MELS, T_h), dtype=np.float32)
    for i in range(N_MELS):
        mel_resampled[i] = np.interp(
            np.linspace(0, T_m - 1, T_h),
            np.arange(T_m),
            mel[i]
        )

    # mel_resampled: (80, T_h) → transpose to (T_h, 80)
    mel_T = mel_resampled.T   # (T_h, 80)

    pairs.append((
        torch.tensor(hubert),   # (T_h, 768)
        torch.tensor(mel_T),    # (T_h, 80)
    ))
    print(f"  {name}: hubert {hubert.shape} → mel {mel_T.shape}")

if not pairs:
    raise RuntimeError("No valid clip pairs found. Check that extract_mel.py ran correctly.")

print(f"\nLoaded {len(pairs)} clips.")

# ── train ─────────────────────────────────────────────────────────────────────
model     = MelProjector(EMBED_DIM, N_MELS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn   = nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for hubert, mel in pairs:
        hubert = hubert.to(DEVICE)
        mel    = mel.to(DEVICE)

        pred = model(hubert)          # (T, 80)
        loss = loss_fn(pred, mel)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg = epoch_loss / len(pairs)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}  |  avg loss: {avg:.6f}")

# ── save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_SAVE), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE)
print(f"\nMelProjector saved → {MODEL_SAVE}")
print("Now update prepare_vocoder_inputs.py to use mel_projector.pth")