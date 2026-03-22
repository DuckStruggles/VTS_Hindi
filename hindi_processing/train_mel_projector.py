"""
train_mel_projector.py
----------------------
Trains a small MelProjector network that learns to map
HuBERT embeddings (768-dim) → mel spectrogram frames (80-dim).

Inputs  : hindi_dataset/hubert_features/*.npy   (T, 768)
Targets : hindi_dataset/mel/*.npy               (80, T)

Output  : models/mel_projector.pth
Log     : logs/mel_projector_log.csv

Run AFTER extract_mel.py and BEFORE prepare_vocoder_inputs.py.
"""

import os, csv
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn

# ── config ────────────────────────────────────────────────────────────────────
HUBERT_FOLDER = "hindi_dataset/hubert_features"
MEL_FOLDER    = "hindi_dataset/mel"
MODEL_SAVE    = "models/mel_projector.pth"
LOG_FILE      = "logs/mel_projector_log.csv"
EPOCHS        = 100
LR            = 1e-3
EMBED_DIM     = 768
N_MELS        = 80
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training MelProjector on: {DEVICE}")

os.makedirs("logs",   exist_ok=True)
os.makedirs("models", exist_ok=True)

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

pairs = []

print(f"Loading {len(hubert_files)} clips...")

for file in hubert_files:
    name     = file.replace(".npy", "")
    h_path   = os.path.join(HUBERT_FOLDER, file)
    mel_path = os.path.join(MEL_FOLDER, name + ".npy")

    if not os.path.exists(mel_path):
        print(f"  [skip] {name} — no mel file found")
        continue

    hubert = np.load(h_path).astype(np.float32)   # (T_h, 768)
    mel    = np.load(mel_path).astype(np.float32)  # (80, T_m)

    T_h = hubert.shape[0]
    T_m = mel.shape[1]

    if T_h == 0 or T_m == 0:
        print(f"  [skip] {name} — empty array")
        continue

    # resample mel time axis to match HuBERT frame count
    mel_resampled = np.zeros((N_MELS, T_h), dtype=np.float32)
    for i in range(N_MELS):
        mel_resampled[i] = np.interp(
            np.linspace(0, T_m - 1, T_h),
            np.arange(T_m),
            mel[i]
        )

    mel_T = mel_resampled.T   # (T_h, 80)

    pairs.append((
        torch.tensor(hubert),
        torch.tensor(mel_T),
    ))
    print(f"  {name}: hubert {hubert.shape} → mel {mel_T.shape}")

if not pairs:
    raise RuntimeError("No valid clip pairs found. Check that extract_mel.py ran correctly.")

print(f"\nLoaded {len(pairs)} clips.")

# ── open log file ─────────────────────────────────────────────────────────────
log_f  = open(LOG_FILE, "w", newline="")
writer = csv.writer(log_f)
writer.writerow(["epoch", "avg_loss", "timestamp"])
print(f"Logging to: {LOG_FILE}")

# ── train ─────────────────────────────────────────────────────────────────────
model     = MelProjector(EMBED_DIM, N_MELS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn   = nn.MSELoss()
best_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for hubert, mel in pairs:
        hubert = hubert.to(DEVICE)
        mel    = mel.to(DEVICE)

        pred = model(hubert)
        loss = loss_fn(pred, mel)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg       = epoch_loss / len(pairs)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    writer.writerow([epoch + 1, round(avg, 8), timestamp])
    log_f.flush()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}  |  avg loss: {avg:.6f}")

    # save best model
    if avg < best_loss:
        best_loss = avg
        torch.save(model.state_dict(), MODEL_SAVE.replace(".pth", "_best.pth"))

log_f.close()

# ── save final model ──────────────────────────────────────────────────────────
torch.save(model.state_dict(), MODEL_SAVE)
print(f"\nMelProjector saved      → {MODEL_SAVE}")
print(f"Best MelProjector saved → {MODEL_SAVE.replace('.pth', '_best.pth')}")
print(f"Log saved               → {LOG_FILE}")