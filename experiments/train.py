import os, sys
import random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # add this
import torch
import numpy as np

from models.lip2speech_model import LipToSpeech, VOCAB_SIZE
from utils.video_loader import load_video

# ── config ────────────────────────────────────────────────────────────────────
VIDEO_FOLDER  = "hindi_dataset/video"
UNITS_FOLDER  = "hindi_dataset/units"
MODEL_SAVE    = "models/lip_model.pth"
EPOCHS        = 3
LR            = 5e-5
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training on: {DEVICE}")
print(f"Vocab size : {VOCAB_SIZE}")

# ── model, optimiser, loss ────────────────────────────────────────────────────
model     = LipToSpeech().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn   = torch.nn.CrossEntropyLoss()

videos = [v for v in os.listdir(VIDEO_FOLDER) if v.endswith(".mp4")]

print(f"Found {len(videos)} videos.")

# ── training loop ─────────────────────────────────────────────────────────────
for epoch in range(EPOCHS):

    # FIX 1: everything below is now correctly indented inside the epoch loop
    random.shuffle(videos)
    model.train()                          # FIX 4: sets BatchNorm to train mode

    epoch_loss = 0.0
    trained    = 0

    for video in videos:

        name       = video.replace(".mp4", "")
        video_path = os.path.join(VIDEO_FOLDER, video)
        units_path = os.path.join(UNITS_FOLDER, name + ".units")

        if not os.path.exists(units_path):
            continue

        # load frames
        frames = load_video(video_path)
        frames = torch.tensor(frames).permute(0, 3, 1, 2).float()
        frames = frames.unsqueeze(0).to(DEVICE)       # (1, T, C, H, W)

        # load target units
        units = np.array(open(units_path).read().split(), dtype=int)
        units = torch.tensor(units).long().to(DEVICE) # (T,)

        # forward
        output = model(frames)                         # (1, T, VOCAB_SIZE)

        # align lengths (video fps vs hubert frame rate may differ slightly)
        min_len = min(output.shape[1], units.shape[0])
        output  = output[:, :min_len, :]               # (1, min_len, VOCAB_SIZE)
        units   = units[:min_len]                      # (min_len,)

        # FIX 3: use VOCAB_SIZE constant instead of hardcoded 100
        loss = loss_fn(
            output.reshape(-1, VOCAB_SIZE),
            units.reshape(-1)
        )

        # FIX 2: zero_grad before backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        trained    += 1

    avg_loss = epoch_loss / max(trained, 1)
    print(f"Epoch {epoch+1}/{EPOCHS}  |  clips: {trained}  |  avg loss: {avg_loss:.4f}")

# ── save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_SAVE), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE)
print(f"Model saved → {MODEL_SAVE}")