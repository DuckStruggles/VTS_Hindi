import os, sys, random, csv
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import cv2
cv2.setNumThreads(0)

torch.set_num_threads(os.cpu_count())

from models.lip2speech_model import LipToSpeech, VOCAB_SIZE
from utils.video_loader import load_video

VIDEO_FOLDER = "hindi_dataset/video"
UNITS_FOLDER = "hindi_dataset/units"
MODEL_SAVE   = "models/lip_model.pth"
LOG_FILE     = "logs/training_log.csv"
EPOCHS       = 80
LR           = 5e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training on : {DEVICE}")
print(f"Vocab size  : {VOCAB_SIZE}")

# ── create logs folder ────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)

# ── open log file ─────────────────────────────────────────────────────────────
log_f  = open(LOG_FILE, "w", newline="")
writer = csv.writer(log_f)
writer.writerow(["epoch", "avg_loss", "clips_trained", "timestamp"])
print(f"Logging to  : {LOG_FILE}")

# ── model setup ───────────────────────────────────────────────────────────────
model     = LipToSpeech().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn   = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

videos = [v for v in os.listdir(VIDEO_FOLDER) if v.endswith(".mp4")]
print(f"Found {len(videos)} videos.")

best_loss = float("inf")

for epoch in range(EPOCHS):

    random.shuffle(videos)
    model.train()

    epoch_loss = 0.0
    trained    = 0

    for video in videos:

        name       = video.replace(".mp4", "")
        video_path = os.path.join(VIDEO_FOLDER, video)
        units_path = os.path.join(UNITS_FOLDER, name + ".units")

        if not os.path.exists(units_path):
            continue

        # load frames -> (1, T, C, H, W)
        frames = load_video(video_path)
        frames = torch.tensor(frames).permute(0, 3, 1, 2).float()
        frames = frames.unsqueeze(0).to(DEVICE)

        # load target units -> (T,)
        units = np.array(open(units_path).read().split(), dtype=int)
        units = units[::2]   # downsample 50Hz → 25Hz to match video fps
        units = torch.tensor(units).long().to(DEVICE)

        # forward pass
        output = model(frames)                         # (1, T, VOCAB_SIZE)

        # align lengths
        min_len = min(output.shape[1], units.shape[0])
        output  = output[:, :min_len, :]               # (1, min_len, VOCAB_SIZE)
        units   = units[:min_len]                      # (min_len,)

        loss = loss_fn(
        output.reshape(-1, VOCAB_SIZE),
        units.reshape(-1)
        )

    # 🔥 diversity regularization
        entropy = -torch.sum(torch.softmax(output, dim=-1) * torch.log_softmax(output, dim=-1))
        loss = loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        trained    += 1

    avg_loss  = epoch_loss / max(trained, 1)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"Epoch {epoch+1}/{EPOCHS}  |  clips: {trained}  |  avg loss: {avg_loss:.4f}")

    # ── log to CSV ────────────────────────────────────────────────────────────
    writer.writerow([epoch + 1, round(avg_loss, 6), trained, timestamp])
    log_f.flush()

    # ── checkpoint every 10 epochs ────────────────────────────────────────────
    if (epoch + 1) % 10 == 0:
        os.makedirs(os.path.dirname(MODEL_SAVE), exist_ok=True)
        torch.save({
            "epoch":                epoch + 1,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss":                 avg_loss,
        }, MODEL_SAVE.replace(".pth", f"_epoch{epoch+1}.pth"))
        print(f"  Checkpoint saved → epoch {epoch+1}")

    # ── save best model ───────────────────────────────────────────────────────
    if avg_loss < best_loss:
        best_loss = avg_loss
        os.makedirs(os.path.dirname(MODEL_SAVE), exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE.replace(".pth", "_best.pth"))

log_f.close()

# ── save final model ──────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_SAVE), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE)
print(f"Model saved      → {MODEL_SAVE}")
print(f"Best model saved → {MODEL_SAVE.replace('.pth', '_best.pth')}")
print(f"Training log     → {LOG_FILE}")