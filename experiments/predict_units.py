import os
import torch
import numpy as np

from models.lip2speech_model import LipToSpeech
from utils.video_loader import load_video

# ── config ────────────────────────────────────────────────────────────────────
VIDEO_FOLDER  = "hindi_dataset/video"
MODEL_PATH    = "models/lip_model.pth"
OUTPUT_FOLDER = "pred_unit/sample"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ── load model ────────────────────────────────────────────────────────────────
model = LipToSpeech().to(DEVICE)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_PATH}. Run train.py first."
    )

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"Model loaded from {MODEL_PATH}")

# ── run inference on every video ──────────────────────────────────────────────
videos = [v for v in os.listdir(VIDEO_FOLDER) if v.endswith(".mp4")]
print(f"Running inference on {len(videos)} videos...")

for video in sorted(videos):

    name       = video.replace(".mp4", "")
    video_path = os.path.join(VIDEO_FOLDER, video)
    save_path  = os.path.join(OUTPUT_FOLDER, name + ".txt")

    frames = load_video(video_path)
    frames = torch.tensor(frames).permute(0, 3, 1, 2).float()
    frames = frames.unsqueeze(0).to(DEVICE)   # (1, T, C, H, W)

    with torch.no_grad():
        output = model(frames)                # (1, T, VOCAB_SIZE)

    # FIX 2: argmax gives the single most-likely unit per frame
    predicted_units = torch.argmax(output, dim=2)  # (1, T)
    predicted_units = predicted_units.squeeze(0).cpu().numpy()  # (T,)

    with open(save_path, "w") as f:
        f.write(" ".join(map(str, predicted_units)))

    print(f"  {name}: {len(predicted_units)} units → {save_path}")

print("Done. Now run:")
print("  python experiments/prepare_vocoder_inputs.py")
print("  python inference/vocoder_infer.py")