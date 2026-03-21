import os
import torch
import numpy as np

from models.lip2speech_model import LipToSpeech
from utils.video_loader import load_video

# ── config ────────────────────────────────────────────────────────────────────
VIDEO_FOLDER  = "hindi_dataset/video"
MODEL_PATH = "models/lip_model_best.pth"
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
    frames = frames.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(frames)

    # advanced decoding
    temperature = 1.3
    probs = torch.softmax(output / 1.5, dim=2)

    predicted_units = torch.multinomial(probs[0], 1).squeeze().cpu().numpy()

    # smoothing
    #window = 2
    #for i in range(window, len(predicted_units)-window):
        #segment = predicted_units[i-window:i+window]
        #predicted_units[i] = max(set(segment), key=list(segment).count)

    # 🔥 FIX: ALSO INSIDE LOOP
    with open(save_path, "w") as f:
        f.write(" ".join(map(str, predicted_units)))

    print(f"{name}: {len(predicted_units)} units → {save_path}")

print("Done. Now run:")
print("  python experiments/prepare_vocoder_inputs.py")
print("  python inference/vocoder_infer.py")