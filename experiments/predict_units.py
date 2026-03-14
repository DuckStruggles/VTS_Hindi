import torch
import numpy as np

from models.lip2speech_model import LipToSpeech
from utils.video_loader import load_video

# Load trained model
model = LipToSpeech()
model.load_state_dict(torch.load("models/lip_model.pth"))

model.eval()

print("Model loaded!")

# choose a test video
video_path = "hindi_dataset/video/clip1.mp4"

frames = load_video(video_path)

frames = torch.tensor(frames).permute(0,3,1,2).float()
frames = frames.unsqueeze(0)

with torch.no_grad():

    output = model(frames)
probs = torch.softmax(output, dim=2)

predicted_units = torch.multinomial(
    probs[0],
    num_samples=1
).squeeze()

print("Predicted speech units:")
print(predicted_units.tolist())
