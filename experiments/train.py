import os
import torch
import numpy as np

from models.lip2speech_model import LipToSpeech
from utils.video_loader import load_video

print("Starting training...")

model = LipToSpeech()

optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

loss_fn = torch.nn.CrossEntropyLoss()

video_folder = "hindi_dataset/video"
units_folder = "hindi_dataset/units"

videos = os.listdir(video_folder)
import random
random.shuffle(videos)

epochs = 40

for epoch in range(epochs):

    print("Epoch:", epoch+1)

for video in videos:

    video_path = os.path.join(video_folder, video)

    name = video.replace(".mp4","")

    units_path = os.path.join(units_folder, name + ".units")

    if not os.path.exists(units_path):
        continue

    frames = load_video(video_path)

    units = open(units_path).read().split()

    units = np.array(units).astype(int)

    frames = torch.tensor(frames).permute(0,3,1,2).float()
    frames = frames.unsqueeze(0)

    units = torch.tensor(units).long()

    output = model(frames)

    pred_len = output.shape[1]
    unit_len = units.shape[0]

    min_len = min(pred_len, unit_len)

    output = output[:, :min_len, :]
    units = units[:min_len]

    loss = loss_fn(
    output.reshape(-1,100),
    units.reshape(-1)
)

    loss.backward()

    optimizer.step()

    print("trained:", video)

torch.save(model.state_dict(),"models/lip_model.pth")

print("Model saved!")