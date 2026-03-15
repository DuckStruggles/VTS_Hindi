import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import torch
import json
import numpy as np
from scipy.io.wavfile import write

from vocoder.hifigan.models import Generator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# paths
mel_folder = "pred_mel/sample/"
output_folder = "outputs"
config_path = "vocoder/hifigan/config.json"
checkpoint_path = "vocoder/hifigan/generator_v1"

os.makedirs(output_folder, exist_ok=True)

# load config
from types import SimpleNamespace

with open(config_path) as f:
    config = json.load(f)

config = SimpleNamespace(**config)

model = Generator(config).to(device)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["generator"])

model.eval()


for file in os.listdir(mel_folder):

    if file.endswith(".npy"):

        mel_path = os.path.join(mel_folder, file)

        mel = np.load(mel_path)

        mel = torch.FloatTensor(mel).unsqueeze(0).to(device)

        with torch.no_grad():
            audio = model(mel)

        audio = audio.squeeze().cpu().numpy()

        output_path = os.path.join(output_folder, file.replace(".npy", ".wav"))
        audio = audio / np.max(np.abs(audio))
        write(output_path, 22050, audio)

        print("Generated:", output_path)