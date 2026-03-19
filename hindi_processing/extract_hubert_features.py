import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960")

audio_folder = "hindi_dataset/audio"
output_folder = "hindi_dataset/hubert_features"

os.makedirs(output_folder, exist_ok=True)

files = os.listdir(audio_folder)

for file in tqdm(files):

    path = os.path.join(audio_folder, file)

    audio, sr = librosa.load(path, sr=16000)

    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    features = outputs.last_hidden_state.squeeze().numpy()

    save_path = os.path.join(output_folder, file.replace(".wav", ".npy"))

    np.save(save_path, features)

print("HuBERT feature extraction completed")