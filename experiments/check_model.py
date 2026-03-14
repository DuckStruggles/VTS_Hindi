import torch

model_data = torch.load("models/lip_model.pth")

print("Model loaded successfully!")

print(type(model_data))

print(model_data.keys())