import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from scipy.io.wavfile import write

MEL_FOLDER      = "pred_mel/sample"
OUTPUT_FOLDER   = "outputs/hi"
CHECKPOINT_PATH = "vocoder/hifigan_hi/best_model.pth"
SAMPLE_RATE     = 22050
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Loading Hindi HiFi-GAN (Indic-TTS)...")

try:
    from TTS.vocoder.models.hifigan_generator import HifiganGenerator
except ImportError:
    raise ImportError(
        "Coqui TTS not installed. Run: pip install TTS"
    )

generator = HifiganGenerator(
    in_channels            = 80,
    out_channels           = 1,
    resblock_type          = "1",
    resblock_dilation_sizes= [[1,3,5],[1,3,5],[1,3,5]],
    resblock_kernel_sizes  = [3,7,11],
    upsample_kernel_sizes  = [16,16,4,4],
    upsample_initial_channel = 512,
    upsample_factors       = [8,8,2,2],
).to(DEVICE)

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Hindi HiFi-GAN checkpoint not found at {CHECKPOINT_PATH}\n"
        "Copy hi/hifigan/best_model.pth to vocoder/hifigan_hi/best_model.pth"
    )

ckpt      = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
state     = ckpt["model"]

gen_state = {
    k.replace("model_g.", ""): v
    for k, v in state.items()
    if k.startswith("model_g.")
}

generator.load_state_dict(gen_state)
generator.eval()

if hasattr(generator, "remove_weight_norm"):
    generator.remove_weight_norm()

print(f"  Hindi HiFi-GAN loaded from {CHECKPOINT_PATH}")

mel_files = sorted([f for f in os.listdir(MEL_FOLDER) if f.endswith(".npy")])

if not mel_files:
    raise FileNotFoundError(
        f"No .npy files found in {MEL_FOLDER}.\n"
        "Run prepare_vocoder_inputs.py first."
    )

print(f"Generating audio for {len(mel_files)} clips...")

for file in mel_files:
    name     = file.replace(".npy", "")
    mel_path = os.path.join(MEL_FOLDER, file)
    out_path = os.path.join(OUTPUT_FOLDER, f"{name}_hi.wav")

    mel = np.load(mel_path)                              
    mel = torch.FloatTensor(mel).unsqueeze(0).to(DEVICE)  

    with torch.no_grad():
        audio = generator(mel)                      

    audio = audio.squeeze().cpu().numpy()
    audio = audio / (np.max(np.abs(audio)) + 1e-8)   

    write(out_path, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    print(f"  {name} → {out_path}")

print(f"\nDone. Hindi vocoder outputs saved to {OUTPUT_FOLDER}/")
print("Files named *_hi.wav — compare with *_en.wav for quality difference.")