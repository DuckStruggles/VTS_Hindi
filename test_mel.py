import numpy as np
import matplotlib.pyplot as plt

gt  = np.load("hindi_dataset/mel/clip01.npy")   # real mel
gen = np.load("pred_mel/sample/clip01.npy")      # your generated mel

fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].imshow(gt,  aspect="auto", origin="lower")
axes[0].set_title("Ground Truth Mel")
axes[1].imshow(gen, aspect="auto", origin="lower")
axes[1].set_title("Generated Mel")
plt.tight_layout()
plt.savefig("mel_comparison.png")
print("Saved mel_comparison.png")