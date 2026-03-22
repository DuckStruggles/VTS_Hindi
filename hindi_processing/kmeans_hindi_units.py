import os
import pickle
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

VOCAB_SIZE     = 100   # must match lip2speech_model.py and train.py
RANDOM_STATE   = 0

feature_folder = "hindi_dataset/hubert_features"
unit_folder    = "hindi_dataset/units"
models_folder  = "models"

os.makedirs(unit_folder,   exist_ok=True)
os.makedirs(models_folder, exist_ok=True)

# ── load all HuBERT features ─────────────────────────────────────────────────
files = [f for f in os.listdir(feature_folder) if f.endswith(".npy")]

if not files:
    raise FileNotFoundError(f"No .npy files found in {feature_folder}. "
                            "Run extract_hubert_features.py first.")

all_features = []
print(f"Loading HuBERT features from {len(files)} files...")

for file in tqdm(files):
    data = np.load(os.path.join(feature_folder, file))
    if data.ndim == 1:
        data = data[np.newaxis, :]   # handle single-frame clips
    all_features.append(data)

all_features = np.vstack(all_features)   # (total_frames, feature_dim)
print(f"Total frames: {all_features.shape[0]}, feature dim: {all_features.shape[1]}")

# ── train KMeans ─────────────────────────────────────────────────────────────
print(f"Training KMeans with {VOCAB_SIZE} clusters...")

kmeans = KMeans(n_clusters=VOCAB_SIZE, random_state=RANDOM_STATE, n_init=10)
kmeans.fit(all_features)

print("KMeans training complete.")

# ── save KMeans model ─────────────────────────────────────────────────────────
kmeans_path = os.path.join(models_folder, "kmeans_model.pkl")
with open(kmeans_path, "wb") as f:
    pickle.dump(kmeans, f)
print(f"KMeans model saved → {kmeans_path}")

# ── FIX: save cluster centroids as unit_embeddings.pkl ───────────────────────
# centroids shape: (VOCAB_SIZE, feature_dim)  e.g. (100, 768)
# prepare_vocoder_inputs.py uses this to convert predicted unit IDs → vectors
centroids = kmeans.cluster_centers_.astype(np.float32)

embeddings_path = os.path.join(models_folder, "unit_embeddings.pkl")
with open(embeddings_path, "wb") as f:
    pickle.dump(centroids, f)
print(f"Unit embeddings saved → {embeddings_path}  shape: {centroids.shape}")

# ── assign unit IDs to every clip ────────────────────────────────────────────
print("Generating speech unit files...")

for file in tqdm(files):
    features  = np.load(os.path.join(feature_folder, file))
    if features.ndim == 1:
        features = features[np.newaxis, :]

    units     = kmeans.predict(features)          # (T,) int array 0..99
    save_path = os.path.join(unit_folder, file.replace(".npy", ".units"))

    with open(save_path, "w") as f:
        f.write(" ".join(map(str, units)))

print("Speech unit generation complete.")
print(f"Vocab size : {VOCAB_SIZE}")
print(f"Embedding dim (HuBERT feature dim): {centroids.shape[1]}")