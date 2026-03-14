import os
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


feature_folder = "hindi_dataset/hubert_features"
unit_folder = "hindi_dataset/units"

os.makedirs(unit_folder, exist_ok=True)


files = os.listdir(feature_folder)

all_features = []

print("Loading HuBERT features...")

for file in files:
    data = np.load(os.path.join(feature_folder, file))
    all_features.append(data)


all_features = np.vstack(all_features)

print("Training KMeans clustering...")


kmeans = KMeans(n_clusters=100, random_state=0)
kmeans.fit(all_features)

print("Generating speech units...")


for file in tqdm(files):

    features = np.load(os.path.join(feature_folder, file))

    units = kmeans.predict(features)

    save_path = os.path.join(unit_folder, file.replace(".npy", ".units"))

    with open(save_path, "w") as f:
        f.write(" ".join(map(str, units)))

print("Speech unit generation completed")