import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

PRED_UNIT_DIR    = os.path.join(ROOT, "pred_unit",  "sample")
SPK_EMB_DIR      = os.path.join(ROOT, "spk_emb",    "sample")
PRED_MEL_DIR     = os.path.join(ROOT, "pred_mel",   "sample")
EMBEDDINGS_PATH  = os.path.join(ROOT, "models",     "unit_embeddings.pkl")
PROJECTOR_PATH   = os.path.join(ROOT, "models",     "mel_projector.pth")

N_MELS  = 80
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(PRED_MEL_DIR, exist_ok=True)

class MelProjector(nn.Module):
    def __init__(self, in_dim: int = 768, out_dim: int = 80):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)

if not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError(
        f"{EMBEDDINGS_PATH} not found.\n"
        "Run  python hindi_processing/kmeans_hindi_units.py  first."
    )

print(f"Loading unit embeddings from {EMBEDDINGS_PATH} ...")
with open(EMBEDDINGS_PATH, "rb") as f:
    embedding_table = pickle.load(f)

VOCAB_SIZE, EMBED_DIM = embedding_table.shape
print(f"  Vocab size    : {VOCAB_SIZE}")
print(f"  Embedding dim : {EMBED_DIM}")

if not os.path.exists(PROJECTOR_PATH):
    raise FileNotFoundError(
        f"{PROJECTOR_PATH} not found.\n"
        "Run  python hindi_processing/train_mel_projector.py  first."
    )

print(f"Loading MelProjector from {PROJECTOR_PATH} ...")
projector = MelProjector(in_dim=EMBED_DIM, out_dim=N_MELS).to(DEVICE)
projector.load_state_dict(torch.load(PROJECTOR_PATH, map_location=DEVICE))
projector.eval()
print("  MelProjector loaded.")

def units_to_mel(unit_ids, spk_emb):
    unit_ids   = np.clip(unit_ids, 0, VOCAB_SIZE - 1)
    embeddings = embedding_table[unit_ids]          

    spk     = np.zeros(EMBED_DIM, dtype=np.float32)
    n       = min(len(spk_emb), EMBED_DIM)
    spk[:n] = spk_emb[:n]
    embeddings = embeddings + spk[None, :]          

    with torch.no_grad():
        emb_tensor = torch.tensor(embeddings).to(DEVICE)
        mel = projector(emb_tensor).cpu().numpy()   

    mel = mel.T

    T_in  = mel.shape[1]
    T_out = int(T_in * (22050 / 256) / 25)    

    mel_upsampled = np.zeros((N_MELS, T_out), dtype=np.float32)
    for i in range(N_MELS):
        mel_upsampled[i] = np.interp(
            np.linspace(0, T_in - 1, T_out),
            np.arange(T_in),
            mel[i]
        )

    return mel_upsampled.astype(np.float32)          

unit_files = sorted([f for f in os.listdir(PRED_UNIT_DIR) if f.endswith(".txt")])

if not unit_files:
    raise FileNotFoundError(
        f"No .txt files found in {PRED_UNIT_DIR}.\n"
        "Run  python experiments/predict_units.py  first."
    )

print(f"\nProcessing {len(unit_files)} clips...")

for fname in unit_files:
    name      = fname.replace(".txt", "")
    unit_path = os.path.join(PRED_UNIT_DIR, fname)
    spk_path  = os.path.join(SPK_EMB_DIR,   name + ".npy")
    mel_path  = os.path.join(PRED_MEL_DIR,  name + ".npy")

    content = open(unit_path).read().strip()
    if not content:
        print(f"  [skip] {name} - empty unit file")
        continue

    unit_ids = np.array(list(map(int, content.split())), dtype=np.int64)

    if os.path.exists(spk_path):
        spk_emb = np.load(spk_path).astype(np.float32).flatten()
    else:
        print(f"  [warn] no speaker embedding for {name}, using zeros")
        spk_emb = np.zeros(EMBED_DIM, dtype=np.float32)

    mel = units_to_mel(unit_ids, spk_emb)
    np.save(mel_path, mel)
    print(f"  {name}: {len(unit_ids)} units → mel {mel.shape} → {mel_path}")

print("\nDone. Run  python inference/vocoder_infer.py  to generate audio.")