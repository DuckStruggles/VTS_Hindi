import os
import pickle
import numpy as np

PRED_UNIT_DIR   = "pred_unit/sample"
SPK_EMB_DIR     = "spk_emb/sample"
PRED_MEL_DIR    = "pred_mel/sample"
EMBEDDINGS_PATH = "models/unit_embeddings.pkl"

os.makedirs(PRED_MEL_DIR, exist_ok=True)

N_MELS = 80

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

# fixed projection EMBED_DIM -> N_MELS, same seed as generate_speech.py
rng    = np.random.default_rng(42)
W_proj = rng.standard_normal((EMBED_DIM, N_MELS)).astype(np.float32) * 0.1
b_proj = np.zeros(N_MELS, dtype=np.float32)


def units_to_mel(unit_ids, spk_emb):
    unit_ids   = np.clip(unit_ids, 0, VOCAB_SIZE - 1)
    embeddings = embedding_table[unit_ids]          # (T, EMBED_DIM)

    spk      = np.zeros(EMBED_DIM, dtype=np.float32)
    n        = min(len(spk_emb), EMBED_DIM)
    spk[:n]  = spk_emb[:n]
    embeddings = embeddings + spk[None, :]

    mel = embeddings @ W_proj + b_proj              # (T, N_MELS)
    mel = np.log(np.maximum(mel, 1e-5))
    return mel.T.astype(np.float32)                 # (N_MELS, T)


unit_files = sorted([f for f in os.listdir(PRED_UNIT_DIR) if f.endswith(".txt")])

if not unit_files:
    raise FileNotFoundError(
        f"No .txt files found in {PRED_UNIT_DIR}.\n"
        "Run  python experiments/predict_units.py  first."
    )

print(f"Processing {len(unit_files)} clips...")

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
    print(f"  {name}: {len(unit_ids)} units -> mel {mel.shape} -> {mel_path}")

print("\nDone. Run  python inference/vocoder_infer.py  to generate audio.")