import os, csv
import numpy as np
import librosa
from datetime import datetime

GT_FOLDER  = "hindi_dataset/audio"
GEN_FOLDER = "outputs"
LOG_FILE   = "logs/mcd_scores.csv"
SR         = 22050
N_MFCC     = 13
HOP_LENGTH = 256

os.makedirs("logs", exist_ok=True)

def compute_mcd(gt_path, gen_path):
    gt_audio,  _ = librosa.load(gt_path,  sr=SR)
    gen_audio, _ = librosa.load(gen_path, sr=SR)

    gt_mfcc  = librosa.feature.mfcc(y=gt_audio,  sr=SR,
                                     n_mfcc=N_MFCC+1, hop_length=HOP_LENGTH)[1:]
    gen_mfcc = librosa.feature.mfcc(y=gen_audio, sr=SR,
                                     n_mfcc=N_MFCC+1, hop_length=HOP_LENGTH)[1:]

    gt_mfcc  = (gt_mfcc  - gt_mfcc.mean(axis=1,  keepdims=True)) \
               / (gt_mfcc.std(axis=1,  keepdims=True) + 1e-8)
    gen_mfcc = (gen_mfcc - gen_mfcc.mean(axis=1, keepdims=True)) \
               / (gen_mfcc.std(axis=1, keepdims=True) + 1e-8)

    min_len  = min(gt_mfcc.shape[1], gen_mfcc.shape[1])
    gt_mfcc  = gt_mfcc[:,  :min_len]
    gen_mfcc = gen_mfcc[:, :min_len]

    diff      = gt_mfcc - gen_mfcc
    per_frame = np.sqrt(2.0 * np.sum(diff ** 2, axis=0))
    mcd       = (10.0 / np.log(10.0)) * np.mean(per_frame)

    return round(float(mcd), 4)

gt_files = sorted([f for f in os.listdir(GT_FOLDER) if f.endswith(".wav")])

if not gt_files:
    raise FileNotFoundError(f"No ground truth .wav files in {GT_FOLDER}")

log_f  = open(LOG_FILE, "w", newline="")
writer = csv.writer(log_f)
writer.writerow(["clip", "mcd_db", "timestamp"])

scores    = []
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"Computing MCD scores for {len(gt_files)} clips...")
print(f"{'Clip':<15} {'MCD (dB)':>10}")
print("-" * 28)

for fname in gt_files:
    name     = fname.replace(".wav", "")
    gt_path  = os.path.join(GT_FOLDER,  fname)
    gen_path = os.path.join(GEN_FOLDER, fname)

    if not os.path.exists(gen_path):
        print(f"  [skip] {name} — no generated file in {GEN_FOLDER}/")
        writer.writerow([name, "N/A", timestamp])
        continue

    try:
        mcd = compute_mcd(gt_path, gen_path)
        scores.append(mcd)
        writer.writerow([name, mcd, timestamp])
        print(f"  {name:<13} {mcd:>10.4f} dB")
    except Exception as e:
        print(f"  [error] {name}: {e}")
        writer.writerow([name, "ERROR", timestamp])

if scores:
    avg = round(float(np.mean(scores)), 4)
    mn  = round(float(np.min(scores)),  4)
    mx  = round(float(np.max(scores)),  4)

    writer.writerow([])
    writer.writerow(["AVERAGE", avg, timestamp])
    writer.writerow(["MIN",     mn,  timestamp])
    writer.writerow(["MAX",     mx,  timestamp])

    print("-" * 28)
    print(f"  Average MCD : {avg:.4f} dB")
    print(f"  Min MCD     : {mn:.4f} dB")
    print(f"  Max MCD     : {mx:.4f} dB")
    print()
    if avg < 8.0:
        print("  Result: GOOD — below 8 dB threshold")
    elif avg < 15.0:
        print("  Result: MODERATE — typical for unit-based systems")
    else:
        print("  Result: POOR — large modality gap (expected given MelProjector limitations)")

log_f.close()
print(f"\nMCD scores saved → {LOG_FILE}")