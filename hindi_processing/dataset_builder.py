import os
import subprocess

BASE_DIR = "D:/VTS/hindi_dataset"
video_folder = os.path.join(BASE_DIR, "video")
audio_folder = os.path.join(BASE_DIR, "audio")
label_folder = os.path.join(BASE_DIR, "labels")

os.makedirs(audio_folder, exist_ok=True)
os.makedirs(label_folder, exist_ok=True)

videos = os.listdir(video_folder)

for video in videos:
    if video.endswith(".mp4"):
        video_path = os.path.join(video_folder, video)
        name = os.path.splitext(video)[0]
        audio_path = os.path.join(audio_folder, name + ".wav")

        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-ar", "16000",
            "-ac", "1",
            audio_path
        ])

        label_path = os.path.join(label_folder, name + ".txt")

        if not os.path.exists(label_path):
            open(label_path, "w", encoding="utf-8").close()

metadata_path = os.path.join(BASE_DIR, "metadata.csv")

with open(metadata_path, "w", encoding="utf-8") as f:
    for file in os.listdir(audio_folder):
        if file.endswith(".wav"):
            name = file.replace(".wav", "")
            label_file = os.path.join(label_folder, name + ".txt")

            text = ""
            if os.path.exists(label_file):
                with open(label_file, "r", encoding="utf-8") as lf:
                    text = lf.read().strip()

            f.write(f"{name}|audio/{file}|{text}\n")

print("Dataset automation completed.")