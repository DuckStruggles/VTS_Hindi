import os
import subprocess

video_folder = "../hindi_dataset/video"
audio_folder = "../hindi_dataset/audio"
label_folder = "../hindi_dataset/labels"

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

        open(label_path, "w").close()

print("Dataset automation script completed.")