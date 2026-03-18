import numpy as np, os

VIDEO_FOLDER = "hindi_dataset/video"
UNITS_FOLDER = "hindi_dataset/units"

import cv2
for v in os.listdir(VIDEO_FOLDER)[:5]:
    name = v.replace(".mp4","")
    cap = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, v))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    units = open(os.path.join(UNITS_FOLDER, name+".units")).read().split()
    print(f"{name}: video_frames={n_frames}, fps={fps:.1f}, units={len(units)}, ratio={n_frames/len(units):.2f}")