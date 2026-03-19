import cv2
import numpy as np

def load_video(path):

    cap = cv2.VideoCapture(path)

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (96,96))

        frame = frame / 255.0

        frames.append(frame)

    cap.release()

    return np.array(frames)