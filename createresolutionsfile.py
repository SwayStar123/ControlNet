import pickle
import cv2
import json

from tutorial_dataset import MyDataset

dataset = MyDataset()

resolutions = []

for idx in range(len(dataset)):
    hint_filename = './training/next_frame/next_frame_dataset/first_frame/' + str(idx) + '.jpg'
    print(idx)
    hint = cv2.imread(hint_filename)
    height, width, _ = hint.shape
    resolutions.append((idx, (width, height)))

# Save the resolutions list to a file
with open("resolutions.pkl", "wb") as f:
    pickle.dump(resolutions, f)