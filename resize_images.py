import os
import json
import cv2
import numpy as np
from collections import defaultdict
from shutil import copyfile

# Set directories
data_dir = './training/next_frame/next_frame_dataset/'
first_frame_dir = os.path.join(data_dir, 'first_frame')
next_frame_dir = os.path.join(data_dir, 'next_frame')
resized_first_frame_dir = os.path.join(data_dir, 'resized_first_frame')
resized_next_frame_dir = os.path.join(data_dir, 'resized_next_frame')

# Create resized image directories if not exist
os.makedirs(resized_first_frame_dir, exist_ok=True)
os.makedirs(resized_next_frame_dir, exist_ok=True)

# Load metadata
with open('./training/next_frame/frame_metadata.json', 'rt') as f:
    metadata = json.load(f)

# Find the resolutions of all the images
resolutions = defaultdict(int)

for idx, item in metadata.items():
    print("processing resolutions", idx)
    hint_filename = os.path.join(first_frame_dir, f"{idx}.jpg")
    hint = cv2.imread(hint_filename)
    resolutions[hint.shape[:2]] += 1

# Based on the resolutions of the images, find a single resolution to preserve the most information
target_resolution = max(resolutions, key=resolutions.get)

# Resize the remaining images and save them with new indexes
new_metadata = {}
new_idx = 0

for idx, item in metadata.items():
    print("processing images", idx, "target_resolution", target_resolution)
    hint_filename = os.path.join(first_frame_dir, f"{idx}.jpg")
    target_filename = os.path.join(next_frame_dir, f"{idx}.jpg")

    hint = cv2.imread(hint_filename)
    target = cv2.imread(target_filename)

    if hint.shape[0] >= target_resolution[0] and hint.shape[1] >= target_resolution[1]:
        # Crop the images to the target resolution
        y_offset = (hint.shape[0] - target_resolution[0]) // 2
        x_offset = (hint.shape[1] - target_resolution[1]) // 2

        cropped_hint = hint[y_offset:y_offset+target_resolution[0], x_offset:x_offset+target_resolution[1]]
        cropped_target = target[y_offset:y_offset+target_resolution[0], x_offset:x_offset+target_resolution[1]]

        # Save the resized images
        cv2.imwrite(os.path.join(resized_first_frame_dir, f"{new_idx}.jpg"), cropped_hint)
        cv2.imwrite(os.path.join(resized_next_frame_dir, f"{new_idx}.jpg"), cropped_target)

        # Update the metadata
        new_metadata[str(new_idx)] = item

        new_idx += 1

# Save the new metadata
with open('./training/next_frame/resized_frame_metadata.json', 'wt') as f:
    json.dump(new_metadata, f)
