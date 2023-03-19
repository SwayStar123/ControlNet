import json
import cv2
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Load dataset metadata
with open('./training/next_frame_dataset2/frame_metadata2.json', 'rt') as f:
    data = json.load(f)

# Count resolutions
resolution_count = defaultdict(int)

# Function to process image resolutions
def process_resolution(idx_str, item):
    hint_filename = './training/next_frame_dataset2/next_frame_dataset/first_frame/' + idx_str + '.jpg'
    hint = cv2.imread(hint_filename)
    resolution = hint.shape[:2]
    return resolution

# Use ThreadPoolExecutor to process images in parallel
with ThreadPoolExecutor(max_workers=32) as executor:
    future_to_resolution = {executor.submit(process_resolution, idx_str, item): idx_str for idx_str, item in data.items()}

    # Display progress bar
    for future in tqdm(as_completed(future_to_resolution), total=len(data)):
        resolution = future.result()
        resolution_count[resolution] += 1

# Plot resolution distribution
resolutions = list(resolution_count.keys())
counts = list(resolution_count.values())

# Convert resolutions to strings
resolutions_str = [f'{res[0]}x{res[1]}' for res in resolutions]

plt.bar(range(len(resolutions)), counts, tick_label=resolutions_str)
plt.xticks(rotation=90)
plt.xlabel('Resolution')
plt.ylabel('Frequency')
plt.title('Resolution Distribution')
plt.show()
