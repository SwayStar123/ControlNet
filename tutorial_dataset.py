import json
import cv2
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        with open('./training/next_frame/frame_metadata.json', 'rt') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)
        item = self.data[idx]

        hint_filename = './training/next_frame/next_frame_dataset_resized/first_frame/' + idx + '.png'
        target_filename = './training/next_frame/next_frame_dataset_resized/next_frame/' + idx + '.png'
        prompt = item

        hint = cv2.imread(hint_filename)
        target = cv2.imread(target_filename)

        hint = cv2.cvtColor(hint, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        hint = hint.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=hint)

def calculate_resolution(dataset, idx):
    item = dataset[idx]
    hint = item['hint']
    h, w, _ = hint.shape
    return (w, h), idx

def get_resolutions(dataset, num_threads=8):
    resolutions = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_idx = {executor.submit(calculate_resolution, dataset, idx): idx for idx in range(len(dataset))}
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), desc="Calculating resolutions", total=len(dataset)):
            resolution, idx = future.result()
            resolutions[idx] = resolution

    return resolutions


def save_resolutions_to_pickle(resolutions, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(resolutions, f)

if __name__ == '__main__':
    dataset = MyDataset()
    resolutions = get_resolutions(dataset, 32)
    save_resolutions_to_pickle(resolutions, 'resolutions.pkl')
    print("Resolutions saved to 'resolutions.pkl'")
