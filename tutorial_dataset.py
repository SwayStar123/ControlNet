import json
import cv2
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, indices=None):
        with open('./training/next_frame_dataset2/frame_metadata.json', 'rt') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx_str = str(idx)
        item = self.data[idx_str]

        hint_filename = './training/next_frame_dataset2/first_frame/' + idx_str + '.jpg'
        target_filename = './training/next_frame_dataset2/next_frame/' + idx_str + '.jpg'
        prompt = item

        hint = cv2.imread(hint_filename)
        target = cv2.imread(target_filename)

        hint = cv2.cvtColor(hint, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        hint = hint.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=hint)
    
#     def generate_resolutions_pickle_files(self, train_output_file, val_output_file, num_threads=8, val_split=0.2):
#         train_indices, val_indices = split_indices(self, val_split=val_split)

#         train_dataset = MyDataset(indices=train_indices)
#         val_dataset = MyDataset(indices=val_indices)

#         train_resolutions = get_resolutions(train_dataset, num_threads)
#         val_resolutions = get_resolutions(val_dataset, num_threads)

#         save_resolutions_to_pickle(train_resolutions, train_output_file)
#         save_resolutions_to_pickle(val_resolutions, val_output_file)

#         print(f"Train resolutions saved to '{train_output_file}'")
#         print(f"Validation resolutions saved to '{val_output_file}'")

#         return train_indices, val_indices

# def split_indices(dataset, val_split=0.2, random_seed=42):
#     n_val = int(len(dataset) * val_split)
#     n_train = len(dataset) - n_val
#     indices = np.arange(len(dataset))
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)
#     return indices[:n_train], indices[n_train:]

# def calculate_resolution(dataset, idx):
#     item = dataset[idx]
#     hint = item['hint']
#     h, w, _ = hint.shape
#     return (w, h), idx

# def get_resolutions(dataset, num_threads=8):
#     resolutions = {}

#     with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
#         future_to_idx = {executor.submit(calculate_resolution, dataset, idx): idx for idx in range(len(dataset))}
#         for future in tqdm(concurrent.futures.as_completed(future_to_idx), desc="Calculating resolutions", total=len(dataset)):
#             resolution, idx = future.result()
#             resolutions[idx] = resolution

#     return resolutions


# def save_resolutions_to_pickle(resolutions, output_file):
#     with open(output_file, 'wb') as f:
#         pickle.dump(resolutions, f)

# if __name__ == '__main__':
#     dataset = MyDataset()
#     resolutions = get_resolutions(dataset, 32)
#     save_resolutions_to_pickle(resolutions, 'resolutions.pkl')
#     print("Resolutions saved to 'resolutions.pkl'")
