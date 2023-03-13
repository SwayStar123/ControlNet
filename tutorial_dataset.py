import json
import cv2
import numpy as np

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

        hint_filename = './training/next_frame/next_frame_dataset/first_frame/' + idx + '.jpg'
        target_filename = './training/next_frame/next_frame_dataset/next_frame/' + idx + '.jpg'
        prompt = item

        hint = cv2.imread(hint_filename)
        target = cv2.imread(target_filename)

        # Do not forget that OpenCV read images in BGR order.
        hint = cv2.cvtColor(hint, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        hint = hint.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=hint)

