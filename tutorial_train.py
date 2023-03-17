import os
import pickle
import numpy as np
import torch

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from pytorch_lightning import LightningDataModule

from share import *
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

import numpy as np
import random

class BucketManager:
    def __init__(self, resolutions, bucket_size, seed=None):
        self.resolutions = resolutions
        self.bucket_size = bucket_size
        self.num_buckets = len(resolutions) // bucket_size
        self.buckets = self.create_buckets()
        if seed:
            random.seed(seed)

    def create_buckets(self):
        sorted_resolutions = sorted(self.resolutions, key=lambda x: x[1][1] / x[1][0])
        buckets = [sorted_resolutions[i:i + self.bucket_size] for i in range(0, len(sorted_resolutions), self.bucket_size)]
        return buckets

    def get_batch(self):
        selected_bucket = random.choice(self.buckets)
        ids, resolutions = zip(*selected_bucket)
        return ids, resolutions


class AspectRatioBucketingCallback(pl.Callback):
    def __init__(self, bucket_manager, dataset, batch_size):
        super().__init__()
        self.bucket_manager = bucket_manager
        self.dataset = dataset
        self.batch_size = batch_size

    def on_train_epoch_start(self, trainer, pl_module):
        data_loader = DataLoader(self.dataset, num_workers=0, batch_size=self.batch_size, collate_fn=self.custom_collate_fn)
        trainer.replace_dataloader('train', data_loader)

    def custom_collate_fn(self, batch):
        ids, resolutions = self.bucket_manager.get_batch()
        selected_data = [self.dataset[post_id] for post_id in ids]
        return torch.utils.data.dataloader.default_collate(selected_data)


class MyDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def setup(self, stage=None):
        # Split the dataset into training and validation sets
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])


# Configs
resume_path = './models/control_sd15_ini.ckpt'
bucket_file = 'resolutions.pkl'
batch_size = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# Load the resolutions list from a file
with open(bucket_file, "rb") as f:
    resolutions = pickle.load(f)

# Initialize the BucketManager
bucket_manager = BucketManager(resolutions, bucket_size=8, seed=42)

# First use CPU to load models. PyTorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = MyDataset()
data_module = MyDataModule(dataset, batch_size=batch_size)
logger = ImageLogger(batch_frequency=logger_freq)

# Define the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath='./models',  # save checkpoints in this directory
    filename='best_checkpoint',  # prefix of checkpoint file names
    monitor='val_loss',  # monitor validation loss to determine best checkpoint
    mode='min',  # minimize validation loss
    save_top_k=1  # save only the best checkpoint
)

# Initialize the AspectRatioBucketingCallback
aspect_ratio_bucketing_callback = AspectRatioBucketingCallback(bucket_manager, dataset, batch_size=batch_size)

trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, checkpoint_callback, aspect_ratio_bucketing_callback])

# Train!
trainer.fit(model, datamodule=data_module)
