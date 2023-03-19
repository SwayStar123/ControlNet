import numpy as np
from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Sampler
from torch.utils.data import random_split
from bucketmanager import BucketManager

# This shit works but not for fsdp
# class BucketSampler(Sampler):
#     def __init__(self, dataset, bucket_file, batch_size, seed=42):
#         self.dataset = dataset
#         self.bucket_manager = BucketManager(bucket_file, bsz=batch_size, seed=seed)
#         self.batch_size = batch_size

#     def __iter__(self):
#         for _ in range(self.bucket_manager.batch_total):
#             batch_ids, _ = self.bucket_manager.get_batch()
#             # Check if all images in the batch have the same resolution
#             if self._has_same_resolution(batch_ids):
#                 yield batch_ids

#     def __len__(self):
#         return self.bucket_manager.batch_total * self.batch_size
    
#     def _has_same_resolution(self, batch_ids):
#         resolutions = [self.bucket_manager.get_resolution(i) for i in batch_ids]
#         return all(res == resolutions[0] for res in resolutions)

# class BucketedDataLoader(DataLoader):
#     def __init__(self, dataset, bucket_file, batch_size, seed=42, **kwargs):
#         sampler = BucketSampler(dataset, bucket_file, batch_size, seed)
#         super().__init__(dataset, batch_sampler=sampler, **kwargs)

# Configs
resume_path = './models/control_any4.5_ini.ckpt'
batch_size = 3
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Define the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath='./trained_models',  # save checkpoints in this directory
    filename='checkpoint_{epoch}',  # checkpoint file names include the epoch number
    every_n_epochs=1,  # save a checkpoint every epoch
)

# Create train and validation datasets
dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, checkpoint_callback])

# Train and validate!
trainer.fit(model, dataloader)
