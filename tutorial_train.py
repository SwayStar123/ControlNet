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

class BucketSampler(Sampler):
    def __init__(self, dataset, bucket_file, batch_size, seed=42):
        self.dataset = dataset
        self.bucket_manager = BucketManager(bucket_file, bsz=batch_size, seed=seed)
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(self.bucket_manager.batch_total):
            batch_ids, _ = self.bucket_manager.get_batch()
            yield batch_ids

    def __len__(self):
        return self.bucket_manager.batch_total * self.batch_size


class BucketedDataLoader(DataLoader):
    def __init__(self, dataset, bucket_file, batch_size, seed=42, **kwargs):
        sampler = BucketSampler(dataset, bucket_file, batch_size, seed)
        super().__init__(dataset, batch_sampler=sampler, **kwargs)

# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 2
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
    dirpath='./models',  # save checkpoints in this directory
    filename='best_checkpoint',  # prefix of checkpoint file names
    monitor='val_loss',  # monitor validation loss to determine best checkpoint
    mode='min',  # minimize validation loss
    save_top_k=1  # save only the best checkpoint
)

# Create train and validation datasets
dataset = MyDataset()
train_indices, val_indices = dataset.generate_resolutions_pickle_files("train_resolutions.pkl", "val_resolutions.pkl", 32)
train_dataset = MyDataset(indices=train_indices)
val_dataset = MyDataset(indices=val_indices)

# DataLoaders
train_bucketed_dataloader = BucketedDataLoader(train_dataset, "train_resolutions.pkl", batch_size=batch_size)
val_bucketed_dataloader = BucketedDataLoader(val_dataset, "val_resolutions.pkl", batch_size=batch_size)

logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])

# Train and validate!
trainer.fit(model, train_dataloader=train_bucketed_dataloader, val_dataloaders=val_bucketed_dataloader)
