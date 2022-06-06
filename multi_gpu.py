#!/usr/bin/env python
# coding: utf-8

# In[20]:


import argparse
import numpy as np
import os
import sys
import time
import warnings

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from monai.apps import DecathlonDataset
from monai.data import DataLoader, partition_dataset, decollate_batch, CacheDataset, ThreadDataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, compute_meandice
from monai.networks.nets import SegResNet, UNet
from monai.optimizers import Novograd
from monai.transforms import (
    Activations,
    AsDiscrete,
    AddChanneld,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    ToDeviced,
    EnsureTyped,
    EnsureType,
    ScaleIntensityRanged,
    Lambdad,
    Lambda
)
from monai.utils import set_determinism, first
from monai.engines import create_multigpu_supervised_trainer
import torch.nn as nn
import json
import matplotlib.pyplot as plt
import torch
import csv

# Ref:
# https://github.com/Project-MONAI/tutorials/blob/8b8c65e059253ae39218ce1a3b95762275cf81cc/acceleration/distributed_training/brats_training_ddp.py
# https://www.zhihu.com/question/67726969
# https://github.com/jia-zhuang/pytorch-multi-gpu-training


# In[11]:


config = {
    'lr': 1e-4,
    'epochs': 5000,
    'val_interval': 20,
    'batch_size': 8,
    'roi_size': (128,128,128)
}


# In[12]:


dataset_file = '/mount/config/dataset_ntuh.json'
data_root = '/mount/data/aorta_4/'
root_dir = '/mount/'

# Opening JSON file
with open(dataset_file) as json_file:
    dataset = json.load(json_file)

train_files = dataset['training']
val_files = dataset['validation']

def prefix_filename(file):
    file['image'] = data_root + file['image']
    file['label'] = data_root + file['label']
    return file

train_files = [prefix_filename(file) for file in train_files]
val_files = [prefix_filename(file) for file in val_files]

set_determinism(seed=0)


# In[13]:


train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image"]),
        # Lambdad(keys=["label"], func=lambda x: x[1:, :, :, :]),
        Spacingd(keys=["image", "label"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear", "nearest")),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        RandSpatialCropd(keys=["image", "label"],
                         roi_size=(128,128,128),
                         random_size=False
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=-300, a_max=300,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        RandFlipd(keys=["image", "label"],
                  prob=0.2,
                  spatial_axis=None),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image"]),
        # Lambdad(keys=["label"], func=lambda x: x[1:, :, :, :]),
        Spacingd(keys=["image", "label"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear", "nearest")),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        RandSpatialCropd(keys=["image", "label"],
                         roi_size=(128,128,128),
                         random_size=False
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=-300, a_max=300,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        RandFlipd(keys=["image", "label"],
                  prob=0.2,
                  spatial_axis=None),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    ]
)


# In[14]:


train_ds = CacheDataset(
    data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=32)
train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=32)
# train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=config['batch_size'], shuffle=True)

val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=16)
val_loader = DataLoader(val_ds, batch_size=config['batch_size'], num_workers=16)
# val_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=config['batch_size'], shuffle=True)


# In[15]:


check = False
if check:
    check_data = first(train_loader)
    print(check_data["image"].shape)
    print(check_data["label"].shape)


# In[16]:


if check:
    slices = 80
    image, label = (check_data["image"][0][0], check_data["label"][0])
    print(image.shape)
    print(label.shape)
    print(f"image shape: {image.shape}, label shape: {label.shape}")
    plt.figure("check", (10, 12))
    plt.subplot(3, 2, 1)
    plt.imshow(image[:, :, slices], cmap="gray")
    plt.title("image")

    image[0, 0, slices] = 10

    labels = ['heart', 'asc', 'arc', 'desc1', 'desc2']
    for i in range(5):
        plt.subplot(3, 2, i+2)
        plt.title(labels[i])
        plt.imshow(label[i, :, :, slices]*10 + image[:, :, slices], cmap='gray')
    plt.show()


# In[17]:


# initialize the distributed training process, every GPU runs in a process
# dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:1234', rank=0, world_size=4)
device = torch.device("cuda:0")
# torch.cuda.set_device(device)
# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()
torch.backends.cudnn.benchmark = True
model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=1,
    out_channels=5,
    dropout_prob=0.5,
)
'''
loss_function = DiceLoss(
    squared_pred=True,
    to_onehot_y=False,
    sigmoid=False,
    batch=False,
)
'''
loss_function = DiceFocalLoss(
    smooth_nr=1e-5,
    smooth_dr=1e-5,
    squared_pred=True,
    to_onehot_y=False,
    sigmoid=True,
    batch=True,
)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), 1e-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

# wrap the model with DistributedDataParallel module
# model = DistributedDataParallel(model, device_ids=[device])


# In[25]:


dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
# dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
post_pred = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
post_label = Compose([EnsureType(), AsDiscrete()])
label_name = ['heart', 'asc', 'arch', 'desc1', 'desc2']

epoch_loss_values = []
metric_values = []
# start a typical PyTorch training
best_metric = -1
best_metric_epoch = -1
train_start = time.time()


# In[26]:


for epoch in range(config['epochs']):
    epoch_start = time.time()
    # print("-" * 10)
    # print(f"epoch {epoch + 1}/{config['epochs']}")
    
    model.train()
    step = 0
    epoch_len = len(train_loader)
    epoch_loss = 0
    step_start = time.time()
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        # with torch.cuda.amp.autocast():
        #     outputs = model(batch_data["image"])
        #     loss = loss_function(outputs, batch_data["label"])
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        epoch_loss += loss.item()
        # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}, step time: {(time.time() - step_start):.4f}s")
        step_start = time.time()
    epoch_loss /= step
    epoch_loss_values.append([epoch_loss])
    lr_scheduler.step()

    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f},"
          f" time consuming: {(time.time() - epoch_start):.4f}s")

    if (epoch + 1) % config['val_interval'] == 0:        
        print("-" * 10)
        model.eval()
        data_number = 0
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size=(128,128,128)
                sw_batch_size = 1
                val_outputs = sliding_window_inference(
                    val_inputs,
                    roi_size,
                    sw_batch_size,
                    model
                )
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                # val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                dice_metric(y_pred=val_outputs, y=val_labels)
                data_number += 1
            # print(dice_metric.aggregate())
            dice_list = [0] * 5
            for i in range(5):
                dice_list[i] = dice_metric.aggregate()[i].item()
            dice_metric.reset()
            # print(dice_list)
            metric = sum(dice_list) / len(dice_list)
            metric_values.append(dice_list)

        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            # if dist.get_rank() == 0:
            torch.save(model.state_dict(), "best_metric_model.pth")
        print(
            f"current mean dice: {metric:.4f}"
            f"\nheart: {round(dice_list[0],4)}, asc: {round(dice_list[1],4)}, arch: {round(dice_list[2],4)}, desc1: {round(dice_list[3],4)}, desc2: {round(dice_list[4],4)}"
            f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
        )
        print("-" * 10)

with open('metric_values.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(metric_values)
with open('epoch_loss_values.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(epoch_loss_values)
print(
    f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch},"
    f" total train time: {(time.time() - train_start):.4f}"
)
# dist.destroy_process_group()


# In[ ]:




