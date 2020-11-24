# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
from os import listdir
from os import path
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import ConcatDataset, Dataset, DataLoader, random_split
import torchvision.transforms
from PIL import Image
import PIL.ImageOps
import numpy as np

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, grayscale=False):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.grayscale = grayscale
        self.ids = [path.basename(f) for f in listdir(self.masks_dir)]

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = path.join(self.imgs_dir, idx)
        mask_file = path.join(self.masks_dir, idx)
        img = Image.open(img_file)
        if self.grayscale:
            img = PIL.ImageOps.grayscale(img)
            img = torch.from_numpy(np.array(img)).type(torch.FloatTensor).unsqueeze(0)
        else:
            img = torch.from_numpy(np.array(img).transpose((2, 0, 1))).type(torch.FloatTensor)
        mask = Image.open(mask_file)
        mask = torch.from_numpy(np.array(mask)).type(torch.FloatTensor).unsqueeze(0) / 255
        return {
            'img': img,
            'mask': mask,
        }

    def __len__(self):
        return len(self.ids)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, input_channels=3):
        filters = 32

        super().__init__()
        self.encoder1 = ConvBlock(input_channels, filters)
        self.encoder2 = ConvBlock(filters, 2 * filters)
        self.encoder3 = ConvBlock(2 * filters, 4 * filters)
        self.encoder4 = ConvBlock(4 * filters, 8 * filters)

        self.decoder1 = ConvBlock(12 * filters, 4 * filters)
        self.decoder2 = ConvBlock(6 * filters, 2 * filters)
        self.decoder3 = ConvBlock(3 * filters, filters)
        self.decoder4 = nn.Conv2d(filters, 1, kernel_size=1)

        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.down(x1))
        x3 = self.encoder3(self.down(x2))
        x4 = self.encoder4(self.down(x3))

        d1 = self.decoder1(torch.cat([self.up(x4), x3], axis=1))
        d2 = self.decoder2(torch.cat([self.up(d1), x2], axis=1))
        d3 = self.decoder3(torch.cat([self.up(d2), x1], axis=1))
        return self.decoder4(d3)

def train_unet(epochs=10):
    dataset = ConcatDataset([
        BasicDataset('data\cap1\img', 'data\cap1\mask'),
        BasicDataset('data\cap3\img', 'data\cap3\mask'),
        BasicDataset('data\cap4\img', 'data\cap4\mask'),
        BasicDataset('data\cap5\img', 'data\cap5\mask'),
    ])

    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    augmentation_transform = torchvision.transforms.ColorJitter(
        brightness=0.5, contrast=0, saturation=0, hue=0.25
    )

    net = UNet()
    optimizer = optim.RMSprop(net.parameters(), lr=1e-3, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    criterion = nn.BCEWithLogitsLoss()

    device = torch.device('cuda')
    net.to(device=device)

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0.0

        print(f"epoch {epoch + 1} of {epochs}")
        for batch in train_loader:
            imgs = batch['img']
            masks = batch['mask']

            imgs = imgs.to(device=device)
            masks = masks.to(device=device)

            masks_pred = net(imgs)
            loss = criterion(masks_pred, masks)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()

        val_loss = 0.0
        for batch in val_loader:
            imgs = batch['img']
            masks = batch['mask']

            imgs = imgs.to(device=device)
            masks = masks.to(device=device)

            with torch.no_grad():
                masks_pred = net(imgs)

            val_loss += criterion(masks_pred, masks).item()

        print(f"epoch loss {epoch_loss}, validation loss {val_loss}")
        scheduler.step(val_loss)
        for param_group in optimizer.param_groups:
            print(f"learning rate {param_group['lr']}")

    return net

    # img = Image.open('data/cap1/img/086.png')
    # img = torch.from_numpy(np.array(img).transpose((2, 0, 1))).type(torch.FloatTensor)
    # img = img.to(device=device)
    # pred = net(img.unsqueeze(0)).squeeze(0)
    # print(pred)
    # cv2.imshow('frame', img.cpu().numpy())
    # cv2.imshow('pred', pred.cpu().numpy())
