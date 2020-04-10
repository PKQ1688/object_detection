# -*- coding:utf-8 -*-
# @author :adolf
import argparse
import json
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import UNetSegmentationDataset as Dataset
from loss import DiceLoss
from transforms import transforms
from unet_model import UNet
from utils import log_images, dsc
import logging


# from config import *


class Train(object):
    def __int__(self, configs):
        self.batch_size = configs.get("batch_size", "16")
        self.epochs = configs.get("epochs", "100")
        self.lr = configs.get("lr", "0.0001")

        device_args = configs.get("device", "cuda:0")
        self.device = torch.device("cpu" if not torch.cuda.is_available() else device_args)

        self.workers = configs.get("workers", "4")

        self.vis_images = configs.get("vis_images", "200")
        self.vis_freq = configs.get("vis_freq", "10")

        self.weights = configs.get("weights", "./weights")
        if not os.path.exists(self.weights):
            os.mkdir(self.weights)

        self.logs = configs.get("logs", "./logs")
        if not os.path.exists(self.weights):
            os.mkdir(self.weights)

        self.images_path = configs.get("images_path", "./data")

        self.image_size = configs.get("image_size", "256")
        self.aug_scale = configs.get("aug_scale", "0.05")
        self.aug_angle = configs.get("aug_angle", "15")

        self.step = 0

    def datasets(self):
        train_datasets = Dataset(images_dir=self.images_path,
                                 image_size=self.image_size,
                                 transform=transforms(scale=self.aug_scale, angle=self.aug_angle, flip_prob=0.5,
                                                      is_train=True),
                                 )

        valid_datasets = Dataset(images_dir=self.images_path,
                                 image_size=self.image_size,
                                 transform=transforms(scale=self.aug_scale, angle=self.aug_angle, flip_prob=0.5,
                                                      is_train=False),
                                 )

        return train_datasets, valid_datasets

    def data_loaders(self):
        dataset_train, dataset_valid = self.datasets()

        loader_train = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.workers,
        )
        loader_valid = DataLoader(
            dataset_valid,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.workers,
        )

        return loader_train, loader_valid

    @staticmethod
    def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
        dsc_list = []
        num_slices = np.bincount([p[0] for p in patient_slice_index])
        index = 0
        for p in range(len(num_slices)):
            y_pred = np.array(validation_pred[index: index + num_slices[p]])
            y_true = np.array(validation_true[index: index + num_slices[p]])
            dsc_list.append(dsc(y_pred, y_true))
            index += num_slices[p]
        return dsc_list

    @staticmethod
    def get_logger(filename, verbosity=1, name=None):
        level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
        formatter = logging.Formatter(
            "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
        )
        logger = logging.getLogger(name)
        logger.setLevel(level_dict[verbosity])

        fh = logging.FileHandler(filename, "w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        return logger

    def train_one_epoch(self, model, optimizer, data_loader, dsc_loss, loss_train, epoch, max_epoch):
        model.train()

        for i, data in enumerate(data_loader):
            self.step += 1
            x, y_true = data
            x, y_true = x.to(self.device), y_true.to(self.device)

            y_pred = model(x)
            loss = dsc_loss(y_pred, y_true)

            loss_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.step % 50 == 0:
                logging.info('Epoch:[{}/{}]\t iter:[{}]\t loss={:.5f}\t '.format(epoch, max_epoch, i, loss))

    def main(self):
        loader_train, loader_valid = self.data_loaders()

        # loaders = {"train": loader_train, "valid": loader_valid}

        unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
        unet.to(self.device)

        dsc_loss = DiceLoss()
        best_validation_dsc = 0.0

        params = [p for p in unet.parameters() if p.requires_grad]

        optimizer = optim.Adam(unet.parameters(), lr=self.lr)

        loss_train = []
        loss_valid = []

        for epoch in tqdm(range(self.epochs), total=self.epochs):
            self.train_one_epoch(unet, optimizer, loader_train, dsc_loss, loss_train, epoch, self.epochs)


if __name__ == '__main__':
    import yaml
    with open('config.yaml', 'r') as fp:
        config = yaml.load(fp.read())

    trainer = Train()
    trainer.main()
