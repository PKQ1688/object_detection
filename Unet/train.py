# -*- coding:utf-8 -*-
# @author :adolf
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import UNetSegmentationDataset as Dataset
from loss import DiceLoss
from transforms import get_transforms
from unet_model import UNet
from utils import dsc
import logging

import cv2


class Train(object):
    def __init__(self, configs):
        self.batch_size = configs.get("batch_size", "16")
        self.epochs = configs.get("epochs", "100")
        self.lr = configs.get("lr", "0.0001")

        device_args = configs.get("device", "cuda")
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

        self.is_resize = config.get("is_resize", False)
        self.image_short_side = config.get("image_short_side", 256)

        self.is_padding = config.get("is_padding", False)

        # self.image_size = configs.get("image_size", "256")
        # self.aug_scale = configs.get("aug_scale", "0.05")
        # self.aug_angle = configs.get("aug_angle", "15")

        self.step = 0

        self.dsc_loss = DiceLoss()
        self.model = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
        self.model.to(self.device)

    def datasets(self):
        train_datasets = Dataset(images_dir=self.images_path,
                                 # image_size=self.image_size,
                                 subset="train",  # train
                                 transform=get_transforms(train=True),
                                 is_resize=self.is_resize,
                                 image_short_side=self.image_short_side,
                                 is_padding=self.is_padding
                                 )
        # valid_datasets = train_datasets

        valid_datasets = Dataset(images_dir=self.images_path,
                                 # image_size=self.image_size,
                                 subset="validation",  # validation
                                 transform=get_transforms(train=False),
                                 is_resize=self.is_resize,
                                 image_short_side=self.image_short_side,
                                 is_padding=False
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
            batch_size=1,
            drop_last=False,
            num_workers=self.workers,
        )

        return loader_train, loader_valid

    @staticmethod
    def dsc_per_volume(validation_pred, validation_true):
        assert len(validation_pred) == len(validation_true)
        dsc_list = []
        for p in range(len(validation_pred)):
            y_pred = np.array([validation_pred[p]])
            y_true = np.array([validation_true[p]])
            dsc_list.append(dsc(y_pred, y_true))
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

    def train_one_epoch(self, optimizer, data_loader, epoch):
        self.model.train()
        loss_train = []
        for i, data in enumerate(data_loader):
            x, y_true = data
            x, y_true = x.to(self.device), y_true.to(self.device)

            y_pred = self.model(x)
            # print('1111', y_pred.size())
            # print('2222', y_true.size())
            loss = self.dsc_loss(y_pred, y_true)

            loss_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.step % 200 == 0:
                print('Epoch:[{}/{}]\t iter:[{}]\t loss={:.5f}\t '.format(epoch, self.epochs, i, loss))

            self.step += 1

    def eval_model(self, data_loader, best_validation_dsc):
        self.model.eval()
        loss_valid = []

        validation_pred = []
        validation_true = []

        for i, data in enumerate(data_loader):
            x, y_true = data
            x, y_true = x.to(self.device), y_true.to(self.device)

            # print(x.size())
            # print(333,x[0][2])
            with torch.no_grad():
                y_pred = self.model(x)
                loss = self.dsc_loss(y_pred, y_true)

            # print(y_pred.shape)
            mask = y_pred > 0.5
            mask = mask * 255
            mask = mask.cpu().numpy()[0][0]
            # print(mask)
            # print(mask.shape())
            cv2.imwrite('result.png', mask)

            loss_valid.append(loss.item())

            y_pred_np = y_pred.detach().cpu().numpy()

            validation_pred.extend(
                [y_pred_np[s] for s in range(y_pred_np.shape[0])]
            )
            y_true_np = y_true.detach().cpu().numpy()
            validation_true.extend(
                [y_true_np[s] for s in range(y_true_np.shape[0])]
            )

        mean_dsc = np.mean(
            self.dsc_per_volume(
                validation_pred,
                validation_true,
            )
        )
        # print('mean_dsc:', mean_dsc)
        if mean_dsc > best_validation_dsc:
            best_validation_dsc = mean_dsc
            torch.save(self.model.state_dict(), os.path.join(self.weights, "unet_idcard_1.pth"))
            print("Best validation mean DSC: {:4f}".format(best_validation_dsc))

    def main(self):
        # print('train is begin.....')
        loader_train, loader_valid = self.data_loaders()
        # print('load data end.....')

        # loaders = {"train": loader_train, "valid": loader_valid}

        best_validation_dsc = 0.0

        params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = optim.Adam(params, lr=self.lr)

        for epoch in tqdm(range(self.epochs), total=self.epochs):
            self.train_one_epoch(optimizer, loader_train, epoch)
            self.eval_model(loader_valid, best_validation_dsc)


if __name__ == '__main__':
    import yaml

    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    with open('config.yaml', 'r') as fp:
        config = yaml.load(fp.read(), Loader=yaml.FullLoader)

    trainer = Train(configs=config)
    trainer.main()
