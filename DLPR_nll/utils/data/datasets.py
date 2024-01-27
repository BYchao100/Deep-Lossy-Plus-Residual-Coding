import torch
from torch.utils import data

import os
import numpy as np

from PIL import Image


class ImageDataset(data.Dataset):

    def __init__(self, path_dir, img_mode = None, transform=None):
        self.path_dir = path_dir
        self.img_mode = img_mode
        self.transform = transform
        self.images = os.listdir(self.path_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        img_path = os.path.join(self.path_dir, image_name)
        img = Image.open(img_path)

        if self.img_mode is not None:
            img = img.convert(self.img_mode)

        if self.transform is not None:
            img = self.transform(img)

        return img

