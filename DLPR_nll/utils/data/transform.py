import torch
import torchvision.transforms as T

from PIL import Image
import numpy as np


class PILToTensor(object):
    def __call__(self, pic):
        assert isinstance(pic, Image.Image), "PILToTensor: Please input a PIL image."

        img = torch.as_tensor(np.array(pic))
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        img = img.permute((2,0,1)).float()
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


def build_transforms(transform_type):

    if transform_type == "p64":
        transform = T.Compose([
            T.RandomCrop(64),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            PILToTensor()
        ])
    elif transform_type == "p64_centercrop":
        transform = T.Compose([
            T.CenterCrop(64),
            PILToTensor()
        ])

    else:
        raise Exception("No existing transform type {}.".format(transform_type))

    return transform