import os
import random
from glob import glob

from PIL import Image

import torch

random.seed(42)
torch.manual_seed(42)

classes = [
    "back_pack",
    "bike",
    "bike_helmet",
    "bookcase",
    "bottle",
    "calculator",
    "desk_chair",
    "desk_lamp",
    "desktop_computer",
    "file_cabinet",
    "headphones",
    "keyboard",
    "laptop_computer",
    "letter_tray",
    "mobile_phone",
    "monitor",
    "mouse",
    "mug",
    "paper_notebook",
    "pen",
    "phone",
    "printer",
    "projector",
    "punchers",
    "ring_binder",
    "ruler",
    "scissors",
    "speaker",
    "stapler",
    "tape_dispenser",
    "trash_can",
]


class Amazon(torch.utils.data.Dataset):
    def __init__(self, path, transforms=None, batch_size=16):
        self.path = path
        self.files = glob(os.path.join(path, "**", "*.jpg"), recursive=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        img = Image.open(file)

        label = file.split(self.path)[-1].split("/")[-2] # .split("/")[0]
        label = classes.index(label)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label


class Webcam(torch.utils.data.Dataset):
    def __init__(self, path, transforms=None, batch_size=16):
        self.path = path
        self.files = glob(os.path.join(path, "**", "*.jpg"), recursive=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        img = Image.open(file)

        label = file.split(self.path)[-1].split("/images/")[-1].split("/")[0]
        label = classes.index(label)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label


class DSLR(torch.utils.data.Dataset):
    def __init__(self, path, transforms=None, batch_size=16):
        self.path = path
        self.files = glob(os.path.join(path, "**", "*.jpg"), recursive=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        img = Image.open(file)

        label = file.split(self.path)[-1].split("/images/")[-1].split("/")[0]
        label = classes.index(label)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label