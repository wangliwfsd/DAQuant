import os
from tqdm import tqdm
import glob
import numpy as np
import torch
import random
from PIL import Image
from transfer_data_loader import Amazon, DSLR, Webcam
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_loaders import *
from office_home_dataset import get_art, get_clipart, get_product, get_real_word

data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_imagenet(nsamples):
    print("get_imagenet")
    folder_path = '/PATH/TO/IMAGENET/Calibration'

    image_files = glob.glob(os.path.join(folder_path, '*.JPEG'))

    random.shuffle(image_files)
    
    trainloader = []
    testenc = []
    for i in range(nsamples):
        image = Image.open(image_files[i])
        trainloader.append(image)
        testenc.append(image)
    return trainloader, testenc


def get_amazon(nsamples):
    print("get amazon")
    train_dataset = Amazon(path='/PATH/TO/OFFICE-31/AMAZON', transforms=data_transforms['test'])
    # create data loader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    trainloader = []
    testenc = []
    i = 0
    for images, labels in tqdm(train_loader,total=nsamples):
        i = i + 1
        trainloader.append(images)
        testenc.append(images)
        if i == nsamples:
            break
    return trainloader, testenc

def get_dslr(nsamples):
    print("get dslr")
    train_dataset = DSLR(path='/PATH/TO/OFFICE-31/DSLR', transforms=data_transforms['test'])
    
    # create data loader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)    
    trainloader = []
    testenc = []
    i = 0
    for images, labels in tqdm(train_loader,total=nsamples):
        i = i + 1
        trainloader.append(images)
        testenc.append(images)
        if i == nsamples:
            break
    return trainloader, testenc

def get_webcam(nsamples):
    print("get webcam")
    train_dataset = Webcam(path='/PATH/TO/OFFICE-31/WEBCAM', transforms=data_transforms['test'])

    # create data loader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)    
    
    trainloader = []
    testenc = []
    i = 0
    for images, labels in tqdm(train_loader,total=nsamples):
        i = i + 1
        trainloader.append(images)
        testenc.append(images)
        if i == nsamples:
            break
    return trainloader, testenc

def load_art(nsamples):
    print("get office-home art")

    # create data loader
    train_loader, val_loader = get_art() 
    
    trainloader = []
    testenc = []
    i = 0
    for images, labels in tqdm(train_loader,total=nsamples):
        i = i + 1
        trainloader.append(images)
        testenc.append(images)
        if i == nsamples:
            break
    return trainloader, testenc

def load_clipart(nsamples):
    print("get office-home clipart")

    # create data loader
    train_loader, val_loader = get_clipart() 
    
    trainloader = []
    testenc = []
    i = 0
    for images, labels in tqdm(train_loader,total=nsamples):
        i = i + 1
        trainloader.append(images)
        testenc.append(images)
        if i == nsamples:
            break
    return trainloader, testenc

def load_product(nsamples):
    print("get office-home product")

    # create data loader
    train_loader, val_loader = get_product() 
    
    trainloader = []
    testenc = []
    i = 0
    for images, labels in tqdm(train_loader,total=nsamples):
        i = i + 1
        trainloader.append(images)
        testenc.append(images)
        if i == nsamples:
            break
    return trainloader, testenc

def load_real_word(nsamples):
    print("get office-home real word")

    # create data loader
    train_loader, val_loader = get_real_word() 
    
    trainloader = []
    testenc = []
    i = 0
    for images, labels in tqdm(train_loader,total=nsamples):
        i = i + 1
        trainloader.append(images)
        testenc.append(images)
        if i == nsamples:
            break
    return trainloader, testenc


def get_loaders(
    name, nsamples=32
):
    if 'ImageNet' in name:
        return get_imagenet(nsamples)
    elif 'amazon' in name:
        return get_amazon(nsamples)
    elif 'dslr' in name:
        return get_dslr(nsamples)
    elif 'webcam' in name:
        return get_webcam(nsamples)
    elif 'clipart' in name:
        return load_clipart(nsamples)
    elif 'art' in name:
        return load_art(nsamples)
    elif 'product' in name:
        return load_product(nsamples)
    elif 'real_word' in name:
        return load_real_word(nsamples)
