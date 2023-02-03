# import package

# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim
from torch.optim.lr_scheduler import StepLR

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# display images
from torchvision import utils
import matplotlib.pyplot as plt

# utils
import numpy as np
from torchsummary import summary
import time
import copy

#load_data
def load_data():
    # if not exists the path, make the directory
    path2data = '/data/emocog_data/data/stl'

    if not os.path.exists(path2data):
        os.mkdir(path2data)

    # load dataset
    train_ds = datasets.STL10(path2data, split='train', download=True, transform=transforms.ToTensor())
    val_ds = datasets.STL10(path2data, split='test', download=True, transform=transforms.ToTensor())

    print(len(train_ds))
    print(len(val_ds))

def main():
    load_data()

if __name__ == "__main__":
    main()