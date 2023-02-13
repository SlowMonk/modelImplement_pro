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
    path2data = '/media/jk_dataset/emocog_data/data/stl'

    if not os.path.exists(path2data):
        os.mkdir(path2data)

    # load dataset
    train_ds = datasets.STL10(path2data, split='train', download=True, transform=transforms.ToTensor())
    val_ds = datasets.STL10(path2data, split='test', download=True, transform=transforms.ToTensor())

    print(len(train_ds))
    print(len(val_ds))
    #print(np.array(train_ds)[0][0].shape)

    return train_ds, val_ds


def calculate_normalize(train_ds, val_ds):
    
    # To normalize the dataset, calculate the mean and std
    # To normalize the dataset, calculate the mean and std
    x, _ = train_ds[0]
    print('train_ds->',np.array(x).shape)
    train_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in train_ds]
    train_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in train_ds]

    train_meanR = np.mean([m[0] for m in train_meanRGB])
    train_meanG = np.mean([m[1] for m in train_meanRGB])
    train_meanB = np.mean([m[2] for m in train_meanRGB])
    train_stdR = np.mean([s[0] for s in train_stdRGB])
    train_stdG = np.mean([s[1] for s in train_stdRGB])
    train_stdB = np.mean([s[2] for s in train_stdRGB])


    val_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in val_ds]
    val_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in val_ds]

    val_meanR = np.mean([m[0] for m in val_meanRGB])
    val_meanG = np.mean([m[1] for m in val_meanRGB])
    val_meanB = np.mean([m[2] for m in val_meanRGB])

    val_stdR = np.mean([s[0] for s in val_stdRGB])
    val_stdG = np.mean([s[1] for s in val_stdRGB])
    val_stdB = np.mean([s[2] for s in val_stdRGB])

    print(train_meanR, train_meanG, train_meanB)
    print(val_meanR, val_meanG, val_meanB)

    return train_meanR, train_meanG, train_meanB, train_stdR, train_stdG, train_stdB

def main():
    train_ds, val_ds = load_data()
    train_meanR, train_meanG, train_meanB, train_stdR, train_stdG, train_stdB = calculate_normalize(train_ds, val_ds)

    # define the image transformation
    # using FiveCrop, normalize, horizontal reflection
    train_transformer = transforms.Compose([
                        transforms.Resize(256),
                        transforms.FiveCrop(224),
                        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                        transforms.Normalize([train_meanR, train_meanG, train_meanB], [train_stdR, train_stdG, train_stdB]),
    ])

    # test_transformer = transforms.Compose([
    #                     transforms.ToTensor(),
    #                     transforms.Resize(224),
    #                     transforms.Normalize([train_meanR, train_meanG, train_meanB], [train_stdR, train_stdG, train_stdB]),
    # ])

    # apply transformation
    train_ds.transform = train_transformer
    val_ds.transform = train_transformer

    # create dataloader
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=True)

    print(f'train_dl->{len(train_dl)}, val_dl->{len(val_dl)}')



if __name__ == "__main__":
    main()