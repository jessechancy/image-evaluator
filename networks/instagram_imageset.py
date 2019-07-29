from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
from preprocess import import_images
import matplotlib.pyplot as plt

##
IMG_SIZE = 224
##

def process_img(pic, mode):
    global IMG_SIZE
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    if mode == "train":
        preprocessing = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            normalize,
        ])
    elif mode == "val":
        preprocessing = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            normalize,
        ])
    return preprocessing(pic)

##

class InstagramImageDataset(Dataset):
    
    def __init__(self, transforms=None):
        #preprocess import
        self.images = import_images()
        print("31/2 check:",type(self.images[0][0]))
        self.transforms = transforms
        
    def __len__(self):
        print(len(self.images), "len")
        return len(self.images)
    
    def __getitem__(self, idx):
        score_class = self.images[idx][1]
        if self.transforms is not None:
            img = self.transforms(self.images[idx][0], "train")
            score_class = torch.from_numpy(np.array([int(score_class)]))
        return (img, score_class)
        
def instagram_imgset():
    ins_data = InstagramImageDataset(transforms=process_img)
    batch_size = 4
    validation_split = .2
    
    dataset_size = len(ins_data)
    indices = list(range(dataset_size))
    split = int(validation_split * dataset_size)
    train_indices, valid_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    
    train_loader = DataLoader(ins_data, batch_size=4,
                        shuffle=False, num_workers=0,
                        sampler=train_sampler)
    
    valid_loader = DataLoader(ins_data, batch_size=4,
                        shuffle=False, num_workers=0,
                        sampler=valid_sampler)
    
    dataloader = dict()
    dataloader["train"] = train_loader
    dataloader["val"] = valid_loader
    dataset_sizes = dict()
    dataset_sizes["train"] = len(train_indices)
    dataset_sizes["val"] = len(valid_indices)
    # inputs, classes = next(iter(train_loader))
    # print(type(inputs))
    # print(type(classes))
    # # Make a grid from batch
    # out = utils.make_grid(inputs)
    
    #to look at data
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    #imshow(out)
    
    return dataloader, dataset_sizes