import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import numpy as np



## Import Images

def import_images(root, train, transforms):
    samples = []
    if train:
        root = os.path.join(root, "train")
    else:
        root = os.path.join(root, "val")
    for score_class in os.listdir(root):
        score_class_folder = os.path.join(root, score_class)
        for image_file in os.listdir(score_class_folder):
            image_filepath = os.path.join(score_class_folder, image_file)
            img = Image.open(image_filepath).convert('RGB')
            #any processing to image can happen here
            img = transforms(img)
            samples.append((img, score_class))
    return samples
    
## Dataset Class

class InstaSet(Dataset):
    def __init__(self, root, train=True, transforms=None):
        self.root = root
        self.train = train
        self.transforms = transforms
        self.samples = import_images(root, train, transforms)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img, label =  self.samples[idx]
        if self.transforms is not None:
            img = self.transforms(img)
        label = torch.from_numpy(int(label))
        return (img, label)

