import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import numpy as np

import random

##
root = "./Dataset"

## Import Images

def import_images(root, train):
    samples = []
    if train:
        root = os.path.join(root, "train")
    else:
        root = os.path.join(root, "val")
    for score_class in os.listdir(root):
        if score_class == ".DS_Store":
            continue
        score_class_folder = os.path.join(root, score_class)
        for image_file in os.listdir(score_class_folder):
            if image_file == ".DS_Store":
                continue
            image_filepath = os.path.join(score_class_folder, image_file)
            img = Image.open(image_filepath).convert('RGB')
            samples.append((img, score_class))
    return samples
    
## Dataset Class

class InstaSet(Dataset):
    def __init__(self, root, train=True, transforms=None):
        self.root = root
        self.train = train
        self.transforms = transforms
        self.samples = import_images(root, train)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img, label =  self.samples[idx]
        
        if self.transforms is not None:
            img = self.transforms(img)
        label = torch.from_numpy(np.array(int(label)))
        return (img, label)

## Pick Random Images

picks = 150000

def import_images(root, train):
    samples = []
    for i in range(picks):
        samples.append(pick_images(root))
    

def pick_images(root, train):
    if train == True:
        root = root + "/train"
    else:
        root = root + "/val"
    users = os.listdir(root)
    users_len = len(users)
    flagged = False
    single = set()
    while not flagged:
        user_ind = random.randint(0, users_len-1)
        user = users[user_ind]
        user_path = os.path.join(root, user)
        user_data = os.listdir(user_path)
        month_ind = random.randint(0, len(user_data)-1)
        month = user_data[month_ind]
        month_path = os.path.join(user_path, month)
        if month_path in single:
            continue
        month_data = os.listdir(month_path)
        month_data.remove("avg_likes.txt")
        if len(month_data) <= 1:
            single.add(month_path)
        try:
            img_ind_1, img_ind_2 = random.sample(range(len(month_data)), 2)
            flagged = True
        except:
            flagged = False
    label1, label2 = month_data[img_ind_1], month_data[img_ind_2]
    
    img_path_1 = os.path.join(month_path, label1)
    img_path_2 = os.path.join(month_path, label2)
    
    img1 = Image.open(img_path_1).convert('RGB')
    img2 = Image.open(img_path_2).convert('RGB')
    # img1 = transforms(img1)
    # img2 = transforms(img2)
    label1 = label1.split("|")[1][:-4]
    label2 = label2.split("|")[1][:-4]
    label1 = torch.from_numpy(np.array(int(label1)))
    label2 = torch.from_numpy(np.array(int(label2)))
    return (img1, label1, img2, label2)
    
def test():
    count = 0
    for i in range(10000):
        try:
            pick_images(root, True)
        except:
            print("broken")
            count += 1
    print("Broken Count: ", count)
    return None

## Dataset Class

class InstaSet(Dataset):
    def __init__(self, root, train=True, transforms=None):
        self.root = root
        self.train = train
        self.transforms = transforms
        self.samples = import_images(root, train)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img, label =  self.samples[idx]
        
        if self.transforms is not None:
            img = self.transforms(img)
        label = torch.from_numpy(np.array(int(label)))
        return (img, label)
