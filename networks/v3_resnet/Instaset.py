import os
import random
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

import random

##
root = "../../../Dataset"

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

## Processing
IMG_SIZE = 224
def process_img(pic):
    global IMG_SIZE
    width, height = pic.size
    dimension_list = [0,0,0,0]
    desired = 0
    dim = 0
    pos = [0,0]
    resize_tuple = (IMG_SIZE,IMG_SIZE)
    if width > height:
        desired = height*IMG_SIZE/width
        dim = int((IMG_SIZE-desired)/2)
        pos = [1,3]
        resize_tuple = (int(desired), IMG_SIZE)
    elif height > width:
        desired = width*IMG_SIZE/height
        dim = int((IMG_SIZE-desired)/2)
        pos = [0,2]
        resize_tuple = (IMG_SIZE, int(desired))
    dimension_list[pos[0]] = dim
    if int(desired+(2*dim)) != IMG_SIZE:
        dim = dim+1
    dimension_list[pos[1]] = dim
    resize_transform = transforms.Resize(resize_tuple)
    pad_transform = transforms.Pad(tuple(dimension_list), fill=(220,220,220), padding_mode='constant')
    #to_tensor = transforms.ToTensor()
    transform = transforms.Compose([resize_transform, pad_transform])
    return transform(pic)

## Pick Random Images

picks = 1000

def pick_images(root, train):
    if train == True:
        root = root + "/train"
    else:
        root = root + "/val"
    users = os.listdir(root)
    flagged = False
    single = set()
    while not flagged:
        user = random.choice(users)
        user_path = os.path.join(root, user)
        user_data = os.listdir(user_path)
        if len(user_data) < 2:
            continue
        else:
            month = random.choice(user_data)
            year = month[:-3]
            year_data = [k for k in user_data if year in k]
            month2 = random.choice(year_data)
            month_path = os.path.join(user_path, month)
            month2_path = os.path.join(user_path, month2)
            month_data = os.listdir(month_path)
            month_data.remove("avg_likes.txt")
            month2_data = os.listdir(month2_path)
            month2_data.remove("avg_likes.txt")
            label1 = random.choice(month_data)
            label2 = random.choice(month2_data)
            flagged = True
        
        # try:
        #     img_ind_1, img_ind_2 = random.sample(range(len(month_data)), 2)
        #     flagged = True
        # except:
        #     flagged = False
    
    img_path_1 = os.path.join(month_path, label1)
    img_path_2 = os.path.join(month2_path, label2)
    print(img_path_1, img_path_2)
    img1 = process_img(Image.open(img_path_1).convert('RGB'))
    img2 = process_img(Image.open(img_path_2).convert('RGB'))
    # img1 = transforms(img1)
    # img2 = transforms(img2)
    label1 = label1.split("|")[1][:-4]
    label2 = label2.split("|")[1][:-4]
    
    return (img1, label1, img2, label2)
    
def test():
    count = 0
    for i in range(1000):
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
        
    def __len__(self):
        if self.train:
            return picks
        else:
            return int(picks * 0.25)
    def __getitem__(self, idx):
        img1, label1, img2, label2 =  pick_images(self.root, self.train)
        
        if self.transforms is not None:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)
            
        label1 = torch.from_numpy(np.array(int(label1)))
        label2 = torch.from_numpy(np.array(int(label2)))
        return (img1, label1, img2, label2)
