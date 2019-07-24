import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import numpy as np
import os

IMG_SIZE = 224
DATA_DIR = "./Influencers/"
INFLUENCERS = os.listdir("./Influencers/")
if ".DS_Store" in INFLUENCERS:
    INFLUENCERS.remove('.DS_Store')

## Resize Images and use grey background
def process_img(pic):
    global IMG_SIZE
    width, height = pic.size
    print(width, height)
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
    if desired%2 != 0:
        dim = dim+1
    dimension_list[pos[1]] = dim
    resize_transform = transforms.Resize(resize_tuple)
    pad_transform = transforms.Pad(tuple(dimension_list), fill=(220,220,220), padding_mode='constant')
    transform = transforms.Compose([resize_transform, pad_transform])
    return transform(pic)

def get_processed_img():
    result = dict()
    for influencer in INFLUENCERS:
        # Transform all the images
        # result[influencer] = datasets.ImageFolder(DATA_DIR,
        #                                           transforms.Lambda(lambda img: process_img(img)))
        img_list = []
        for img_filename in os.listdir(DATA_DIR + influencer):
            if img_filename != ".DS_Store":
                img = Image.open(DATA_DIR + influencer + "/" + img_filename)
                img_list.append((process_img(img), img_filename))
        result[influencer] = img_list
    return result
    
    # Result is now a dict with format
    #{Influencer_1: [(pil_img_1, filename_1), (pil_img_2, filename_2)...], Influencer_2: ...}

# # Test display
# for pic in result: 
#     np_pic = np.array(pic[0])
#     print(np_pic.shape)


## Assign Dates and Likes to each picture in result
def translate_datetime(coded_date):
    date = int(coded_date)
    return date
    
def get_dates_likes():
    result = dict()
    processed_result = get_processed_img()
    for influencer in processed_result:
        img_list = []
        for img, filename in processed_result[influencer]:
            img_info = dict()
            try:
                filename_values = filename.split('-')
                #-4 to remove ".jpg"
                img_info["date"] = translate_datetime(filename_values[3][:-4])
                img_info["likes"] = int(filename_values[1])
                img_info["img"] = img
            except:
                print(filename)
            img_list.append(img_info)
        result[influencer] = img_list
    return result

## Assign a class (1-10) based on likes and dates



## Randomly Sample For Train and Validation (Maybe 80% vs 20%)

