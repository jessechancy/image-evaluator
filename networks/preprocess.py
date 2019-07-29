import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import numpy as np
import os
import datetime
import pandas as pd
import random

IMG_SIZE = 224
#os.chdir("/Volumes/My Passport")
DATA_DIR = "/home/angelica/external4/top-100/"
INFLUENCERS = os.listdir(DATA_DIR)
if ".DS_Store" in INFLUENCERS:
    INFLUENCERS.remove('.DS_Store')

## Resize Images and use grey background
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
    if int(desired)+2(dim) != IMG_SIZE:
        dim = dim+1
    dimension_list[pos[1]] = dim
    resize_transform = transforms.Resize(resize_tuple)
    pad_transform = transforms.Pad(tuple(dimension_list), fill=(220,220,220), padding_mode='constant')
    to_tensor = transforms.ToTensor()
    transform = transforms.Compose([resize_transform, pad_transform, to_tensor])
    return transform(pic)
    
def get_processed_img():
    result = dict()
    err_count = 0
    for i in range(len(INFLUENCERS)):
        # Transform all the images
        # result[influencer] = datasets.ImageFolder(DATA_DIR,
        #                                           transforms.Lambda(lambda img: process_img(img)))
        print("influencer " + str(i) + " of " + str(len(INFLUENCERS)))
        influencer = INFLUENCERS[i]
        img_list = []
        for img_filename in os.listdir(DATA_DIR + influencer):
            if img_filename != ".DS_Store":
                try:
                    img = Image.open(DATA_DIR + influencer + "/" + img_filename).convert('RGB')
                    #processed = process_img(img)
                    processed = img
                    img_list.append((processed, img_filename))
                except:
                    err_count += 1
                    print("Error Count: " + str(err_count))
                    continue
        result[influencer] = img_list
    print("Error Count: " + str(err_count))
    return result
    
    # Result is now a dict with format
    #{Influencer_1: [(pil_img_1, filename_1), (pil_img_2, filename_2)...], Influencer_2: ...}

# # Test display
# for pic in result: 
#     np_pic = np.array(pic[0])
#     print(np_pic.shape)


## Assign Dates and Likes to each picture in result

def translate_datetime(time_stamp):
    date = datetime.datetime.fromtimestamp(int(time_stamp)).strftime('%Y-%m')
    return date
    
def get_dates_likes():
    result = dict()
    processed_result = get_processed_img()
    for influencer in processed_result:
        sorted_by_date = dict()
        for img, filename in processed_result[influencer]:
            try:
                filename_values = filename.split('-')
                #-4 to remove ".jpg"
                date = translate_datetime(filename_values[3][:-4])
                if date not in sorted_by_date:
                    sorted_by_date[date] = dict()
                    sorted_by_date[date]["img_list"] = []
                    sorted_by_date[date]["info"] = [0, 0] #(total, count)
                img_info = dict()
                img_info["likes"] = int(filename_values[1])
                img_info["img"] = img
                sorted_by_date[date]["info"][0] += img_info["likes"]
                sorted_by_date[date]["info"][1] += 1
                sorted_by_date[date]["img_list"].append(img_info)
            except:
                print(filename)
        result[influencer] = sorted_by_date
    return result

#returns the average number of images in each month
def check_data_quality(result):
    count = 0
    total = 0
    for influencer in result:
        for date in result[influencer]:
            count += 1
            total += len(result[influencer][date])
    avg_imgs_in_period = total/count
    print(avg_imgs_in_period, "images / time period")
    return avg_imgs_in_period
    
## Assign a class (1-10) based on likes and dates

def assign_class():
    classes = ["1","2","3","4","5","6","7","8","9","10"]
    ratio_boundaries = [(0.0,0.4),(0.4,0.65),(0.65,0.8),(0.8,0.9),(0.9,1.0),
                        (1.0,1.1),(1.1,1.2),(1.2,1.35),(1.35,1.6),(1.6,2.0)]
    result = {score_class:[] for score_class in classes}
    processed_results = get_dates_likes()
    for influencer in processed_results:
        for date in processed_results[influencer]:
            total, count = processed_results[influencer][date]["info"]
            avg = total/count
            for img_info in processed_results[influencer][date]["img_list"]:
                likes = img_info["likes"]
                ratio = likes / avg
                img = img_info["img"]
                for i in range(len(ratio_boundaries)):
                    score_class = str(i+1)
                    low, high = ratio_boundaries[i]
                    if i == len(ratio_boundaries) - 1:
                        if low <= ratio:
                            result[score_class].append(img)
                    elif low <= ratio < high:
                        result[score_class].append(img)
    return result
    
def import_images():
    result = []
    for key in result:
        print(key, len(result[key]))
    processed_result = assign_class()
    for score_class in processed_result:
        for post in processed_result[score_class]:
            result.append([post, score_class])
    random.shuffle(result)
    return result
