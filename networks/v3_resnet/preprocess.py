##

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import numpy as np
import os
import datetime
import pandas as pd
import random
import time
from queue import Queue, Empty
from threading import Thread

##

IMG_SIZE = 224
#os.chdir("/Volumes/My Passport")
DATA_DIR = "./Influencers/"
INFLUENCERS = os.listdir(DATA_DIR)
if ".DS_Store" in INFLUENCERS:
    INFLUENCERS.remove('.DS_Store')


##

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
    if int(desired+(2*dim)) != IMG_SIZE:
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
    count = 0
    for i in range(len(INFLUENCERS)):
        # Transform all the images
        # result[influencer] = datasets.ImageFolder(DATA_DIR,
        #                                           transforms.Lambda(lambda img: process_img(img)))
        print("influencer " + str(i) + " of " + str(len(INFLUENCERS)))
        influencer = INFLUENCERS[i]
        def get_processed_img_wrapper(influencer):
            nonlocal err_count, count
            img_list = []
            for img_filename in os.listdir(DATA_DIR + influencer):
                if img_filename != ".DS_Store":
                    try:
                        img = Image.open(DATA_DIR + influencer + "/" + img_filename).convert('RGB')
                        #processed = process_img(img)
                        processed = img
                        img_list.append((processed, img_filename))
                        count += 1
                        print("Done Count:", count)
                    except Exception as e:
                        print(e)
                        err_count += 1
                        print("Error Count: " + str(err_count))
                        continue
            print("Img List:", img_list)
            result[influencer] = img_list
        get_processed_img_wrapper(influencer)
    print("Error Count: " + str(err_count))
    return result

##

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
    
##
        
def save_image(result):
    folder_path = "Dataset/"
    random.shuffle(result)
    split = int(len(result)*0.2)
    val, train = result[:split], result[split:]
    def save_to_folder(list, folder_path):
        ctr = 0
        for img in list:
            try:
                img[2].save(os.path.join(folder_path, img[0], img[1], 
                            str(ctr) + "|" + str(img[3])+".jpg"))
            except Exception as error:
                print("Error with post", error)
            ctr = ctr + 1
    save_to_folder(train, folder_path+"train/")
    save_to_folder(val, folder_path+"val/")

def generate_folders(result):
    #add this when you have hard disk connected
    #os.chdir("/Volumes/My Passport")
    path = "Dataset"
    def generate_sub_folders(path):
        for influencer in result:
            for month in result[influencer]:
                avg_likes = result[influencer][month]["info"][0] / \
                            result[influencer][month]["info"][1]
                path_name = path + "/" + str(influencer) + "/" + month
                try:
                    os.makedirs(path_name)
                    with open(os.path.join(path_name, "avg_likes.txt"), "a+") as myfile:
                        myfile.write(str(avg_likes))
                except Exception as e:
                    print(e)
                    print(path_name + " already made!")
    generate_sub_folders(path+"/train")
    generate_sub_folders(path+"/val")
    
def reformat_result(result):
    final_result = []
    for influencer in result:
        for month in result[influencer]:
            for img_info in result[influencer][month]["img_list"]:
                img = img_info["img"] 
                likes = img_info["likes"]
                final_result.append([influencer, month, img, likes])
    return final_result
    
def generate_dataset():
    result = get_dates_likes()
    generate_folders(result)
    result = reformat_result(result)
    save_image(result)

generate_dataset()