from PIL import Image
import os
import face_recognition
import torch
import subprocess
import numpy as np
import torchvision.transforms as transforms

## GPU Setting

device = torch.device('cuda:'+str(0) if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax([int(x.split()[2]) for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))

## Processing
IMG_SIZE = 720
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
        desired = width*IMG_SIZE//height
        dim = int((IMG_SIZE-desired)/2)
        pos = [0,2]
        resize_tuple = (IMG_SIZE, int(desired))
    dimension_list[pos[0]] = dim
    if int(desired+(2*dim)) != IMG_SIZE:
        dim = dim+1
    dimension_list[pos[1]] = dim
    resize_transform = transforms.Resize(resize_tuple)
    pad_transform = transforms.Pad(tuple(dimension_list), fill=(220,220,220), padding_mode='constant')
    crop = transforms.RandomCrop(720, padding=4)
    #to_tensor = transforms.ToTensor()
    transform = transforms.Compose([resize_transform, pad_transform, crop])
    return transform(pic)

##

root = "../../.."
new_dataset = "./FilteredDataset/"
os.chdir(root)
# Load the jpg file into a numpy array
train = "Dataset/train/"
count = 0
saved_count = 0
batch_size = 128
tmp_imgs = []
tmp_imgs_path = []

for influencer in os.listdir(train):
    inf_path = os.path.join(train, influencer)
    path_i_new = os.path.join(new_dataset, train, influencer)
    for month in os.listdir(inf_path):
        mon_path = os.path.join(inf_path, month)
        path_m_new = os.path.join(path_i_new, month)
        os.makedirs(path_m_new, exist_ok=True)
        for img in os.listdir(mon_path):
            if img == "avg_likes.txt":
                continue
            else:
                count += 1
                print("train: ", count)
                path_img_new = os.path.join(path_m_new, img)
                img_path = os.path.join(mon_path, img)
                image = process_img(Image.open(img_path).convert('RGB'))
                image = np.array(image)
                if len(tmp_imgs) < batch_size - 1:
                    tmp_imgs.append(image)
                    tmp_imgs_path.append(path_img_new)
                elif len(tmp_imgs) == batch_size - 1:
                    for i in tmp_imgs:
                        print(i.shape)
                    tmp_imgs.append(image)
                    tmp_imgs_path.append(path_img_new)
                    face_locations_list = face_recognition.api.batch_face_locations(tmp_imgs,
                                     number_of_times_to_upsample=1, batch_size=batch_size)
                    for i in range(len(face_locations_list)):
                        face_locations = face_locations_list[i]
                        filtered_img = tmp_imgs[i]
                        f_img_path = tmp_imgs_path[i]
                        if len(face_locations) >= 1:
                            
                            saved_count += 1
                            filtered_img = Image.fromarray(filtered_img)
                            filtered_img.save(f_img_path)
                    tmp_imgs = []
                
print("Found ", str(saved_count), " in ", str(count))
 
val = "Dataset/val/"
count = 0
saved_count = 0
tmp_imgs = []
tmp_imgs_path = []

for influencer in os.listdir(val):
    inf_path = os.path.join(val, influencer)
    path_i_new = os.path.join(new_dataset, val, influencer)
    for month in os.listdir(inf_path):
        mon_path = os.path.join(inf_path, month)
        path_m_new = os.path.join(path_i_new, month)
        os.makedirs(path_m_new, exist_ok=True)
        for img in os.listdir(mon_path):
            if img == "avg_likes.txt":
                continue
            else:
                count += 1
                print("val: ", count)
                path_img_new = os.path.join(path_m_new, img)
                img_path = os.path.join(mon_path, img)
                image = process_img(Image.open(img_path).convert('RGB'))
                image = np.array(image)
                if len(tmp_imgs) < batch_size - 1:
                    tmp_imgs.append(image)
                    tmp_imgs_path.append(path_img_new)
                elif len(tmp_imgs) == batch_size - 1:
                    tmp_imgs.append(image)
                    tmp_imgs_path.append(path_img_new)
                    face_locations_list = face_recognition.api.batch_face_locations(tmp_imgs,
                                     number_of_times_to_upsample=1, batch_size=batch_size)
                    for i in range(len(face_locations_list)):
                        face_locations = face_locations_list[i]
                        filtered_img = tmp_imgs[i]
                        f_img_path = tmp_imgs_path[i]
                        if len(face_locations) >= 1:
                            
                            saved_count += 1
                            filtered_img = Image.fromarray(filtered_img)
                            filtered_img.save(f_img_path)
                    tmp_imgs = []
                
print("Found ", str(saved_count), " in ", str(count))
