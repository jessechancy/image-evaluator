import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image

IMG_SIZE = 224
DATADIR = "./Influencers/"

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

# Transform all the images
result = datasets.ImageFolder(DATA_DIR, transforms.Lambda(lambda img: process_img(img)))
# Test display
for pic in result:
    # print(pic)
    pic[0].show()
