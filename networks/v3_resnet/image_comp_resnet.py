'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision import models, datasets

import os
import argparse
import numpy as np
import subprocess

from Instaset import InstaSet
from models.resnet import ResNet18

## Parse Arguments

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pretrained", help="choose pretrained", action="store_true", default=False)
parser.add_argument("-l", "--learning", help="change learning rate", default=0.01)
parser.add_argument("-f", "--filepath", type=str, default="./Dataset/", help="data filepath")
parser.add_argument("-g", "--gpu", type=int, default=0, help="gpu")
args = parser.parse_args()

## File Directories

DATASET_DIR = args.filepath

## Hyper Parameters

BATCH_SIZE = 1
learning_rate = args.learning
gpu = args.gpu
pretrain_model = args.pretrained

print("Learning Rate: ", learning_rate)
print("Pretrain: ", pretrain_model)
print("Dataset Directory: ", DATASET_DIR)
print("GPU: ", gpu)

## GPU Setting

device = torch.device('cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax([int(x.split()[2]) for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))

##

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
## Loads InstaSet
train_set = InstaSet(DATASET_DIR, True, transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, 
                                           shuffle=True, num_workers=0)

val_set = InstaSet(DATASET_DIR, False, transform_val)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, 
                                         shuffle=True, num_workers=0)

#create a dataloader that gets one tensor based on index and the other randomly 
#based on chosen month and user

#our data should be arranged in user/month/img[title=likes]

##

#train_loader = [(tensor1, label1, tensor2, label2),(),(),(),()]

## Loads CIFAR10
# train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
# 
# val_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
# val_loader = torch.utils.data.DataLoader(val_set, batch_size=100, shuffle=False, num_workers=2)
##

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
net = models.resnet18()
net.fc = nn.Linear(512, 1)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

class pairwiseloss():
    pass
    #output1/like count 1 - outpu2/like count2 + 1

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (input1, target1, input2, target2) in enumerate(train_loader):
        input1, target1, input2, target2 = input1.to(device), target1.to(device), input2.to(device), target2.to(device)
        #inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        print(input1.size(), input2.size())
        output1 = net(input1)
        output2 = net(input2)
        print(output1.size(), output2.size())
        #(like count1, like count2]
        ## Have to write the criterion function
        loss = criterion(output, target)
        print(loss, loss.item())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(0, 100):
    train(epoch)
    test(epoch)