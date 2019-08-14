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

import lera

## Parse Arguments

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pretrained", help="choose pretrained", action="store_true", default=False)
parser.add_argument("-l", "--learning", help="change learning rate", default=0.0001)
#parser.add_argument("-l", "--learning", help="change learning rate", default=0.001) # or 0.0001
parser.add_argument("-f", "--filepath", type=str, default="../../../Dataset", help="data filepath")
parser.add_argument("-g", "--gpu", type=int, default=0, help="gpu")
args = parser.parse_args()

## File Directories

DATASET_DIR = args.filepath
epoch_count = 1000
## Hyper Parameters

BATCH_SIZE = 1
learning_rate = float(args.learning)
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

lera.log_hyperparams({
  'title': 'Image Evaluator',
  'batch_size': BATCH_SIZE,
  'epochs': epoch_count,
  'optimizer': 'Adam',
  'lr': learning_rate,
  })

##

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(224, padding=4),
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
net = models.resnet18(pretrained=pretrain_model)
net.fc = nn.Linear(512, 1)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# try Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
def pairwiseloss(output1, output2, label1, label2):
    #euclid_dist = F.pairwise_distance(output1/label1,output2/label2)
    euclid_dist = F.pairwise_distance(output1-output2,torch.log(label1)-torch.log(label2))
    euclid_dist_pow = torch.pow(euclid_dist, 2)
    return torch.mean(euclid_dist_pow)

class PairwiseLoss(torch.nn.Module):
    
    def __init__(self):
        super(PairwiseLoss,self).__init__()
        self.flag = 0
        
    def forward(self,output1,output2,label1,label2):
        
        if self.flag <= 5:
            print(output1, output2, label1, label2)
            euclid_dist = F.pairwise_distance(output1/output2,label1/label2) #change to log later
            print(output1/output2)
            print(label1/label2)
            print(euclid_dist)
            euclid_dist_pow = torch.pow(euclid_dist, 2)
            print(euclid_dist_pow)
            avg = torch.mean(euclid_dist_pow)
            print(avg)
            subtracted = avg - 0.2
            print(subtracted)
            subtracted[subtracted < 0] = 0
            final = subtracted
            print(final)
        else:
            euclid_dist = F.pairwise_distance(output1/output2,label1/label2) #change to log later
            euclid_dist_pow = torch.pow(euclid_dist, 2)
            avg = torch.mean(euclid_dist_pow)
            subtracted = avg - 0.2
            subtracted[subtracted < 0] = 0
            final = subtracted
        self.flag += 1
        return final

criterion = PairwiseLoss()

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    print("Train")
    net.train()
    train_loss = 0
    correct_count = 0
    total = 0
    for batch_idx, (input1, target1, input2, target2) in enumerate(train_loader):
        input1, target1, input2, target2 = input1.to(device), target1.to(device), input2.to(device), target2.to(device)
        #inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output1 = net(input1).float()
        output2 = net(input2).float()
        target1 = target1.float()
        target2 = target2.float()

        ##
        if output1 > output2 and target1 > target2:
            if epoch == 20:
                print(output1, output2, target1, target2)
            correct = True
        elif output2 > output1 and target2 > target1:
            if epoch == 20:
                print(output1, output2, target1, target2)
            correct = True
        elif output1 == output2 and target1 == target2:
            if epoch == 20:
                print(output1, output2, target1, target2)
            correct = True
        else:
            correct = False
        #(like count1, like count2]
        ## Have to write the criterion function
        loss = criterion(output1, output2, target1, target2)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # _, predicted = outputs.max(1)
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()
        # loss.data[0]

        total += 1
        correct_count += 1 if correct else 0

        # print(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print("Total Loss: %.3f | Acc: %.3f" % (train_loss/(batch_idx+1), 100. * correct_count/total))
    lera.log('train_loss', train_loss/(batch_idx+1))
    lera.log('train_acc', 100. * correct_count/total)


def test(epoch):
    print("Validation")
    global best_acc
    net.eval()
    test_loss = 0
    correct_count = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (input1, target1, input2, target2) in enumerate(val_loader):
            input1, target1, input2, target2 = input1.to(device), target1.to(device), input2.to(device), target2.to(device)
            output1 = net(input1).float()
            output2 = net(input2).float()
            target1 = target1.float()
            target2 = target2.float()
            loss = criterion(output1, output2, target1, target2)

            test_loss += loss.item()
            # _, predicted = outputs.max(1)
            # total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()
            
            # print(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            if output1 > output2 and target1 > target2:
                correct = True
            elif output2 > output1 and target2 > target1:
                correct = True
            elif output1 == output2 and target1 == target2:
                correct = True
            else:
                correct = False
                
            total += 1
            correct_count += 1 if correct else 0
            
        print("Total Loss: %.3f | Acc: %.3f" % (test_loss/(batch_idx+1), 100. * correct_count/total))
        lera.log('val_loss', test_loss/(batch_idx+1))
        lera.log('val_acc', 100. * correct_count/total)
            
    # Save checkpoint.
    acc = 100.*correct_count/total
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


for epoch in range(0, epoch_count):
    train(epoch)
    test(epoch)
print("Best Accuracy:", best_acc)