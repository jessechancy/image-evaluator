import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import torch.optim as optim

from Instaset import InstaSet

# https://www.tensorflow.org/beta/tutorials/images/transfer_learning

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Transform Images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
trainset = InstaSet('../v3_resnet/Dataset', True, transform)
valset = InstaSet('../v3_resnet/Dataset', False, transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
            shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(valset, batch_size=100,
            shuffle=True, num_workers=1)
print(train_loader)
## Create Model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 1)
if device != "cpu":
    model = model.to(device)

## Define Loss Function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_contrastive = torch.mean((1 - batch_label_c) * torch.pow(euclidean_distance, 2) +
                                      batch_label_c * torch.pow(torch.clamp(2 - euclidean_distance, min=0.0), 2))

## Train the network
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        if device == "cpu":
            inputs, labels = data
        else:
            inputs, labels = data[0].to(device), data[1].to(device)
        print(inputs)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

## Test networkß
print('fin')
