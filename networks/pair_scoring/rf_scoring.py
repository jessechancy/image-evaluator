import torch
import torch.nn as nn

from InstaSet import InstaSet
import torchvision.models as models

# https://www.tensorflow.org/beta/tutorials/images/transfer_learning

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
trainset = InstaSet('./Dataset', True, transform)
valset = InstaSet('./Dataset', False, transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
            shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(valset, batch_size=100,
            shuffle=True, num_workers=1)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 1)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
for epoch in range(2):
    running_loss = 0.0
    prev = 0
    outputs = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        print(inputs)
        optimizer.zero_grad()
        prev = outputs
        outputs = model(inputs)
        if not (i%2):
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('fin')
