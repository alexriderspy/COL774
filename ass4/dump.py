import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import pandas as pd
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import tqdm

batch_size = 1000

class ImageLoader(Dataset):
    def __init__(self, dataframe_x, dataframe_y, root_dir, transform = None):
        self.root_dir = root_dir
        self.dataframe_x = dataframe_x
        self.dataframe_y = dataframe_y
        self.transform = transform

    def __len__(self):
        return len(self.dataframe_x)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.dataframe_x[idx,1])
        image = np.array(cv2.imread(img_name))
        #print("Image" + str(image))
        image = Image.fromarray(image)
        labelKey = self.dataframe_y[idx, 1]
        label = torch.tensor(int(labelKey))
        #print(image.shape)
        if self.transform:
            
            image = self.transform(image)
            image = (image - torch.mean(image))/ torch.std(image)
        #print(image.shape)

        return image, label

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

size = 64
transform = transforms.Compose(
    [transforms.Resize((size,size)),
    transforms.ToTensor(),
    #transforms.Normalize()
    ]

)

directory = '/kaggle/input/col774-2022/'
dataframe_x = pd.read_csv(os.path.join(directory,'train_x.csv'))
dataframe_y = pd.read_csv(os.path.join(directory, 'train_y.csv'))
dataframe_val_x = pd.read_csv(os.path.join(directory,'non_comp_test_x.csv'))
dataframe_val_y = pd.read_csv(os.path.join(directory, 'non_comp_test_y.csv'))

dataset = ImageLoader(dataframe_x = np.vstack((dataframe_x,dataframe_val_x)), dataframe_y = np.vstack((dataframe_y,dataframe_val_y)), root_dir = os.path.join(directory, 'images/images/'), transform = transform)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

datasets = {}

datasets['train'] = train_dataset
datasets['val'] = val_dataset

dataloaders = {}

dataloaders['train'] = DataLoader(dataset = datasets['train'], batch_size = batch_size, shuffle=True, num_workers = 0)
dataloaders['val'] = DataLoader(dataset = datasets['val'], batch_size = batch_size, shuffle=True, num_workers = 0)

dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

model_conv = torchvision.models.resnet18(pretrained=True)

for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 30)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.AdamW(model_conv.fc.parameters(), lr=5e-5)

num_epochs = 5

best_acc = 0.0

model = model_conv
optimizer = optimizer_conv
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device, dtype = torch.float)
            labels = labels.to(device)

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                #print("Hello")
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

#         if phase == 'train':
#             scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

    print()
