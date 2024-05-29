# Import PyTorch Modules 
import torch 
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pandas
from torchvision import transforms
import torch.nn as nn
torch.manual_seed(0)

# Import Non-PyTorch Modules 
import time
from imageio import imread
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
from figurePlot import perfromPlot
from customDataset import Dataset
import numpy as np

# Hyperparameters
num_epochs = 25
learning_rate = 0.002
train_batch_size = 15
val_batch_size =10

#Dataset Class and Object
train_csv_file = '../dataset/training_labels.csv'
validation_csv_file = '../dataset/validation_labels.csv'

# Absolute path for finding the directory contains image datasets
train_data_dir = '../dataset/training_data_pytorch/'
validation_data_dir = '../dataset/validation_data_pytorch/'


# Construct the composed object for transforming the image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
composed = transforms.Compose([transforms.Resize((224, 224))
                               , transforms.ToTensor()
                               , transforms.Normalize(mean, std)])

# Create the train dataset and validation dataset

train_dataset = Dataset(transform=composed
                        ,csv_file=train_csv_file
                        ,data_dir=train_data_dir)

validation_dataset = Dataset(transform=composed
                          ,csv_file=validation_csv_file
                          ,data_dir=validation_data_dir)

#Load the pre-trained model resnet18
model = models.resnet18(pretrained=True)

# Set the parameter cannot be trained for the pre-trained model
for param in model.parameters():
    param.requires_grad=False

# Re-defined the last layer
model.fc = nn.Linear(512,7)

# Print the model (PLEASE DO NOT MODIFY THIS BOX)
#print(model)

#TRAIN THE MODEL
# Create the loss function
criterion = nn.CrossEntropyLoss()

# Create the data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch_size)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=val_batch_size)

# Use the pre-defined optimizer Adam with learning rate 0.003
optimizer = torch.optim.Adam([parameters for parameters in model.parameters() if parameters.requires_grad], lr=learning_rate)


# Train the model
N_EPOCHS = num_epochs 
loss_list = []
accuracy_list = []
correct = 0
n_test = len(validation_dataset)

for epoch in range(N_EPOCHS):
    loss_sublist = []
    for x,y in train_loader:
        model.train()
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z,y)
        loss_sublist.append(loss.data.item())
        loss.backward()
        optimizer.step()
    loss_list.append(np.mean(loss_sublist))
    
    correct = 0
    for x_test, y_test in validation_loader:
        model.eval()
        z = model(x_test)
        _, yhat = torch.max(z.data, 1)
        correct += (yhat==y_test).sum().item()
        
    accuracy = correct/n_test
    accuracy_list.append(accuracy)
    
    print('Epoch_'+ str(epoch)+' : ' +'Accuracy '+ str(accuracy)) 
        
# Step 5: Plot the loss and Accuracy of the training dataset
perfromPlot(loss_list, accuracy_list)

# Save the model
torch.save(model, "../models/currency_classifier.pt")


