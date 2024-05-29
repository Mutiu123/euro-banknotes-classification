
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
import sys
sys.path.insert(1, 'C:/Users/adegb/Desktop/Computer Vision Projects/Recognition and Clasification of Currency note using Transfer Learning/Bank-notes-classification-with-transfer-learning-using-Pytorch/src')
from customDataset import Dataset
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np

# directory contains CSV files and image dataset folder

test_csv_file = '../dataset/test_labels.csv'
test_data_dir = '../dataset/test_data_pytorch/'

# Create Dataset Class

    
# Construct the composed object for transforming the image 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
trans_step = [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)]

composed = transforms.Compose(trans_step)
# Type your code here

# Create a test_dataset

test_dataset = Dataset(transform=composed
                       , csv_file=test_csv_file
                       , data_dir=test_data_dir)

# Load pre-trained model
model = torch.load("../models/currency_classifier.pt")

# Print model structure

#print("ResNet18:\n", model)

# Set Data Loader object
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10)

# Predict the data using ResNet18 model and print out accuracy

correct = 0
accuracy = 0
N = len(test_dataset)
for x_test, y_test in test_loader:
    model.eval()
    z = model(x_test)
    _, yhat = torch.max(z.data, 1)
    correct += (yhat == y_test).sum().item()
accuracy = correct / N
print("Accuracy using ResNet18 model is: ", accuracy) 

