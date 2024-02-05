import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


    

''' Models a simple Convolutional Neural Network'''
class CustomModel(nn.Module):
    def __init__(self, in_channels, num_classes, image_size):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # Calculate the size of the feature maps after convolutions and pooling
        def conv_output_size(size, kernel_size=5, stride=1, padding=0):
            return (size - kernel_size + 2 * padding) // stride + 1

        conv1_size = conv_output_size(image_size, 5)
        pooled_size = conv_output_size(conv1_size, 2, 2)
        conv2_size = conv_output_size(pooled_size, 5)
        pooled_size = conv_output_size(conv2_size, 2, 2)

        linear_input_size = 16 * pooled_size * pooled_size
        self.fc1 = nn.Linear(linear_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

