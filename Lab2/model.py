import torch
import torch.nn as nn
import torch.nn.functional as F


class SnoutNet(nn.Module):
    """
    SnoutNet architecture for pet nose localization.
    
    Input: (batch_size, 3, 227, 227)
    Output: (batch_size, 2) - (u, v) coordinates of nose location
    """
    def __init__(self):
        super(SnoutNet, self).__init__()
        
        # Conv1 block: 3x3 conv, 3->64 channels, ReLU + MaxPool
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=4)  # 227x227 -> 57x57
        
        # Conv2 block: 3x3 conv, 64->128 channels, ReLU + MaxPool
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=4, padding=1)  # 57x57 -> 15x15
        
        # Conv3 block: 3x3 conv, 128->256 channels, ReLU + MaxPool
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=4)  # 15x15 -> 4x4
        
        # Fully connected layers
        self.fc1 = nn.Linear(4 * 4 * 256, 1024)  # 4096 -> 1024
        self.fc2 = nn.Linear(1024, 1024)  # 1024 -> 1024
        self.fc3 = nn.Linear(1024, 2)  # 1024 -> 2 (u, v coordinates)
        
        # Store input shape for model summary
        self.input_shape = (3, 227, 227)
        
    def forward(self, x):
        # Conv1 block
        x = F.relu(self.conv1(x))
        x = self.pool1(x)  # (batch, 64, 57, 57)
        
        # Conv2 block
        x = F.relu(self.conv2(x))
        x = self.pool2(x)  # (batch, 128, 15, 15)
        
        # Conv3 block
        x = F.relu(self.conv3(x))
        x = self.pool3(x)  # (batch, 256, 4, 4)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # (batch, 4096)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))  # (batch, 1024)
        x = F.relu(self.fc2(x))  # (batch, 1024)
        x = self.fc3(x)  # (batch, 2) - linear output for regression
        
        return x


def init_weights(m):
    """
    Initialize weights for the model.
    Xavier uniform initialization for Linear layers.
    """
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
