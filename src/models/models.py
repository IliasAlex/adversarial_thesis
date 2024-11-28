import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    '''
    CNN baseline model with three convolutional layers followed by max-pooling, and two fully connected layers.
    The model uses ReLU activations and dropout for regularization. The final layer outputs a prediction for each class.
    
    Parameters:
    - num_classes (int): The number of classes in the classification task. Defaults to 10 (for urbansound8k dataset).
    
    Architecture:
    - 3 Convolutional layers with increasing filter sizes (16, 32, 64)
    - Max Pooling after each convolutional layer to reduce spatial dimensions
    - 2 Fully connected layers: the first with 128 units and the second with 'num_classes' units
    - Dropout layer with a dropout rate of 0.5 after the first fully connected layer
    '''
    def __init__(self, num_classes=10):
        super(BaselineCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 16 * 10, 128) 
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Convolutional layers with ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))  # Output: [batch_size, 16, 64, 42]
        x = self.pool(F.relu(self.conv2(x)))  # Output: [batch_size, 32, 32, 21]
        x = self.pool(F.relu(self.conv3(x)))  # Output: [batch_size, 64, 16, 10]
        
        # Flatten the output from the convolutional layers
        x = x.view(-1, 64 * 16 * 10)  # Flatten to match the input size of fc1
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
