import torch
import torch.nn as nn

class regression_head(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(regression_head, self).__init__()
        
        self.flatten = nn.Flatten()  # Flatten layer to reshape input to 1D
        self.relu = nn.ReLU()  # ReLU activation
        self.fc1 = nn.Linear(input_dim, 128)  # first fully connected layer
        self.dropout = nn.Dropout(p=0.2)  # Dropout layer, p is the drop probability
        self.fc2 = nn.Linear(128, output_dim)  # second fully connected layer
        

    
    def forward(self, x):
        x = self.flatten(x)  # flatten input
        
        x = self.relu(x)  # ReLU activation

        x = self.fc1(x)  # first fully connected layer

        x = self.dropout(x)  # Dropout

        x = self.fc2(x)  # second fully connected layer
        
        x = torch.squeeze(x, -1)
        
        return x
