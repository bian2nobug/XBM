import torch
import torch.nn as nn

class regression_head(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(regression_head, self).__init__()
        
        self.flatten = nn.Flatten()  # Flatten层将输入拉成一维
        self.relu = nn.ReLU()  # 激活函数ReLU
        self.fc1 = nn.Linear(input_dim, 128)  # 第一个全连接层
        self.dropout = nn.Dropout(p=0.2)  # Dropout层，p为丢弃概率
        self.fc2 = nn.Linear(128, output_dim)  # 第二个全连接层
        

    
    def forward(self, x):
        x = self.flatten(x)  # 展平输入
        
        x = self.relu(x)  # ReLU激活

        x = self.fc1(x)  # 第一个全连接层

        x = self.dropout(x)  # Dropout

        x = self.fc2(x)  # 第二个全连接层
        
        x = torch.squeeze(x, -1)
        
        return x
    