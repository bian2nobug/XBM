
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleChannelAttention(nn.Module):
    """Multi-scale channel attention module (MS-CAM)"""
    def __init__(self, channels=1536, ratio=16):
        super().__init__()
        mid_channels = channels // ratio
        
        # global feature branch
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        # local feature branch
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W) = (1536, M, 21, 1)
        g_att = self.global_att(x)  # global attention weights
        l_att = self.local_att(x)   # local attention weights
        
        # combine global and local attention
        att_weights = self.sigmoid(g_att * l_att)
        return x * att_weights

class AttentionFeatureFusion(nn.Module):
    """Attention-based feature fusion module"""
    def __init__(self, feature_dims=21, channels=1536):
        super().__init__()
        self.ms_attention = MultiScaleChannelAttention(channels)
        
        # fully connected layer after feature fusion
        self.fusion_fc = nn.Sequential(
            nn.Linear(feature_dims, feature_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dims // 2, 1),
            nn.Sigmoid()  # output a fusion weight in [0,1]
        )

    def forward(self, x):
        # x: (1536, M, 21)
        batch_size, M, features = x.shape
        
        # reshape to the format required by convolution
        x_reshaped = x.unsqueeze(-1)  # (1536, M, 21, 1)
        
        # apply multi-scale channel attention
        attended_features = self.ms_attention(x_reshaped)  # (1536, M, 21, 1)
        
        # fuse features
        fused_features = self.fusion_fc(attended_features.squeeze(-1))  # (1536, M, 1)
        
        return fused_features