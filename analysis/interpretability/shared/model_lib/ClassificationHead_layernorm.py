import torch.nn as nn


class ClassificationHead_layernorm(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ClassificationHead_layernorm, self).__init__()
        self.head = nn.Sequential(
            nn.Flatten(),                           # flatten input
            nn.LayerNorm(input_dim),              # normalization
            nn.Dropout(0.2),                        # dropout rate 20%
            nn.Linear(input_dim, 128),              # fully connected layer to 128 dims
            nn.ReLU(),                              # activation
            nn.LayerNorm(128),                    # normalization
            nn.Dropout(0.2),                        # dropout rate 20%
            nn.Linear(128, output_dim),                  # output to output_dim
        )
    
    def forward(self, x):
        return self.head(x)
