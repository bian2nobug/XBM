import torch.nn as nn


class ClassificationHead_layernorm(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ClassificationHead_layernorm, self).__init__()
        self.head = nn.Sequential(
            nn.Flatten(),                           # 展平输入
            nn.LayerNorm(input_dim),              # 批标准化
            nn.Dropout(0.2),                        # 丢弃率为 20%
            nn.Linear(input_dim, 128),              # 输入到 256 维的全连接层
            nn.ReLU(),                              # 激活函数
            nn.LayerNorm(128),                    # 批标准化
            nn.Dropout(0.2),                        # 丢弃率为 20%
            nn.Linear(128, output_dim),                  # 输出到 n_out 维
        )
        
    def forward(self, x):
        return self.head(x)
