"""
multimodal transformer model. Heavily influenced by:
https://github.com/yaohungt/Multimodal-Transformer
"""

import torch
from torch import nn

'''
class CrossModalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossModalMultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)

        attended_values = torch.matmul(attention_weights, value)
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        output = self.out_proj(attended_values)
        output = self.layer_norm(
            output + attended_values)  # Adding the residual connection and applying layer normalization
        return output, attention_weights


class MultiLayerCrossModalAttention(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads):
        super(MultiLayerCrossModalAttention, self).__init__()
        self.layers = nn.ModuleList([CrossModalMultiHeadAttention(embed_dim, num_heads) for _ in range(num_layers)])

    def forward(self, query, key, value):
        attention_weights = []
        for layer in self.layers:
            query, attn_weights = layer(query, key, value)
            attention_weights.append(attn_weights)
        return query, attention_weights

'''





import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 中导入神经网络模块


class CrossModalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossModalMultiHeadAttention, self).__init__()  # 调用父类的构造函数
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"  # 确保嵌入维度可以被头数整除

        self.num_heads = num_heads  # 设置注意力头的数量
        self.embed_dim = embed_dim  # 设置嵌入维度
        self.head_dim = embed_dim // num_heads  # 计算每个头的维度

        self.query_proj = nn.Linear(embed_dim, embed_dim)  # 定义查询的线性投影层
        self.key_proj = nn.Linear(embed_dim, embed_dim)  # 定义键的线性投影层
        self.value_proj = nn.Linear(embed_dim, embed_dim)  # 定义值的线性投影层

        self.out_proj = nn.Linear(embed_dim, embed_dim)  # 定义输出的线性投影层

        self.layer_norm = nn.LayerNorm(embed_dim)  # 定义层归一化

    def forward(self, query, key, value):
        batch_size = query.size(0)  # 获取批次大小

        query = self.query_proj(query)  # 对查询进行线性投影
        key = self.key_proj(key)  # 对键进行线性投影
        value = self.value_proj(value)  # 对值进行线性投影

        # 重塑并转置查询以适应多头注意力机制
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        raw_scores = torch.matmul(query, key.transpose(-2, -1))  # 计算查询和键的点积
        scores = raw_scores / (self.head_dim ** 0.5)  # 缩放点积结果
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)  # 应用 softmax 获取注意力权重
        
        attended_values = torch.matmul(attention_weights, value)  # 将注意力权重应用于值
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)  # 重塑 attended_values

        output = self.out_proj(attended_values)  # 通过输出投影层
        output = self.layer_norm(
            output + attended_values)  # 添加残差连接并应用层归一化
        return output, scores  # 返回输出和注意力权重


class MultiLayerCrossModalAttention(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads):
        super(MultiLayerCrossModalAttention, self).__init__()  # 调用父类的构造函数
        self.layers = nn.ModuleList([CrossModalMultiHeadAttention(embed_dim, num_heads) for _ in range(num_layers)])  # 创建多个 CrossModalMultiHeadAttention 层的模块列表

    def forward(self, query, key, value):
        attention_weights = []  # 可选：用于存储每层的注意力权重
        for layer in self.layers:  # 遍历每一层注意力模块
            query, attn_weights = layer(query, key, value)  # 通过当前层并更新查询
            attention_weights.append(attn_weights)  # 可选：收集每层的注意力权重
        return query, attention_weights  # 返回最终的查询结果和注意力权重（注意：这里可能有拼写错误）


'''
#测试代码

import torch
import sys
sys.path.append('/WorkSpace/liudongbo/fusion')
from CrossModalMultiHeadAttention import MultiLayerCrossModalAttention,  CrossModalMultiHeadAttention

# 定义模型参数
num_layers = 2       # 注意力层数
embed_dim = 512      # 嵌入维度
num_heads = 8        # 注意力头数

# 初始化 MultiLayerCrossModalAttention 模块
cross_modal_attention = MultiLayerCrossModalAttention(num_layers, embed_dim, num_heads)

# 示例输入张量
batch_size = 16
text_seq_len = 20
image_seq_len = 30

# 随机生成的嵌入向量（实际应用中应为经过预处理的文本和图像特征）
text_embeddings = torch.randn(batch_size, text_seq_len, embed_dim)   # 文本模态
image_embeddings = torch.randn(batch_size, image_seq_len, embed_dim) # 图像模态

# 执行跨模态注意力：文本关注图像特征
output, attn_weights = cross_modal_attention(query=text_embeddings, key=image_embeddings, value=image_embeddings)

print("输出形状:", output.shape)                   # 预期输出形状：(batch_size, text_seq_len, embed_dim)
print("注意力权重形状:", attn_weights.shape)       # 预期注意力权重形状：(batch_size, num_heads, text_seq_len, image_seq_len)




import torch
import sys
sys.path.append('/WorkSpace/liudongbo/General_Model_Architecture/model')
from CrossModalMultiHeadAttention import MultiLayerCrossModalAttention,  CrossModalMultiHeadAttention

# 定义模型参数
num_layers = 5       # 注意力层数
embed_dim = 512      # 嵌入维度
num_heads = 8        # 注意力头数

# 初始化 MultiLayerCrossModalAttention 模块
cross_modal_attention = MultiLayerCrossModalAttention(num_layers, embed_dim, num_heads)

# 示例输入张量
batch_size = 16
text_seq_len = 20
image_seq_len = 30

# 随机生成的嵌入向量（实际应用中应为经过预处理的文本和图像特征）
text_embeddings = torch.randn(batch_size, text_seq_len, embed_dim)   # 文本模态
image_embeddings = torch.randn(batch_size, image_seq_len, embed_dim) # 图像模态

# 执行跨模态注意力：文本关注图像特征
output, attn_weights = cross_modal_attention(query=text_embeddings, key=image_embeddings, value=image_embeddings)

print("输出形状:", output.shape)                   # 预期输出形状：(batch_size, text_seq_len, embed_dim)
print("注意力权重形状:", attn_weights.shape)       # 预期注意力权重形状：(batch_size, num_heads, text_seq_len, image_seq_len)

'''