"""
multimodal transformer model. Heavily influenced by:
https://github.com/yaohungt/Multimodal-Transformer
"""

import os
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





import torch
from torch import nn


class CrossModalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossModalMultiHeadAttention, self).__init__()  # call the parent constructor
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"  # ensure embed_dim is divisible by num_heads

        self.num_heads = num_heads  # number of attention heads
        self.embed_dim = embed_dim  # embedding dimension
        self.head_dim = embed_dim // num_heads  # dimension per head

        self.query_proj = nn.Linear(embed_dim, embed_dim)  # linear projection for query
        self.key_proj = nn.Linear(embed_dim, embed_dim)  # linear projection for key
        self.value_proj = nn.Linear(embed_dim, embed_dim)  # linear projection for value

        self.out_proj = nn.Linear(embed_dim, embed_dim)  # linear projection for output

        self.layer_norm = nn.LayerNorm(embed_dim)  # layer normalization

    def forward(self, query, key, value):
        batch_size = query.size(0)  # get the batch size

        query = self.query_proj(query)  # linear projection of query
        key = self.key_proj(key)  # linear projection of key
        value = self.value_proj(value)  # linear projection of value

        # reshape and transpose query for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        raw_scores = torch.matmul(query, key.transpose(-2, -1))  # dot product of query and key
        scores = raw_scores / (self.head_dim ** 0.5)  # scale the dot product
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)  # softmax to get attention weights
        
        attended_values = torch.matmul(attention_weights, value)  # apply attention weights to value
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)  # reshape attended_values

        output = self.out_proj(attended_values)  # pass through the output projection
        output = self.layer_norm(
            output + attended_values)  # add residual connection and apply layer normalization
        return output, attention_weights  # return normalized attention weights


class MultiLayerCrossModalAttention(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads):
        super(MultiLayerCrossModalAttention, self).__init__()  # call the parent constructor
        self.layers = nn.ModuleList([CrossModalMultiHeadAttention(embed_dim, num_heads) for _ in range(num_layers)])  # a module list of multiple CrossModalMultiHeadAttention layers

    def forward(self, query, key, value):
        attention_weights = []  # optional: store per-layer attention weights
        for layer in self.layers:  # iterate over each attention layer
            query, attn_weights = layer(query, key, value)  # pass through the current layer and update query
            attention_weights.append(attn_weights)  # optional: collect per-layer attention weights
        return query, attention_weights  # return the final query and attention weights


'''
# test code

import torch
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared", "model_lib"))
from CrossModalMultiHeadAttention import MultiLayerCrossModalAttention,  CrossModalMultiHeadAttention

# model parameters
num_layers = 2       # number of attention layers
embed_dim = 512      # embedding dimension
num_heads = 8        # number of attention heads

# initialize the MultiLayerCrossModalAttention module
cross_modal_attention = MultiLayerCrossModalAttention(num_layers, embed_dim, num_heads)

# example input tensors
batch_size = 16
text_seq_len = 20
image_seq_len = 30

# randomly generated embeddings (in practice these are preprocessed text and image features)
text_embeddings = torch.randn(batch_size, text_seq_len, embed_dim)   # text modality
image_embeddings = torch.randn(batch_size, image_seq_len, embed_dim) # image modality

# perform cross-modal attention: text attends to image features
output, attn_weights = cross_modal_attention(query=text_embeddings, key=image_embeddings, value=image_embeddings)

print("output shape:", output.shape)                   # expected: (batch_size, text_seq_len, embed_dim)
print("attention weights shape:", attn_weights.shape)       # expected: (batch_size, num_heads, text_seq_len, image_seq_len)




import torch
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared", "model_lib"))
from CrossModalMultiHeadAttention import MultiLayerCrossModalAttention,  CrossModalMultiHeadAttention

# model parameters
num_layers = 5       # number of attention layers
embed_dim = 512      # embedding dimension
num_heads = 8        # number of attention heads

# initialize the MultiLayerCrossModalAttention module
cross_modal_attention = MultiLayerCrossModalAttention(num_layers, embed_dim, num_heads)

# example input tensors
batch_size = 16
text_seq_len = 20
image_seq_len = 30

# randomly generated embeddings (in practice these are preprocessed text and image features)
text_embeddings = torch.randn(batch_size, text_seq_len, embed_dim)   # text modality
image_embeddings = torch.randn(batch_size, image_seq_len, embed_dim) # image modality

# perform cross-modal attention: text attends to image features
output, attn_weights = cross_modal_attention(query=text_embeddings, key=image_embeddings, value=image_embeddings)

print("output shape:", output.shape)                   # expected: (batch_size, text_seq_len, embed_dim)
print("attention weights shape:", attn_weights.shape)       # expected: (batch_size, num_heads, text_seq_len, image_seq_len)

'''
