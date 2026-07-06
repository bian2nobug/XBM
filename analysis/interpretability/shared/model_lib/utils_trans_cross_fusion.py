import os
import torch
from torch import nn
from math import ceil
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared", "model_lib"))
from LongTokenSeqEncoder import PerceiverIOSequenceEncoder
from CrossModalMultiHeadAttention import  MultiLayerCrossModalAttention
from ClassificationHead_layernorm import ClassificationHead_layernorm
from regression_head import regression_head
from TokenViT_2 import TokenViT
from math import ceil                  # for ceiling division

# The modules below are only required by the legacy fusion classes (multi_crossmodel/LinearEmbed, etc.)
# that are unused in this file; the current backbone only needs AttnPool1D. To keep the repo self-contained
# (avoiding extra dependencies such as einops), these heavy imports are fault-tolerant; if missing, the
# related legacy classes are unavailable but the interpretability main flow is unaffected.
try:
    from Transformer import Transformer_Encoder
except ImportError:
    Transformer_Encoder = None
try:
    from MLPEncoder import MLPEncoder
except ImportError:
    MLPEncoder = None


class LinearEmbed(nn.Module):
    """ (B, dim_feature) -> (B, dim_feature, 256)
        Map a 1D feature (each dim treated as a "token") to a vector of length out_dim, then stack back on dim 1
    """
    def __init__(self, dim_feature: int, out_dim: int = 256):
        super().__init__()                                             # call the parent constructor
        # create an independent linear layer for each scalar feature: in=1, out=out_dim
        self.proj_list = nn.ModuleList([nn.Linear(1, out_dim)          # linear maps (B,1) -> (B,out_dim)
                                        for _ in range(dim_feature)])   # dim_feature of them in total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, dim_feature)                                      # input: batch x feature dim
        outs = []                                                  # collect the mapped vector of each feature
        for i, layer in enumerate(self.proj_list):                 # iterate over each linear layer
            outs.append(layer(x[:, i:i+1]))                       # take the i-th scalar (B,1) -> (B,out_dim)
        return torch.stack(outs, dim=1)                            # stack into (B, dim_feature, out_dim)


class AttnPool1D(nn.Module):
    """ Attention pooling: a learnable weighted average over the sequence dim, output (B, D) """
    def __init__(self, dim, hidden=128, dropout=0.0):
        super().__init__()                                         # parent constructor
        # scoring network: LN first, then a two-layer MLP producing a scalar weight
        self.score = nn.Sequential(
            nn.LayerNorm(dim),                                     # layer-norm each token vector
            nn.Linear(dim, hidden),                                # reduce to hidden
            nn.Tanh(),                                             # activation (could be GELU)
            nn.Linear(hidden, 1)                                   # output scalar score (B,L,1)
        )
        self.drop = nn.Dropout(dropout)                            # dropout on the weights (regularization)

    def forward(self, x, mask=None):
        """
        x: (B, L, D)                                               # input sequence
        mask: (B, L)  True=valid, False=padding (optional; implemented below per this convention)
        """
        s = self.score(x).squeeze(-1)                              # (B, L) drop the last dim of the score
        if mask is not None:                                       # if a mask is provided
            s = s.masked_fill(~mask, -1e9)                         # set padding positions to a very small score
        w = torch.softmax(s, dim=-1)                               # (B, L) normalize to weights
        w = self.drop(w)                                           # randomly zero out weights
        out = (w.unsqueeze(-1) * x).sum(dim=1)                     # weighted sum -> (B, D)
        return out                                                 # return the pooled vector


class PMA(nn.Module):
    """ Pooling by Multihead Attention (Set Transformer): read out a global representation with a learnable query """
    def __init__(self, dim, n_heads=8, dropout=0.2):
        super().__init__()                                         # parent constructor
        self.seed = nn.Parameter(torch.randn(1, 1, dim))           # (1,1,D) learnable global query
        self.mha  = nn.MultiheadAttention(dim, n_heads,            # multi-head attention module
                                          dropout=dropout,
                                          batch_first=True)         # use the (B,L,E) interface
        self.norm = nn.LayerNorm(dim)                              # layer-norm the output

    def forward(self, x, key_padding_mask=None):
        """
        x: (B, L, D)                                               # input sequence
        key_padding_mask: (B, L), True=padding (matches nn.MultiheadAttention semantics)
        """
        q = self.seed.expand(x.size(0), -1, -1)                    # (B,1,D) expand query to batch size
        out, attn = self.mha(q, x, x,                              # attend q over the whole sequence
                             key_padding_mask=key_padding_mask)    # mask out padding positions
        return self.norm(out.squeeze(1)), attn                     # squeeze the length dim -> (B,D) and attn (B,1,L)


class SplitModalities(nn.Module):
    """ Split the last dim into three modalities by (dim1, dim2, dim3), taking the first step's dim2/dim3 as vectors """
    def __init__(self, dim1=1536, dim2=1, dim3=10):
        super().__init__()                                         # parent constructor
        self.sizes = [dim1, dim2, dim3]                            # record the channel size of each modality

    def forward(self, x):
        # x: (B, N, sum(sizes))                                    # input: batch x seq len x total channels
        m1, m2, m3 = torch.split(x, self.sizes, dim=-1)            # split the last dim into three parts
        # m1: (B, N, dim1)                                         # first modality (e.g. pathology MIL features)
        # m2: (B, N, dim2)                                         # second modality (e.g. subtype)
        # m3: (B, N, dim3)                                         # third modality (e.g. clinical clin)
        MILfeature = m1                                            # keep the whole sequence for the main modality
        subtype = m2[:, 0, :]                                      # take only position 0 -> (B, dim2)
        clin    = m3[:, 0, :]                                      # same -> (B, dim3)
        return MILfeature, subtype, clin                           # return the three modalities


class multi_crossmodel(nn.Module):
    """ Backbone: cross-attention fusion of a main modality (e.g. pathology) + two auxiliary modalities (subtype/clinic) + pooling + classification """
    def __init__(self, 
                 split_dims=(1536, 3, 65),                         # channel sizes of the three modalities (matches the input last-dim split)
                 Transformer_dim=512,                               # Transformer dim for the main modality after MLP
                 Transformer_depth=2,                               # number of Transformer layers
                 Transformer_head=2,                                # number of attention heads
                 Cross_num_layers=2,                                # number of stacked cross-attention layers
                 Cross_embed_dim=256,                               # common embedding dim for cross-attention
                 Cross_num_heads=2,                                 # number of cross-attention heads
                 Classify_dim=3,                                    # number of output classes
                 linearE=LinearEmbed,                               # module mapping a vector modality to a sequence
                 Pooling=AttnPool1D,                                # pooling module (AttnPool1D / PMA, etc.)
                 second_cross = True,
                 regression = True,
                 first_subtype = True,                           # True: fuse the subtype modality first; False: fuse clinic first
                 # ================== four extra switches, passed through to TokenViT ==================
                 enc_return_attention=False,   # whether the encoder returns attention
                 enc_last_only=True,           # keep only the last-layer attention
                 enc_topk=None,                # attention Top-K, can be None
                 enc_cpu_offload=True):        # move attention to CPU
        super().__init__()                                         # parent constructor

        d1, d2, d3 = split_dims                                    # unpack per-modality channel sizes

        self.split = SplitModalities(dim1=d1, dim2=d2, dim3=d3)    # build the splitter
                                          # dropout probability

        self.Transformer_Encoder = TokenViT(in_dim=d1,
                                               num_tokens=12000, 
                                               dim=Transformer_dim, 
                                               depth=Transformer_depth, 
                                               heads=Transformer_head,
                                               dim_head=int(ceil(Transformer_dim/Transformer_head)), 
                                               mlp_dim=Transformer_dim, 
                                               num_classes=Transformer_dim, 
                                               pool_stride=1)



        self.proj_histo = nn.Linear(Transformer_dim,                # map the main-modality channels
                                    Cross_embed_dim)                # to the common cross-attention dim

        # map the vector modality (B, d2) to a sequence (B, d2, Cross_embed_dim)
        self.proj_subtype = linearE(dim_feature=d2,                 # number of feature dims = sequence length
                                    out_dim=Cross_embed_dim)        # dim of each "token"

        # map the vector modality (B, d3) to a sequence (B, d3, Cross_embed_dim)
        self.proj_Clinic  = linearE(dim_feature=d3,                 # same as above
                                    out_dim=Cross_embed_dim)

        # cross-attention: stage 1 (subtype as Q, main modality as K/V)
        self.MultiLayerCrossModalAttention_Subtype = MultiLayerCrossModalAttention(
            num_layers=Cross_num_layers,                            # number of stacked layers
            embed_dim=Cross_embed_dim,                              # shared embedding dim
            num_heads=Cross_num_heads)                              # number of heads

        # cross-attention: stage 2 (clinic as Q, stage-1 output as K/V)
        self.MultiLayerCrossModalAttention_Clinic  = MultiLayerCrossModalAttention(
            num_layers=Cross_num_layers,
            embed_dim=Cross_embed_dim,
            num_heads=Cross_num_heads)

        self.pool = Pooling(Cross_embed_dim,hidden = Cross_embed_dim*2)                        # pool to (B, Cross_embed_dim)

        if regression:
            self.head = regression_head(input_dim = Cross_embed_dim, output_dim = Classify_dim)
        else:
            self.head = ClassificationHead_layernorm(input_dim = Cross_embed_dim, output_dim = Classify_dim)

        self.second_cross = second_cross

        self.multi_score = {}

        self.first_subtype = first_subtype

        self.enc_return_attention = enc_return_attention       # whether TokenViT returns attention
        self.enc_last_only       = enc_last_only       # keep only the last-layer attention
        self.enc_topk            = enc_topk        # set an integer k for sparse return; None=full
        self.enc_cpu_offload     = enc_cpu_offload       # move attention to CPU to save GPU memory

    def forward(self, x):
        # x: (B, N, d1+d2+d3)                                       # input: the three modalities concatenated as a sequence

        histo_feat, subtype, clin = self.split(x)                   # split three modalities; subtype/clin are (B,d2)/(B,d3)

        histo_feat_2,self.enc_attn  = self.Transformer_Encoder(histo_feat, 
                                                               return_attention=self.enc_return_attention, 
                                                               last_only=self.enc_last_only, 
                                                               topk=self.enc_topk, 
                                                               cpu_offload=self.enc_cpu_offload)       # main-modality sequence Transformer encoding (keeps N)


        histo_feat_3 = self.proj_histo(histo_feat_2)                # map main-modality channels to Cross_embed_dim

        if self.second_cross:
            subtype_feat_1 = self.proj_subtype(subtype)
            clin_feat_1    = self.proj_Clinic(clin)
        else:
            if self.first_subtype:
                subtype_feat_1 = self.proj_subtype(subtype)
            else:
                clin_feat_1    = self.proj_Clinic(clin)    


        # stage-1 cross-attention: Q=subtype sequence, K=main-modality sequence, V=main-modality sequence
        # the first returned tensor has shape (B, d2, Cross_embed_dim),
        # histo_feat_4: "subtype representation that has fused main-modality information"

        feat_first = subtype_feat_1 if self.first_subtype == True else clin_feat_1

        histo_feat_4, self.attn_weights_bySubtype = \
            self.MultiLayerCrossModalAttention_Subtype(
                query=feat_first,                              # Q: subtype sequence (length d2)
                key=histo_feat_3,                                  # K: main-modality sequence (length N)
                value=histo_feat_3)                                # V: main-modality sequence (length N)
                
        # self.attn_weights_bySubtype shape is typically: a list or the last layer (B, heads, d2, N)
        
        

        if self.second_cross == True:
            # stage-2 cross-attention: Q=clin sequence, K/V=stage-1 output (length d2)
            # returned tensor has shape (B, d3, Cross_embed_dim); it means "the Clin representation that fused Subtype->Histo"
            
            feat_second = clin_feat_1 if self.first_subtype == True else subtype_feat_1

            histo_feat_5, self.attn_weights_byClinic = \
                self.MultiLayerCrossModalAttention_Clinic(
                    query=feat_second,                                 # Q: clin sequence (length d3)
                    key=histo_feat_4,                                  # K: stage-1 output (length d2)
                    value=histo_feat_4)                                # V: same as K
            # self.attn_weights_byClinic shape: (B, heads, d3, d2)

            histo_feat_6 = self.pool(histo_feat_5)                     # pool (B, d3, D) -> (B, D)

            histo_feat_7 = self.head(histo_feat_6)                     # classification head -> (B, Classify_dim)
        
        else:
            histo_feat_5 = self.pool(histo_feat_4)

            histo_feat_7 = self.head(histo_feat_5)
        
        #------ attention interface --------!!
        
        self.multi_score = {
            "cross_subtype": self.attn_weights_bySubtype,   # shape ~ (B, heads, d2, N)
            "enc_attn": self.enc_attn                            # list or tensor, from TokenViT
        }

        return histo_feat_7                                        # return prediction logits

# Here the Transformer is swapped for Perceiver IO (see the multimodal deep learning WGD regression task slides)
class multi_crossmodel_IOS(nn.Module):
    """ Backbone: cross-attention fusion of a main modality (e.g. pathology) + two auxiliary modalities (subtype/clinic) + pooling + classification """
    def __init__(self, 
                 split_dims=(1536, 3, 65),                         # channel sizes of the three modalities (matches the input last-dim split)
                 Transformer_dim=512,                               # Transformer dim for the main modality after MLP
                 Transformer_depth=2,                               # number of Transformer layers
                 Transformer_head=2,                                # number of attention heads
                 Cross_num_layers=2,                                # number of stacked cross-attention layers
                 Cross_embed_dim=256,                               # common embedding dim for cross-attention
                 Cross_num_heads=2,                                 # number of cross-attention heads
                 Classify_dim=3,                                    # number of output classes
                 linearE=LinearEmbed,                               # module mapping a vector modality to a sequence
                 Pooling=AttnPool1D,                                # pooling module (AttnPool1D / PMA, etc.)
                 second_cross = True,                               # how many modalities the two-stage cross-attention processes
                 regression = True,
                 first_subtype = True):                              
        super().__init__()                                         # parent constructor

        d1, d2, d3 = split_dims                                    # unpack per-modality channel sizes

        self.split = SplitModalities(dim1=d1, dim2=d2, dim3=d3)    # build the splitter
                                          # dropout probability

        self.Transformer_Encoder = PerceiverIOSequenceEncoder(
            in_dim=d1,              # 1536
            dim=Transformer_dim,    # 256
            depth=Transformer_depth,                # 2~3
            heads=Transformer_head,                # 6~8
            num_latents=128,        # 64~192
            dropout=0.2,
            ffn_mult=4.0,
            add_pos_emb=True,
            track_attn=True        # set True for visualization
        )
        


        self.proj_histo = nn.Linear(Transformer_dim,                # map the main-modality channels
                                    Cross_embed_dim)                # to the common cross-attention dim

        # map the vector modality (B, d2) to a sequence (B, d2, Cross_embed_dim)
        self.proj_subtype = linearE(dim_feature=d2,                 # number of feature dims = sequence length
                                    out_dim=Cross_embed_dim)        # dim of each "token"

        # map the vector modality (B, d3) to a sequence (B, d3, Cross_embed_dim)
        self.proj_Clinic  = linearE(dim_feature=d3,                 # same as above
                                    out_dim=Cross_embed_dim)

        self.CrossBlock = MultiLayerCrossModalAttention(
            num_layers=Cross_num_layers, embed_dim=Cross_embed_dim, num_heads=Cross_num_heads)

        self.pool = Pooling(Cross_embed_dim)                        # pool to (B, Cross_embed_dim)

        if regression:
            self.head = regression_head(input_dim = Cross_embed_dim, output_dim = Classify_dim)
        else:
            self.head = ClassificationHead_layernorm(input_dim = Cross_embed_dim, output_dim = Classify_dim)

        self.second_cross = second_cross

        self.first_subtype = first_subtype

        self.multi_score = {}

    def forward(self, x):
        # x: (B, N, d1+d2+d3)                                       # input: the three modalities concatenated as a sequence

        histo_feat, subtype, clin = self.split(x)                   # split three modalities; subtype/clin are (B,d2)/(B,d3)

        histo_feat_2,self.enc_attn = self.Transformer_Encoder(histo_feat)       # main-modality sequence Transformer encoding (keeps N)

        histo_feat_3 = self.proj_histo(histo_feat_2)                # map main-modality channels to Cross_embed_dim
        
        if self.second_cross == True:
            subtype_feat_1 = self.proj_subtype(subtype)                 # (B,d2) -> (B,d2,Cross_embed_dim) treated as a sequence
                                                                     # here each subtype dim is treated as a token
            clin_feat_1 = self.proj_Clinic(clin)                        # (B,d3) -> (B,d3,Cross_embed_dim)
        else:
            if self.first_subtype == True:
                subtype_feat_1 = self.proj_subtype(subtype)
            else:
                clin_feat_1 = self.proj_Clinic(clin)    
            
        # stage-1 cross-attention: Q=subtype sequence, K=main-modality sequence, V=main-modality sequence
        # the first returned tensor has shape (B, d2, Cross_embed_dim),
        # histo_feat_4: "subtype representation that has fused main-modality information"
        feat_first = subtype_feat_1 if self.first_subtype == True else clin_feat_1

        histo_feat_4, self.attn_weights_bySubtype = \
            self.CrossBlock(
                query=feat_first,                              # Q: subtype sequence (length d2)
                key=histo_feat_3,                                  # K: main-modality sequence (length N)
                value=histo_feat_3)                                # V: main-modality sequence (length N)
        # self.attn_weights_bySubtype shape is typically: a list or the last layer (B, heads, d2, N)
        if self.second_cross == True:
            # stage-2 cross-attention: Q=clin sequence, K/V=stage-1 output (length d2)
            # returned tensor has shape (B, d3, Cross_embed_dim); it means "the Clin representation that fused Subtype->Histo"
            
            feat_second = clin_feat_1 if self.first_subtype == True else subtype_feat_1
            
            histo_feat_5, self.attn_weights_byClinic = \
                self.CrossBlock(
                    query=feat_second,                                 # Q: clin sequence (length d3)
                    key=histo_feat_4,                                  # K: stage-1 output (length d2)
                    value=histo_feat_4)                                # V: same as K
            # self.attn_weights_byClinic shape: (B, heads, d3, d2)

            histo_feat_6 = self.pool(histo_feat_5)                     # pool (B, d3, D) -> (B, D)

            histo_feat_7 = self.head(histo_feat_6)                     # classification head -> (B, Classify_dim)
        
        else:
            histo_feat_5 = self.pool(histo_feat_4)

            histo_feat_7 = self.head(histo_feat_5)
        
        #------ attention interface --------!!

        
        self.multi_score = {
            "cross_subtype": self.attn_weights_bySubtype,   # shape ~ (B, heads, d2, N)
            "enc_attn": self.enc_attn                            # list or tensor, from TokenViT
        }

        return histo_feat_7                                        # return prediction logits
                                         
'''

import torch
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared", "model_lib"))
from utils_trans_cross_fusion import multi_crossmodel,LinearEmbed,AttnPool1D,multi_crossmodel_IOS

c = torch.randn(2,20000,1604)

model = multi_crossmodel(
                 split_dims=(1536, 3, 65),

                 Transformer_dim=512, 
                 Transformer_depth=2,
                 Transformer_head = 2,

                 Cross_num_layers = 2,
                 Cross_embed_dim = 256,
                 Cross_num_heads = 2,
                 
                 Classify_dim = 1,
                 
                 linearE=LinearEmbed,                               # module mapping a vector modality to a sequence
                 Pooling=AttnPool1D,                                # pooling module (AttnPool1D / PMA, etc.)
                 second_cross = False,
                 regression = False,
                 first_subtype = False,
                 enc_return_attention=True,   # whether the encoder returns attention
                 enc_last_only=True,           # keep only the last-layer attention
                 enc_topk=None,                # attention Top-K, can be None
                 enc_cpu_offload=True)                              # number of output classes

f = model(c)
c = model.multi_score

print(c['cross_subtype'].shape,c['enc_attn'].shape)

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared", "model_lib"))
from model_summary import model_summary
model_summary(model, input_size=(2, 20000, 1604))

import torch
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared", "model_lib"))
from utils_trans_cross_fusion import multi_crossmodel,LinearEmbed,AttnPool1D,multi_crossmodel_IOS

c = torch.randn(2,20000,1604)

model = multi_crossmodel_IOS(
                 split_dims=(1536, 3, 65),

                 Transformer_dim=512, 
                 Transformer_depth=2,
                 Transformer_head = 2,

                 Cross_num_layers = 2,
                 Cross_embed_dim = 256,
                 Cross_num_heads = 2,
                 
                 Classify_dim = 1,
                 
                 linearE=LinearEmbed,                               # module mapping a vector modality to a sequence
                 Pooling=AttnPool1D,                                # pooling module (AttnPool1D / PMA, etc.)
                 second_cross = False,
                 regression = False,
                 first_subtype = False)                              # number of output classes

f = model(c)
c = model.multi_score

print(c['cross_subtype'].shape,c['enc_attn'].shape)




'''
