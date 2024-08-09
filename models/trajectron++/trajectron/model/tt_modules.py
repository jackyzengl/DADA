import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
import copy


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def attention(query, key, value,  # [head, bs, d_k]
              mask=None, dropout=None,):
    d_k = query.shape[2]
    
    key_ = key.permute(0, 2, 1)  # K^T, [head, d_k, bs]
    scores = torch.matmul(query, key_)  # Q*(K^T), [heads, bs, bs]
    scores = scores / math.sqrt(d_k)  # Q*(K^T)/sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    out_value = torch.matmul(p_attn, value)  # [head, bs, d_k]
    return out_value


class SelfAttn_layer(nn.Module):
    def __init__(self, 
                 head=4, dims_in=64, 
                 dropout=0.1,):
        super(SelfAttn_layer, self).__init__()
        assert dims_in % head == 0
        self.d_k = dims_in // head  # dims_in//h
        self.head = head
        self.linears = clones(nn.Linear(dims_in, dims_in), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, 
                query, key, value,  # [bs, 64]
                mask=None):
        nbatches = query.shape[0]  # bs
        query, key, value = [linear(x).view(nbatches, self.head, self.d_k).permute(1, 0, 2)
                             for linear, x in zip(self.linears, (query, key, value))]  # [head, bs, d_k]
        x_attned = attention(query, key, value,
                             mask=mask, dropout=self.dropout)
        x_attned = x_attned.permute(1, 0, 2).contiguous().view(nbatches, self.head * self.d_k)  # [bs, dims_in]
        return self.linears[-1](x_attned)  # [bs, dims_in]


class SelfAttn_module(nn.Module):
    def __init__(self):
        super(SelfAttn_module, self).__init__()
        self.layer = SelfAttn_layer(head=4, dims_in=64)

    def forward(self, 
                x,  # [bs, 64]
                mask=None):
        x_ = self.layer(x, x, x, mask)  # [bs, 64]
        return x_  # [bs, 64]
