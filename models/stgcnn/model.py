import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim


class ConvTemporalGraphical(nn.Module):
    # Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel。特征提取时考虑的帧数
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,  # 2
                 out_channels,  # 5
                 kernel_size,  # obs_len = 8
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        """
        args:
            x: [1, 2, obs_len, N]
            A: [obs_len, N, N]
        return:
            x: [1, output_channel, obs_len, N]
            A: [obs_len, N, N]
        """
        assert A.size(0) == self.kernel_size
        x = self.conv(x)

        x = torch.einsum('nctv,tvw->nctw', (x, A))  # [output_channel,N] @ [N,N]
        return x.contiguous(), A


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,  # 2
                 out_channels,  # 5
                 kernel_size,  # (kernel_size, obs_len) = (3, 8)
                 use_mdn=False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn, self).__init__()

        #         print("outstg",out_channels)

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):
        """
        args:
            x: [1， 2, obs_len, N]
            A: [obs_len, N, N]
        return:
            x: [1， 2, obs_len, N]
            A: [obs_len, N, N]
        """

        res = self.residual(x)
        x, A = self.gcn(x, A)  # x:[1, output_channel, obs_len, N]; A:[obs_len, N, N]
        x_afterGCN = x.clone().view(x.shape[-1], -1)  # [N, feat_len*obs_len]

        x = self.tcn(x) + res

        if not self.use_mdn:
            x = self.prelu(x)

        return x, A, x_afterGCN


class social_stgcnn(nn.Module):
    def __init__(self, n_stgcnn=1, n_txpcnn=1, input_feat=2, output_feat=5,
                 seq_len=8, pred_seq_len=12, kernel_size=3):
        super(social_stgcnn, self).__init__()
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn

        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat, output_feat, (kernel_size, seq_len)))
        for j in range(1, self.n_stgcnn):
            self.st_gcns.append(st_gcn(output_feat, output_feat, (kernel_size, seq_len)))

        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len, pred_seq_len, 3, padding=1))
        for j in range(1, self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1)

        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())

    def forward(self, v, a):
        """
        args:
            v: [1, 2, obs_len, N]
            a: [obs_len, N, N]
        return:
            v: [1, feature_len, pred_len, N]
            a: [obs_len, N, N]
        """

        for k in range(self.n_stgcnn):
            v, a, _ = self.st_gcns[k](v, a)

        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])  # [1, obs_len, feature_len, N]

        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1, self.n_txpcnn - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v

        v = self.tpcnn_ouput(v)  # [1, pred_len, feature_len, N]
        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        return v, a


class stgcnn_FLA(nn.Module):
    def __init__(self, n_stgcnn=1, n_txpcnn=1, input_feat=2, output_feat=5,
                 seq_len=8, pred_seq_len=12, kernel_size=3):
        super(stgcnn_FLA, self).__init__()
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn

        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat, output_feat, (kernel_size, seq_len)))
        for j in range(1, self.n_stgcnn):
            self.st_gcns.append(st_gcn(output_feat, output_feat, (kernel_size, seq_len)))

        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len, pred_seq_len, 3, padding=1))
        for j in range(1, self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1)

        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())

    def forward(self, v, a):
        """
        args:
            v: [1, 2, obs_len, N]
            a: [obs_len, N, N]
        return:
            v: [1, feature_len, pred_len, N]
            a: [obs_len, N, N]
            hidden: [N, feature_len×obs_len]
        """

        for k in range(self.n_stgcnn):
            v, a, hidden = self.st_gcns[k](v, a)

        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])  # [1, obs_len, feature_len, N]
        hidden = v.clone().view(v.shape[3], -1)

        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1, self.n_txpcnn - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v

        v = self.tpcnn_ouput(v)  # [1, pred_len, feature_len, N]
        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        return v, a, hidden


class Discriminator_inlay(nn.Module):
    def __init__(self,
                 input_dim=40,
                 mlp_dim=80,
                 ):
        super(Discriminator_inlay, self).__init__()

        # self.attn_layer = SelfAttn_module()
        self.mlp_layers = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),
        )

    def forward(self,
                feature_in,  # fake or real[nums, 5*8]
                ):
        feature_out = self.mlp_layers(feature_in)  # [nums, 1]
        return feature_out  # fake_score, [nums, 1]
