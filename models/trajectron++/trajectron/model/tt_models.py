import torch
import torch.nn as nn
import torch.nn.functional as F

#from model.tt_modules import SelfAttn_module


class Discriminator(nn.Module):
    def __init__(self, 
                 d_conv_dim=64, d_conv_k_size=3, d_conv_padding=1,  # [64,3,1]
                 d_pool_k_size=3, d_pool_stride=2, d_pool_padding=1,  # [3,2,1]
                 gat_headdim_list=[64, 32, 64], gat_head_list=[4, 1],
                 ):
        super(Discriminator, self).__init__()
        self.conv_layers_1 = nn.Sequential(
            nn.Conv1d(2, d_conv_dim//2, 
                      kernel_size=d_conv_k_size, padding=d_conv_padding),  # [nums, 32, 20]
            nn.ReLU(),

            nn.Conv1d(d_conv_dim//2, d_conv_dim,
                      kernel_size=d_conv_k_size, padding=d_conv_padding),  # [nums, 64, 20]
            nn.ReLU(),
        )
        
        self.conv_layers_2 = nn.Sequential(
            nn.Conv1d(d_conv_dim, d_conv_dim//2, 
                      kernel_size=d_conv_k_size, padding=d_conv_padding),  # [nums, 32, 20]
            nn.ReLU(),

            nn.Conv1d(d_conv_dim//2, 1, 
                      kernel_size=d_conv_k_size, padding=d_conv_padding),  # [nums, 1, 20]
        )

    def forward(self, 
                traj,  # [nums, 20, 2]
                traj_rel,  # [nums, 20, 2]
                ):
        input_ = traj_rel.permute(0, 2, 1)  # [nums, 2, 20]
        
        feature_1 = self.conv_layers_1(input_)  # [nums, 64, 20]
        output = self.conv_layers_2(feature_1)  # [nums, 1, 20]
        
        output_ = output.squeeze()  # [nums, 20]
        return output_  # fake_score, [nums, 20]


class Generator(nn.Module):
    def __init__(self, 
                 g_conv_dim=64, g_conv_k_size=5, g_conv_pad_size=2,  # [64,3,1]
                 gat_headdim_list=[64, 32, 64], gat_head_list=[4, 1],
                 ):
        super(Generator, self).__init__()
        self.conv_layers_1 = nn.Sequential(
            nn.Conv1d(2, g_conv_dim//2, 
                      kernel_size=g_conv_k_size, padding=g_conv_pad_size),  # [nums, 32, 20]
            nn.ReLU(),

            nn.Conv1d(g_conv_dim//2, g_conv_dim, 
                      kernel_size=g_conv_k_size, padding=g_conv_pad_size),  # [nums, 64, 20]
            nn.ReLU(),
        )
        
        self.conv_layers_2 = nn.Sequential(
            nn.Conv1d(g_conv_dim, g_conv_dim//2, 
                      kernel_size=g_conv_k_size, padding=g_conv_pad_size),  # [nums, 32, 20]
            nn.ReLU(),

            nn.Conv1d(g_conv_dim//2, 2, 
                      kernel_size=g_conv_k_size, padding=g_conv_pad_size),  # [nums, 2, 20]
        )

    def forward(self, 
                traj,  # [nums, 20, 2]
                traj_rel,  # [nums, 20, 2]
                ):
        input_ = traj_rel.permute(0, 2, 1)  # [nums, 2, 20]
        
        feature_1 = self.conv_layers_1(input_)  # [nums, 64, 20]
        output = self.conv_layers_2(feature_1)  # [nums, 2, 20]
        
        output_ = output.permute(0, 2, 1)  # [nums, 20, 2]
        return output_  # fake_traj_rel, [nums, 20, 2]


class Discriminator(nn.Module):
    def __init__(self, 
                 d_conv_dim=64, d_conv_k_size=3, d_conv_padding=1,  # [64,3,1]
                 d_pool_k_size=3, d_pool_stride=2, d_pool_padding=1,  # [3,2,1]
                 gat_headdim_list=[64, 32, 64], gat_head_list=[4, 1],
                 ):
        super(Discriminator, self).__init__()
        self.conv_layers_1 = nn.Sequential(
            nn.Conv1d(2, d_conv_dim//2, 
                      kernel_size=d_conv_k_size, padding=d_conv_padding),  # [nums, 32, 20]
            nn.ReLU(),

            nn.Conv1d(d_conv_dim//2, d_conv_dim,
                      kernel_size=d_conv_k_size, padding=d_conv_padding),  # [nums, 64, 20]
            nn.ReLU(),
        )
        
        self.conv_layers_2 = nn.Sequential(
            nn.Conv1d(d_conv_dim, d_conv_dim//2, 
                      kernel_size=d_conv_k_size, padding=d_conv_padding),  # [nums, 32, 20]
            nn.ReLU(),

            nn.Conv1d(d_conv_dim//2, 1, 
                      kernel_size=d_conv_k_size, padding=d_conv_padding),  # [nums, 1, 20]
        )

    def forward(self, 
                traj,  # [nums, 20, 2]
                traj_rel,  # [nums, 20, 2]
                ):
        input_ = traj_rel.permute(0, 2, 1)  # [nums, 2, 20]
        
        feature_1 = self.conv_layers_1(input_)  # [nums, 64, 20]
        output = self.conv_layers_2(feature_1)  # [nums, 1, 20]
        
        output_ = output.squeeze()  # [nums, 20]
        return output_  # fake_score, [nums, 20]


class Discriminator_inlay(nn.Module):
    def __init__(self,
                 input_dim=64, 
                 mlp_dim=64,
                 ):
        super(Discriminator_inlay, self).__init__()
        
        #self.attn_layer = SelfAttn_module()
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
                feature_in,  # fake or real[nums, 64]
                ):
        feature_out = self.mlp_layers(feature_in)  # [nums, 1]
        return feature_out  # fake_score, [nums, 1]
