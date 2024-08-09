import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from modules import *


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
                traj,  # [20, nums, 2]
                traj_rel,  # [20, nums, 2]
                seq_start_end,):  # [(start0, end0), (start1, end1), ...])
        input_ = traj_rel.permute(1, 2, 0)  # [nums, 2, 20]
        
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
        
        self.gat_module = GAT_module(gat_headdim_list, gat_head_list)  # [nums, 64, 20]
        self.fc = nn.Linear(g_conv_dim*2, g_conv_dim*2)  # [nums, 20, 128]
        
        self.conv_layers_2 = nn.Sequential(
            nn.Conv1d(g_conv_dim*2, g_conv_dim, 
                      kernel_size=g_conv_k_size, padding=g_conv_pad_size),  # [nums, 64, 20]
            nn.ReLU(),
            
            nn.Conv1d(g_conv_dim, g_conv_dim//2, 
                      kernel_size=g_conv_k_size, padding=g_conv_pad_size),  # [nums, 32, 20]
            nn.ReLU(),

            nn.Conv1d(g_conv_dim//2, 2, 
                      kernel_size=g_conv_k_size, padding=g_conv_pad_size),  # [nums, 2, 20]
        )

    def forward(self, 
                traj,  # [20, nums, 2]
                traj_rel,  # [20, nums, 2]
                seq_start_end,):  # [(start0, end0), (start1, end1), ...]
        input_ = traj_rel.permute(1, 2, 0)  # [nums, 2, 20]
        
        feature_1 = self.conv_layers_1(input_)  # [nums, 64, 20]
        
        feature_gat = self.gat_module(feature_1, seq_start_end)  # [nums, 64, 20]
        
        feature_2 = torch.cat([feature_1, feature_gat], dim=1)  # [nums, 128, 20]
        feature_2_ = feature_2.permute(0, 2, 1)  # [nums, 20, 128]
        feature_2_ = self.fc(feature_2_)
        feature_2_ = feature_2_.permute(0, 2, 1)  # [nums, 128, 20]
        
        output = self.conv_layers_2(feature_2_)  # [nums, 2, 20]
        
        output_ = output.permute(2, 0, 1)  # [20, nums, 2]
        return output_  # fake_traj_rel, [20, nums, 2]


if __name__ == '__main__':
    # for test this script
    from .utils import relative_to_abs
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    def init_weights(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight)
    generater = Generator(g_conv_dim=64, g_conv_k_size=5, g_conv_pad_size=2,
                          gat_headdim_list=[64, 32, 64], gat_head_list=[4, 1],).cuda()
    generater.apply(init_weights)
    discriminator = Discriminator(d_conv_dim=64, d_conv_k_size=3, d_conv_padding=1,
                                  d_pool_k_size=3, d_pool_stride=2, d_pool_padding=1,
                                  gat_headdim_list=[64, 32, 64], gat_head_list=[4, 1],).cuda()
    discriminator.apply(init_weights)

    whole_traj = torch.randn([20, 25, 2]).cuda()
    whole_traj_rel = torch.randn([20, 25, 2]).cuda()
    start_end = torch.LongTensor([[0, 4], [4, 9], [9, 18], [18, 22], [22, 25]]).cuda()

    fake_traj_rel = generater(whole_traj, whole_traj_rel, start_end)
    print(fake_traj_rel.shape)  # [20, 25, 2]
    fake_traj = relative_to_abs(fake_traj_rel, whole_traj[0])
    score_fake = discriminator(fake_traj, fake_traj_rel, start_end)
    print(score_fake.shape)  # [25, 20]
    print("****************************")

    loss_func = nn.MSELoss()
    y_real = torch.ones_like(score_fake)
    print(loss_func(score_fake, y_real))
