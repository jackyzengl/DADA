import os
import math
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
from utils import *
from metrics import *
from model import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_HighDimTensor():
    global metrics, source_loader, target_loader

    model.eval()
    HighDim_s = []
    HighDim_t = []

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    batch_count = 0

    for index_batch in range(len_max):
        batch_count+=1

        #Get data
        if index_batch % len_source == 0:
            del source_iter
            source_iter = iter(source_loader)
            # print('*** Source Iter is reset ***')
        if index_batch % len_target == 0:
            del target_iter
            target_iter = iter(target_loader)
            # print('*** Target Iter is reset ***')
        batch_s, batch_t = next(source_iter), next(target_iter)

        batch_s = [tensor.cuda() for tensor in batch_s]
        obs_traj_s, pred_traj_gt_s, obs_traj_rel_s, pred_traj_gt_rel_s, non_linear_ped_s,\
         loss_mask_s,V_obs_s,A_obs_s,V_tr_s,A_tr_s = batch_s

        batch_t = [tensor.cuda() for tensor in batch_t]
        obs_traj_t, pred_traj_gt_t, obs_traj_rel_t, pred_traj_gt_rel_t, non_linear_ped_t,\
         loss_mask_t,V_obs_t,A_obs_t,V_tr_t,A_tr_t = batch_t

        #Forward
        V_obs_s_tmp =V_obs_s.permute(0,3,1,2)
        V_pred_s,_, hidden_s = model(V_obs_s_tmp,A_obs_s.squeeze())
        HighDim_s.append(hidden_s.detach().cpu().numpy())  # [N, feature_len×obs_len]

        V_obs_t_tmp =V_obs_t.permute(0,3,1,2)
        V_pred_t,_, hidden_t = model(V_obs_t_tmp,A_obs_t.squeeze())
        HighDim_t.append(hidden_t.detach().cpu().numpy())  # [N, feature_len×obs_len]

    HighDim_s = np.concatenate(HighDim_s, axis=0)
    HighDim_t = np.concatenate(HighDim_t, axis=0)
    print(f"### High Dim Source Tensor shape: {HighDim_s.shape} ###")
    print(f"### High Dim Target Tensor shape: {HighDim_t.shape} ###")

    return HighDim_s, HighDim_t



def visual(feat):  # [num, dim=64]
    ts = TSNE(n_components=2, init='pca', random_state=0)
    x_ts = ts.fit_transform(feat)
    # x_min, x_max = x_ts.min(0), x_ts.max(0)
    # x_final = (x_ts - x_min) / (x_max - x_min)
    x_final = x_ts
    return x_final


dada_datasets = ['A2B']
for subset in dada_datasets:
    paths = [f'./checkpoint/DADA/{subset}']
    print("*"*50)


    for feta in range(len(paths)):
        path = paths[feta]
        exps = glob.glob(path)  # 返回符合paths命名的文件名的列表
        print('Model being tested are:',exps)

        for exp_path in exps:
            print("*"*50)

            model_path = exp_path+'/val_best.pth'
            args_path = exp_path+'/args.pkl'
            with open(args_path,'rb') as f:
                args = pickle.load(f)  # train的时候把argparse保存成pkl文件了！

            stats= exp_path+'/constant_metrics.pkl'
            with open(stats,'rb') as f:
                cm = pickle.load(f)  # cm= "{'min_val_epoch': 234, 'min_val_loss': -0.014858260246866567}"
            print("Stats:",cm)



            #Data prep
            obs_seq_len = args.obs_seq_len
            pred_seq_len = args.pred_seq_len
            data_set = '../../datasets/'+args.dataset+'/'

            source_dset = TrajectoryDataset(
                    data_set+'train_origin/',
                    obs_len=obs_seq_len,
                    pred_len=pred_seq_len,
                    skip=1,norm_lap_matr=True)

            source_loader = DataLoader(
                    source_dset,
                    batch_size=1, #This is irrelative to the args batch size parameter
                    shuffle =True,
                    num_workers=0)


            target_dset = TrajectoryDataset(
                    data_set+'val/',
                    obs_len=obs_seq_len,
                    pred_len=pred_seq_len,
                    skip=1,norm_lap_matr=True)

            target_loader = DataLoader(
                    target_dset,
                    batch_size=1, #This is irrelative to the args batch size parameter
                    shuffle =False,
                    num_workers=1)

            len_source, len_target = len(source_loader), len(target_loader)
            len_max = max(len_source, len_target)

            #Defining the model
            model = stgcnn_FLA(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
            output_feat=args.output_size,seq_len=args.obs_seq_len,
            kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).cuda()
            model.load_state_dict(torch.load(model_path))

            HighDim_s, HighDim_t = get_HighDimTensor()
            LowDim_s, LowDim_t = visual(HighDim_s), visual(HighDim_t)


            fig, ax = plt.subplots()
            ax.scatter(LowDim_s[:, 0], LowDim_s[:, 1], marker='^', c='orange', s=10, label='source')
            ax.scatter(LowDim_t[:, 0], LowDim_t[:, 1], marker='.', c='cyan', s=10, label='target')
            plt.title(subset, fontsize=12, fontweight='normal')
            plt.xticks([])
            plt.yticks([])

            ax.legend()
            plt.show()

