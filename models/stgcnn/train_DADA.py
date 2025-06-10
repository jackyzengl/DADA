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

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx

from utils import *
from metrics import *
import pickle
import argparse
from torch import autograd
import torch.optim.lr_scheduler as lr_scheduler
from model import *

torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()

#Model specific parameters
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_stgcnn', type=int, default=1,help='Number of ST-GCNN layers')
parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--mlp_dim', type=int, default=80)

#Data specifc paremeters
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)
parser.add_argument('--dataset', default='eth',
                    help='eth,hotel,univ,zara1,zara2')

#Training specifc parameters
parser.add_argument('--batch_size', type=int, default=128,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=250,
                    help='number of epochs')
parser.add_argument('--clip_grad', type=float, default=None,
                    help='gadient clipping')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--lr_d_inlay', type=float, default=0.01,
                    help='learning rate of discriminator inlay')
parser.add_argument('--lr_sh_rate', type=int, default=150,
                    help='number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true", default=False,
                    help='Use lr rate scheduler')
parser.add_argument('--tag', default='tag',
                    help='personal tag for the model ')

args = parser.parse_args()


def graph_loss(V_pred,V_target):
    return bivariate_loss(V_pred,V_target)


def train(epoch, subset):
    global metrics, source_loader, target_loader
    model.train()
    d_inlay.train()

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    d_loss_batch, g_loss_batch, pred_loss_batch = 0, 0, 0
    batch_count = 0
    is_fst_loss = True
    turn_point =int(len_max/args.batch_size)*args.batch_size + len_max%args.batch_size -1


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

        batch_s = [tensor.to(device) for tensor in batch_s]
        obs_traj_s, pred_traj_gt_s, obs_traj_rel_s, pred_traj_gt_rel_s, non_linear_ped_s,\
         loss_mask_s,V_obs_s,A_obs_s,V_tr_s,A_tr_s = batch_s

        batch_t = [tensor.to(device) for tensor in batch_t]
        obs_traj_t, pred_traj_gt_t, obs_traj_rel_t, pred_traj_gt_rel_t, non_linear_ped_t,\
         loss_mask_t,V_obs_t,A_obs_t,V_tr_t,A_tr_t = batch_t
        """
        obs_traj: [1,N,2,obs_len]
        pred_traj: [1,N,2,pred_len]
        obs_traj_rel: [1,N,2,obs_len]
        pred_traj_rel: [1,N,2,pred_len]
        non_linear_ped: useless
        loss_mask: [1,N,seq_len]
        v_obs: [1,obs_len, N, 2]
        A_obs: [1,obs_len, N, N]
        v_pred: [1,pred_len, N, 2]
        A_pred: [1,pred_len, N, N]
        """

        #Forward
        V_obs_s =V_obs_s.contiguous().permute(0,3,1,2)
        V_pred_s,_, _ = model(V_obs_s,A_obs_s.squeeze())
        _ , _, hidden_s = model(V_obs_s, A_obs_s.squeeze())
        V_pred_s = V_pred_s.permute(0,2,3,1)
        score_fake = d_inlay(hidden_s)

        V_tr_s = V_tr_s.squeeze()  # [pred_len, N, 2]
        A_tr_s = A_tr_s.squeeze()
        V_pred_s = V_pred_s.squeeze()  # [pred_len, N, feature_len]

        V_obs_t =V_obs_t.contiguous().permute(0,3,1,2)
        V_pred_t, _, _ = model(V_obs_t,A_obs_t.squeeze())
        _, _, hidden_t = model(V_obs_t, A_obs_t.squeeze())
        V_pred_t = V_pred_t.permute(0,2,3,1)
        score_real = d_inlay(hidden_t)

        V_tr_t = V_tr_t.squeeze()  # [pred_len, N, 2]
        A_tr_t = A_tr_t.squeeze()
        V_pred_t = V_pred_t.squeeze()  # [pred_len, N, feature_len]

        # # 相当于手动制造batch=args.batch_size
        # if batch_count%args.batch_size !=0 and index_batch != turn_point:
        #     # 1.Discriminator
        #     d_loss_real, d_loss_fake = gan_d_loss(score_real, score_fake, mode= 'mse')
        #     d_inlay_l = d_loss_real + d_loss_fake
        #
        #     # 2.Generator
        #     g_inlay_l = gan_g_loss(score_fake, mode= 'mse')
        #
        #     # 3.predictor
        #     pred_l = graph_loss(V_pred_s,V_tr_s)
        #
        #     if is_fst_loss:
        #         d_inlay_loss = d_inlay_l
        #         g_inlay_loss = g_inlay_l
        #         pred_loss = pred_l
        #         is_fst_loss = False
        #
        #     else:
        #         d_inlay_loss += d_inlay_l
        #         g_inlay_loss += g_inlay_l
        #         pred_loss += pred_l
        #
        # else:
        #     d_inlay_loss = (d_inlay_loss / args.batch_size)
        #     g_inlay_loss = (g_inlay_loss / args.batch_size)
        #     pred_loss = (pred_loss / args.batch_size)
        #     is_fst_loss = True
        #
        #
        #
        #
        #
        #
        #     # 1.Discriminator
        #     optimizer_d_inlay.zero_grad()
        #     d_inlay_loss.backward(retain_graph=True)
        #     optimizer_d_inlay.step()
        #
        #     # 2.Generator
        #     optimizer.zero_grad()
        #     g_inlay_loss.backward(retain_graph=True)
        #     print(f'&&&&&&&& g_loss_grad:{model.st_gcns[0].gcn.conv.weight.grad} &&&&&&&&&&&&&&&&&&&&&&&&\n')
        #     gradd = model.st_gcns[0].gcn.conv.weight.grad
        #     # optimizer.step()
        #
        #     # 3.predictor
        #     # optimizer.zero_grad()
        #     pred_loss.backward()
        #     if args.clip_grad is not None:
        #         torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)
        #     print(f'&&&&&&&& pred_loss_grad:{model.st_gcns[0].gcn.conv.weight.grad-gradd} &&&&&&&&&&&&&&&&&&&&&&&&\n')
        #     optimizer.step()
        #
        #     #Metrics
        #     d_loss_batch += d_inlay_loss.item()
        #     g_loss_batch += g_inlay_loss.item()
        #     pred_loss_batch += pred_loss.item()
        #     print(f'TRAIN:\tsubset:{subset}\nEpoch:{epoch}\tDiscriminator_Loss:{d_loss_batch/batch_count}'
        #           f'Gernerator_Loss:{g_loss_batch/batch_count}Pred_Loss:{pred_loss_batch/batch_count}\t')

        # 1.Predictor
        pred_l = graph_loss(V_pred_s, V_tr_s) / args.batch_size
        optimizer.zero_grad()
        pred_l.backward()
        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)
        optimizer.step()

        # 2.Generator
        _ , _, hidden_s = model(V_obs_s, A_obs_s.squeeze())
        score_fake = d_inlay(hidden_s)
        g_inlay_l = gan_g_loss(score_fake, mode= 'mse')

        optimizer.zero_grad()
        g_inlay_l.backward()
        # print(f'&&&&&&&& g_loss_grad:{model.st_gcns[0].gcn.conv.weight.grad} &&&&&&&&&&&&&&&&&&&&&&&&\n')
        optimizer.step()

        # 3.Discriminator
        _ , _, hidden_s = model(V_obs_s, A_obs_s.squeeze())
        score_fake = d_inlay(hidden_s)
        _ , _, hidden_t = model(V_obs_t, A_obs_t.squeeze())
        score_real = d_inlay(hidden_t)
        d_loss_real, d_loss_fake = gan_d_loss(score_real, score_fake, mode= 'mse')
        d_inlay_l = d_loss_real + d_loss_fake

        optimizer_d_inlay.zero_grad()
        d_inlay_l.backward()
        optimizer_d_inlay.step()

        if batch_count%args.batch_size !=0 and index_batch != turn_point:
            if is_fst_loss:
                d_inlay_loss = d_inlay_l
                g_inlay_loss = g_inlay_l
                pred_loss = pred_l
                is_fst_loss = False

            else:
                d_inlay_loss += d_inlay_l
                g_inlay_loss += g_inlay_l
                pred_loss += pred_l

        else:
            is_fst_loss = True

            # Metrics
            d_loss_batch += d_inlay_loss.item()
            g_loss_batch += g_inlay_loss.item()
            pred_loss_batch += pred_loss.item()
            print(f'TRAIN:\tsubset:{subset}\nEpoch:{epoch}\tDiscriminator_Loss:{d_loss_batch/batch_count}'
                  f'Gernerator_Loss:{g_loss_batch/batch_count}Pred_Loss:{pred_loss_batch/batch_count}\t')



    metrics['d_loss'].append(d_loss_batch/batch_count)
    metrics['g_loss'].append(g_loss_batch / batch_count)
    metrics['pred_loss'].append(pred_loss_batch / batch_count)


def vald(epoch, subset):
    global metrics,target_loader,constant_metrics
    model.eval()
    d_inlay.eval()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(target_loader)
    turn_point =int(loader_len/args.batch_size)*args.batch_size+ loader_len%args.batch_size -1

    for cnt,batch in enumerate(target_loader):
        batch_count+=1

        #Get data
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask,V_obs,A_obs,V_tr,A_tr = batch


        V_obs_tmp =V_obs.permute(0,3,1,2)

        V_pred,_,_ = model(V_obs_tmp,A_obs.squeeze())

        V_pred = V_pred.permute(0,2,3,1)

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count%args.batch_size !=0 and cnt != turn_point :
            l = graph_loss(V_pred,V_tr)
            if is_fst_loss :
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss/args.batch_size
            is_fst_loss = True
            #Metrics
            loss_batch += loss.item()
            print('VALD:','\tsubset:',subset,'\t Epoch:', epoch,'\t Loss:',loss_batch/batch_count)

    metrics['val_loss'].append(loss_batch/batch_count)

    if  metrics['val_loss'][-1]< constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] =  metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(),checkpoint_dir+'val_best.pth')  # OK
        torch.save(d_inlay.state_dict(), checkpoint_dir+'d_inlay_val_best.pth')

if __name__ == '__main__':
    dada_datasets = [
        'A2B', 'A2C', 'A2D', 'A2E',
        'B2A', 'B2C', 'B2D', 'B2E',
        'C2A', 'C2B', 'C2D', 'C2E',
        'D2A', 'D2B', 'D2C', 'D2E',
        'E2A', 'E2B', 'E2C', 'E2D']
    # dada_datasets = ['A2B']
    # dada_datasets = ['C2A']

    device = torch.device('cuda', index=2)

    for subset in dada_datasets:
        start_time = time.time()
        args.dataset = subset
        print('*' * 30)
        print("Training initiating....")
        print(args)

        #Data prep
        obs_seq_len = args.obs_seq_len
        pred_seq_len = args.pred_seq_len
        data_set = '/data/lfh/DADA/datasets/'+args.dataset+'/'

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
        kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).to(device)

        d_inlay = Discriminator_inlay(args.output_size*args.obs_seq_len, args.mlp_dim).to(device)


        #Training settings
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        optimizer_d_inlay = optim.Adam(d_inlay.parameters(), lr=args.lr_d_inlay)

        if args.use_lrschd:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)
            scheduler_d_inlay = optim.lr_scheduler.StepLR(optimizer_d_inlay, step_size=args.lr_sh_rate, gamma=0.2)


        checkpoint_dir = './checkpoint_FLA/'+args.dataset+'/'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        with open(checkpoint_dir+'args.pkl', 'wb') as fp:
            pickle.dump(args, fp)

        print('Data and model loaded')
        print('Checkpoint dir:', checkpoint_dir)

        #Training
        metrics = {'pred_loss':[],  'val_loss':[], 'd_loss':[], 'g_loss':[]}  # 每个epoch存一下
        constant_metrics = {'min_val_epoch':-1, 'min_val_loss':9999999999999999}



        print('Training started ...')
        for epoch in range(args.num_epochs):
            train(epoch, subset)
            vald(epoch, subset)
            if args.use_lrschd:
                scheduler.step()


            print('*'*30)
            print('Epoch:',args.tag,":", epoch)
            for k,v in metrics.items():
                if len(v)>0:
                    print(k,v[-1])


            print(constant_metrics)
            print('*'*30)

            with open(checkpoint_dir+'metrics.pkl', 'wb') as fp:
                pickle.dump(metrics, fp)

            with open(checkpoint_dir+'constant_metrics.pkl', 'wb') as fp:
                pickle.dump(constant_metrics, fp)

        print(f"train_subset:{subset}\tuse_time_NObenchmark:{time.time()-start_time}")



