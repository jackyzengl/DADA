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


def ade(predAll,targetAll,count_):
    """
    predAll: [[pred_len, 1, 2]]
    targetAll: [[pred_len, 1, 2]]
    count_: [1]
    """
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)  # [1, pred_len, 2]
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N*T)
        
    return sum_all/All


def fde(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T-1,T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N)

    return sum_all/All


def seq_to_nodes(seq_):
    # seq_: [1,N,2,obs_len]
    max_nodes = seq_.shape[1] #number of pedestrians in the graph
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]  # [N,2,obs_len]
    
    V = np.zeros((seq_len,max_nodes,2))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_[h]
            
    return V.squeeze()  # [seq_len, N, 2]

def nodes_rel_to_nodes_abs(nodes,init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s,ped,:] = np.sum(nodes[:s+1,ped,:],axis=0) + init_node[ped,:]

    return nodes_.squeeze()

def closer_to_zero(current,new_v):
    dec =  min([(abs(current),current),(abs(new_v),new_v)])[1]
    if dec != current:
        return True
    else: 
        return False
        
def bivariate_loss(V_pred,V_trgt):
    """
    V_pred: [pred_len, N, feature_len]
    V_grgt: [pred_len, N, 2]
    """
    #mux, muy, sx, sy, corr
    #assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:,:,0]- V_pred[:,:,0]
    normy = V_trgt[:,:,1]- V_pred[:,:,1]

    sx = torch.exp(V_pred[:,:,2]) #sx
    sy = torch.exp(V_pred[:,:,3]) #sy
    corr = torch.tanh(V_pred[:,:,4]) #corr
    
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)
    
    return result


# loss of G
def gan_g_loss(scores_fake,  # [nums, 1]
               mode='mse'):
    if mode == 'bce':
        loss_func = nn.BCEWithLogitsLoss()
    elif mode == 'mse':
        loss_func = nn.MSELoss()

    y_real = torch.ones_like(scores_fake)  # * random.uniform(0.7, 1.2)
    return loss_func(scores_fake, y_real)  # make the fake more real, [1]


# loss of D
def gan_d_loss(scores_real,  # [nums, 1]
               scores_fake,  # [nums, 1]
               mode='mse'):
    if mode == 'bce':
        loss_func = nn.BCEWithLogitsLoss()
    elif mode == 'mse':
        loss_func = nn.MSELoss()

    y_real = torch.ones_like(scores_real)  # * random.uniform(0.7, 1.2)  # [nums] ~ U(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake)  # * random.uniform(0, 0.3)  # [nums] = [0.0, 0.0, ..., 0.0]

    loss_real = loss_func(scores_real, y_real)  # make the real more real
    loss_fake = loss_func(scores_fake, y_fake)  # make the fake faker

    return loss_real, loss_fake  # [1], [1]
