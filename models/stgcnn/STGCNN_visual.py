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
from model import social_stgcnn
import copy

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D


import debugpy
try:
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

H_mat = np.array([[2.8128700e-02,2.0091900e-03,-4.6693600e+00],
                  [8.0625700e-04,2.5195500e-02,5.0608800e+00],
                  [3.4555400e-04,9.2512200e-05,4.6255300e-01]])

x_list, y_list = [], []

def world_to_pixel(world_points, H):
    points_dim = world_points.ndim
    if points_dim == 2:
        world_points = world_points[np.newaxis, :]    # world_points: [KSTEPS, T, 2]

    world_points_list = []
    for points in world_points:
        points = np.array(points, dtype=float)
        
        n = points.shape[0]
        homogeneous = np.column_stack((points, np.ones(n)))
        
        transformed = (H @ homogeneous.T).T
        
        pixel_points = transformed[:, :2] / transformed[:, 2, np.newaxis]
        world_points_list.append(pixel_points)
    
    return world_points_list[0] if points_dim == 2 else np.stack(world_points_list)

def plot_trajectories(ax,
                      prediction_dict,  # [N, 20, 12, 2]
                      histories_dict,  # [N, 8, 2]
                      futures_dict,  # [N, 12, 2]
                      ):  
    global x_list, y_list
    cmap1 = ['Purples', 'Blues', 'Oranges', 'Greens', 'Reds', 'Greys']
    cmap2 = ['purple', 'blue', 'orange', 'green', 'red', 'grey']
    t = 0
    la = 6
    
    for node in range(histories_dict.shape[0]):
        history = world_to_pixel(histories_dict[node], np.linalg.inv(H_mat))
        future = world_to_pixel(futures_dict[node], np.linalg.inv(H_mat))
        predictions = world_to_pixel(prediction_dict[node], np.linalg.inv(H_mat))

        x_list.append(history[..., 0])
        y_list.append(history[..., 1])
    
        if np.isnan(history[-1]).any():
            continue
    
        traj_data_reshape = predictions.reshape(-1, 2)  # [20*12, 2]  # [k, 2]
        pc = sns.kdeplot(x=traj_data_reshape[:, 0], y=traj_data_reshape[:, 1],
                    ax=ax, fill=True, thresh=0.2, cmap=cmap1[t%la], alpha=0.9, label='Prediction', legend=True)
        t += 1
    
    t = 0
    for node in range(histories_dict.shape[0]):
        history = world_to_pixel(histories_dict[node], np.linalg.inv(H_mat))
        future = world_to_pixel(futures_dict[node], np.linalg.inv(H_mat))
        predictions = world_to_pixel(prediction_dict[node], np.linalg.inv(H_mat))
        
        if np.isnan(history[-1]).any():
            continue
        
        ax.plot(history[:, 0], history[:, 1], linestyle='--', color=cmap2[t%la], label='Observed')
        ax.scatter(history[-1, 0], history[-1, 1], c=cmap2[t%la], s=10)

        future = np.vstack((history[-1], future))
        ax.plot(future[:, 0], future[:, 1], linestyle='-', color=cmap2[t%la], label='Ground Truth')
        # path_effects=[pe.Stroke(linewidth=edge_width, foreground='g'), pe.Normal()]
        
        dx, dy = future[-1, 0] - future[-2, 0], future[-1, 1] - future[-2, 1]
        ax.arrow(future[-2, 0], future[-2, 1], dx, dy, head_width=0.3, fc=cmap2[t%la], ec=cmap2[t%la])
        
        t += 1
        
    ax.axis('equal')


def visual(KSTEPS=1):
    global loader_test,model
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step =0 
    for batch in loader_test:
        print(f"batch: {step}/{len(loader_test)}")
        step+=1
        #Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask,V_obs,A_obs,V_tr,A_tr = batch


        num_of_objs = obs_traj_rel.shape[1]

        #Forward
        #V_obs = batch,seq,node,feat
        #V_obs_tmp = batch,feat,seq,node
        V_obs_tmp =V_obs.permute(0,3,1,2)

        V_pred,_ = model(V_obs_tmp,A_obs.squeeze())
        # print(V_pred.shape)
        # torch.Size([1, 5, 12, 2])
        # torch.Size([12, 2, 5])
        V_pred = V_pred.permute(0,2,3,1)
        # torch.Size([1, 12, 2, 5])>>seq,node,feat
        # V_pred= torch.rand_like(V_tr).cuda()


        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred,V_tr =  V_pred[:,:num_of_objs,:],V_tr[:,:num_of_objs,:]
        #print(V_pred.shape)

        #For now I have my bi-variate parameters 
        #normx =  V_pred[:,:,0:1]
        #normy =  V_pred[:,:,1:2]
        sx = torch.exp(V_pred[:,:,2]) #sx
        sy = torch.exp(V_pred[:,:,3]) #sy
        corr = torch.tanh(V_pred[:,:,4]) #corr
        
        cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2).cuda()
        cov[:,:,0,0]= sx*sx
        cov[:,:,0,1]= corr*sx*sy
        cov[:,:,1,0]= corr*sx*sy
        cov[:,:,1,1]= sy*sy
        mean = V_pred[:,:,0:2]
        
        mvnormal = torchdist.MultivariateNormal(mean,cov)


        ### Rel to abs 
        ##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len 
        
        #Now sample 20 samples
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                V_x[0,:,:].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                V_x[-1,:,:].copy())
        
        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        for n in range(num_of_objs):
            ade_ls[n]=[]
            fde_ls[n]=[]

        for k in range(KSTEPS):

            V_pred = mvnormal.sample()

            #V_pred = seq_to_nodes(pred_traj_gt.data.numpy().copy())
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                       V_x[-1,:,:].copy())  # [12,2,2]
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))  # [20,12,n,2]
        
        
        prediction_dict = np.stack(raw_data_dict[step]['pred'],axis=0).transpose(2,0,1,3)  # [n, 20, 12, 2]
        histories_dict = raw_data_dict[step]['obs'].transpose(1,0,2)  
        futures_dict = raw_data_dict[step]['trgt'].transpose(1,0,2)
        
        fig, ax = plt.subplots()
        plot_trajectories(ax,
                      prediction_dict,  # [N, 20, 12, 2]
                      histories_dict,  # [N, 8, 2]
                      futures_dict,  # [N, 12, 2]
                      )
        ax.set_xlim([-320, 320])
        ax.set_ylim([-480, 0])
        ax.set_aspect(1)
        # plt.legend()
        #ax.set_title(f"{scene.name}-t: {timestep}")
        save_path = './traj_visual/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f"{step}.png"), bbox_inches='tight', transparent=False, dpi=300)
        plt.close(fig)

paths = ['./checkpoint/DADA/A2B']
KSTEPS=20

print("*"*50)
print('Number of samples:',KSTEPS)
print("*"*50)


for feta in range(len(paths)):
    ade_ls = [] 
    fde_ls = [] 
    path = paths[feta]
    exps = glob.glob(path)
    print('Model being tested are:',exps)

    for exp_path in exps:
        print("*"*50)
        print("Evaluating model:",exp_path)

        model_path = exp_path+'/val_best.pth'
        args_path = exp_path+'/args.pkl'
        with open(args_path,'rb') as f: 
            args = pickle.load(f)

        stats= exp_path+'/constant_metrics.pkl'
        with open(stats,'rb') as f: 
            cm = pickle.load(f)
        print("Stats:",cm)



        #Data prep     
        obs_seq_len = args.obs_seq_len
        pred_seq_len = args.pred_seq_len
        data_set = '../../datasets/'+args.dataset+'/'

        dset_test = TrajectoryDataset(
                data_set+'test/',
                obs_len=obs_seq_len,
                pred_len=pred_seq_len,
                skip=1,norm_lap_matr=True)

        loader_test = DataLoader(
                dset_test,
                batch_size=1,#This is irrelative to the args batch size parameter
                shuffle =False,
                num_workers=1)



        #Defining the model 
        model = social_stgcnn(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
        output_feat=args.output_size,seq_len=args.obs_seq_len,
        kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).cuda()
        model.load_state_dict(torch.load(model_path))


        ade_ =999999
        fde_ =999999
        print("Testing ....")
        visual()
print("*"*50)