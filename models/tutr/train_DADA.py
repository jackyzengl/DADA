import argparse
import random
import numpy as np
import torch
import os
import importlib

from dataset import TrajectoryDataset
from torch.utils.data import DataLoader

from model import TrajectoryModel, Discriminator_inlay
from torch import optim
import torch.nn.functional as F

from utils import get_motion_modes, gan_g_loss, gan_d_loss

import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_path', type=str, default='./dataset/')
parser.add_argument('--dataset_name', type=str, default='A2B')
# TODO: modify dataset_path and dataset_name
parser.add_argument("--hp_config", type=str, default='config/hotel.py', help='hyper-parameter')
parser.add_argument('--lr_scaling', action='store_true', default=False)
parser.add_argument('--num_works', type=int, default=8)
parser.add_argument('--obs_len', type=int, default=8)
parser.add_argument('--pred_len', type=int, default=12)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--data_scaling', type=list, default=[1.9, 0.4])
parser.add_argument('--dist_threshold', type=float, default=2)
parser.add_argument('--checkpoint', type=str, default='./checkpoint/')

args = parser.parse_args()
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# python train.py --dataset_name sdd --gpu 0 --hp_config config/sdd.py
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def get_cls_label(gt, motion_modes, soft_label=True):
    # motion_modes [K pred_len 2]
    # gt [B pred_len 2]

    gt = gt.reshape(gt.shape[0], -1).unsqueeze(1)  # [B 1 pred_len*2]
    motion_modes = motion_modes.reshape(motion_modes.shape[0], -1).unsqueeze(0)  # [1 K pred_len*2]
    distance = torch.norm(gt - motion_modes, dim=-1)  # [B K]
    soft_label = F.softmax(-distance, dim=-1)  # [B K]
    closest_mode_indices = torch.argmin(distance, dim=-1)  # [B]

    return soft_label, closest_mode_indices


def train(epoch, model, d_inlay, reg_criterion, cls_criterion, optimizer, optimizer_d_inlay,
          train_dataloader, val_dataloader, motion_modes_s, motion_modes_t):
    model.train()
    d_inlay.train()
    total_loss = []
    total_d_loss = []
    total_g_loss = []

    len_source, len_target = len(train_dataloader), len(val_dataloader)
    len_max = max(len_source, len_target)
    len_min = min(len_source, len_target)
    source_iter = iter(train_dataloader)
    target_iter = iter(val_dataloader)

    for index_batch in range(len_max):
        if index_batch % len_source == 0:
            del source_iter
            source_iter = iter(train_dataloader)
            # print(f'*** Source Iter is reset{index_batch} ***')
        if index_batch % len_target == 0:
            del target_iter
            target_iter = iter(val_dataloader)
            # print(f'*** Target Iter is reset{index_batch} ***')
        (ped_s, neis_s, mask_s), (ped_t, neis_t, mask_t) = next(source_iter), next(target_iter)
        ped_s = ped_s.cuda()
        neis_s = neis_s.cuda()
        mask_s = mask_s.cuda()
        ped_t = ped_t.cuda()
        neis_t = neis_t.cuda()
        mask_t = mask_t.cuda()

        if args.dataset_name[2] == 'A':
            ped_s[:, :, 0] = ped_s[:, :, 0] * args.data_scaling[0]
            ped_s[:, :, 1] = ped_s[:, :, 1] * args.data_scaling[1]

        scale = torch.randn(ped_s.shape[0]) * 0.05 + 1
        scale = scale.cuda()
        scale = scale.reshape(ped_s.shape[0], 1, 1)
        ped_s = ped_s * scale
        scale = scale.reshape(ped_s.shape[0], 1, 1, 1)
        neis_s = neis_s * scale

        scale = torch.randn(ped_t.shape[0]) * 0.05 + 1
        scale = scale.cuda()
        scale = scale.reshape(ped_t.shape[0], 1, 1)
        ped_t = ped_t * scale
        scale = scale.reshape(ped_t.shape[0], 1, 1, 1)
        neis_t = neis_t * scale

        ped_obs_s = ped_s[:, :args.obs_len]
        gt_s = ped_s[:, args.obs_len:]
        neis_obs_s = neis_s[:, :, :args.obs_len]

        ped_obs_t = ped_t[:, :args.obs_len]
        gt_t = ped_t[:, args.obs_len:]
        neis_obs_t = neis_t[:, :, :args.obs_len]


        with torch.no_grad():
            soft_label_s, closest_mode_indices_s = get_cls_label(gt_s, motion_modes_s)
            soft_label_t, closest_mode_indices_t = get_cls_label(gt_t, motion_modes_t)

        # 1. Predictor
        optimizer.zero_grad()
        pred_traj, scores, _ = model(ped_obs_s, neis_obs_s, motion_modes_s, mask_s, closest_mode_indices_s)
        reg_label = gt_s.reshape(pred_traj.shape)
        reg_loss = reg_criterion(pred_traj, reg_label)
        clf_loss = cls_criterion(scores.squeeze(), soft_label_s)
        loss = reg_loss + clf_loss
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

        # 2. Discriminator
        _, _, hidden_s = model(ped_obs_s, neis_obs_s, motion_modes_s, mask_s, closest_mode_indices_s)
        score_fake = d_inlay(hidden_s)
        _, _, hidden_t = model(ped_obs_t, neis_obs_t, motion_modes_t, mask_t, closest_mode_indices_t)
        score_real = d_inlay(hidden_t)
        d_inlay_loss = gan_d_loss(score_real, score_fake) * 0.002

        optimizer_d_inlay.zero_grad()
        d_inlay_loss.backward()
        optimizer_d_inlay.step()
        total_d_loss.append(d_inlay_loss.item())

        # 3. Generator
        _, _, hidden_s = model(ped_obs_s, neis_obs_s, motion_modes_s, mask_s, closest_mode_indices_s)
        score_fake = d_inlay(hidden_s)
        g_inlay_loss = gan_g_loss(score_fake) * 0.002
        optimizer.zero_grad()
        g_inlay_loss.backward()
        optimizer.step()
        total_g_loss.append(g_inlay_loss.item())

    return total_loss, total_d_loss, total_g_loss


def vis_predicted_trajectories(obs_traj, gt, pred_trajs, pred_probabilities, min_index):
    # obs_traj [B T_obs 2]
    # gt [B T_pred 2]
    # pred_trajs [B 20 T_pred 2]
    # pred_probabilities [B 20]

    for i in range(obs_traj.shape[0]):
        plt.clf()
        curr_obs = obs_traj[i].cpu().numpy()  # [T_obs 2]
        curr_gt = gt[i].cpu().numpy()
        curr_preds = pred_trajs[i].cpu().numpy()

        curr_pros = pred_probabilities[i].cpu().numpy()
        curr_min_index = min_index[i].cpu().numpy()
        obs_x = curr_obs[:, 0]
        obs_y = curr_obs[:, 1]
        gt_x = np.concatenate((obs_x[-1:], curr_gt[:, 0]))
        gt_y = np.concatenate((obs_y[-1:], curr_gt[:, 1]))
        plt.plot(obs_x, obs_y, marker='o', color='green')
        plt.plot(gt_x, gt_y, marker='o', color='blue')
        plt.scatter(gt_x[-1], gt_y[-1], marker='*', color='blue', s=300)

        for j in range(curr_preds.shape[0]):

            pred_x = np.concatenate((obs_x[-1:], curr_preds[j][:, 0]))
            pred_y = np.concatenate((obs_y[-1:], curr_preds[j][:, 1]))
            if j == curr_min_index:
                plt.plot(pred_x, pred_y, ls='-.', lw=2.0, color='red')
                plt.scatter(pred_x[-1], pred_y[-1], marker='*', color='orange', s=300)
            else:
                plt.plot(pred_x, pred_y, ls='-.', lw=0.5, color='red')
                plt.scatter(pred_x[-1], pred_y[-1], marker='*', color='red', s=300)
            plt.text(pred_x[-1], pred_y[-1], str("%.2f" % curr_pros[j]), ha='center')

        plt.tight_layout()
        save_path = './fig/' + args.dataset_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + '/' + str(time.time()) + '.png')

    return


def test(model, test_dataloader, motion_modes):
    model.eval()
    ade = 0
    fde = 0
    num_traj = 0

    for (ped, neis, mask) in test_dataloader:
        ped = ped.cuda()
        neis = neis.cuda()
        mask = mask.cuda()

        ped_obs = ped[:, :args.obs_len]
        gt = ped[:, args.obs_len:]
        neis_obs = neis[:, :, :args.obs_len]

        with torch.no_grad():
            num_traj += ped_obs.shape[0]
            pred_trajs, scores = model(ped_obs, neis_obs, motion_modes, mask, None, test=True)
            # top_k_scores = torch.topk(scores, k=20, dim=-1).values
            # top_k_scores = F.softmax(top_k_scores, dim=-1)
            pred_trajs = pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[1], 2)
            gt_ = gt.unsqueeze(1)
            norm_ = torch.norm(pred_trajs - gt_, p=2, dim=-1)
            ade_ = torch.mean(norm_, dim=-1)
            fde_ = norm_[:, :, -1]
            min_ade, min_ade_index = torch.min(ade_, dim=-1)
            min_fde, min_fde_index = torch.min(fde_, dim=-1)
            # vis_predicted_trajectories(ped_obs, gt, pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[-2], -1), top_k_scores,
            #                             min_fde_index)

            # b-ade/fde
            # batch_index = torch.LongTensor(range(top_k_scores.shape[0])).cuda()
            # min_ade_p = top_k_scores[batch_index, min_ade_index]
            # min_fde_p = top_k_scores[batch_index, min_fde_index]
            # min_ade = min_ade + (1 - min_ade_p)**2
            # min_fde = min_fde + (1 - min_fde_p)**2

            min_ade = torch.sum(min_ade)
            min_fde = torch.sum(min_fde)
            ade += min_ade.item()
            fde += min_fde.item()

    ade = ade / num_traj
    fde = fde / num_traj
    return ade, fde, num_traj


def plot_loss(train_loss, d_loss, g_loss, save_path):
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(train_loss)
    ax[1].plot(d_loss)
    ax[1].plot(g_loss)

    ax[0].legend(['train_loss'])
    ax[1].legend(['d_loss', 'g_loss'])
    plt.savefig(save_path)



if __name__ == "__main__":
    sum_ade, sum_fde = 0, 0
    datasets = ['A2B']
    for subset in datasets:
        args.dataset_name = subset
        args.hp_config = f'config/config_DADA/{subset}.py'

        spec = importlib.util.spec_from_file_location("hp_config", args.hp_config)
        hp_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hp_config)

        train_dataset = TrajectoryDataset(dataset_path=args.dataset_path, dataset_name=args.dataset_name,
                                          dataset_type='train', translation=True, rotation=True,
                                          scaling=True, obs_len=args.obs_len,
                                          dist_threshold=hp_config.dist_threshold, smooth=False)

        val_dataset = TrajectoryDataset(dataset_path=args.dataset_path, dataset_name=args.dataset_name,
                                        dataset_type='test', translation=True, rotation=True,
                                        scaling=False, obs_len=args.obs_len)

        motion_modes_file_s = f'./checkpoint/checkpoint_DADA/{args.dataset_name}_motion_modes.pkl'
        motion_modes_file_t = f'./checkpoint/checkpoint_DADA/{args.dataset_name}_motion_modes_t.pkl'

        print('motion modes generating ... ')
        motion_modes_s = get_motion_modes(train_dataset, args.obs_len, args.pred_len, hp_config.n_clusters,
                                          args.dataset_path, args.dataset_name,
                                          smooth_size=hp_config.smooth_size,
                                          random_rotation=hp_config.random_rotation, traj_seg=hp_config.traj_seg,
                                          motion_modes_file=motion_modes_file_s)
        motion_modes_t = get_motion_modes(val_dataset, args.obs_len, args.pred_len, hp_config.n_clusters,
                                          args.dataset_path, args.dataset_name,
                                          smooth_size=hp_config.smooth_size,
                                          random_rotation=hp_config.random_rotation, traj_seg=hp_config.traj_seg,
                                          motion_modes_file=motion_modes_file_t)
        motion_modes_s = torch.tensor(motion_modes_s, dtype=torch.float32).cuda()
        motion_modes_t = torch.tensor(motion_modes_t, dtype=torch.float32).cuda()

        train_loader = DataLoader(train_dataset, collate_fn=train_dataset.coll_fn, batch_size=hp_config.batch_size,
                                  shuffle=True, num_workers=args.num_works)
        val_loader = DataLoader(val_dataset, collate_fn=val_dataset.coll_fn, batch_size=hp_config.batch_size,
                                shuffle=True, num_workers=args.num_works)

        model = TrajectoryModel(in_size=2, obs_len=args.obs_len, pred_len=args.pred_len,
                                embed_size=hp_config.model_hidden_dim,
                                enc_num_layers=2, int_num_layers_list=[1, 1], heads=4, forward_expansion=2)
        model = model.cuda()

        d_inlay = Discriminator_inlay(input_dim=hp_config.model_hidden_dim,
                                      mlp_dim=2 * hp_config.model_hidden_dim)
        d_inlay = d_inlay.cuda()

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hp_config.lr)
        optimizer_d_inlay = optim.Adam(d_inlay.parameters(), lr=hp_config.lr)

        reg_criterion = torch.nn.SmoothL1Loss().cuda()
        cls_criterion = torch.nn.CrossEntropyLoss().cuda()

        if args.lr_scaling:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[270, 400], gamma=0.5)

        min_ade = 99
        min_fde = 99

        all_pred_loss = []
        all_d_loss = []
        all_g_loss = []

        for ep in range(hp_config.epoch):

            total_loss, total_d_loss, total_g_loss = train(ep, model, d_inlay, reg_criterion, cls_criterion,
                                optimizer, optimizer_d_inlay, train_loader, val_loader, motion_modes_s, motion_modes_t)
            ade, fde, num_traj = test(model, val_loader, motion_modes_s)
            if args.lr_scaling:
                scheduler.step()

            # if not os.path.exists(args.checkpoint + args.dataset_name):
            #     os.makedirs(args.checkpoint + args.dataset_name)

            # if min_fde + min_ade > ade + fde:
            if min_ade > ade:
                min_fde = fde
                min_ade = ade
                min_fde_epoch = ep
                torch.save(model.state_dict(), f'./checkpoint/checkpoint_DADA/{args.dataset_name}_best.pth')  # OK

            # pred_loss = sum(total_loss) / len(total_loss)
            # all_pred_loss.append(pred_loss)
            # d_loss = sum(total_d_loss) / len(total_d_loss)
            # all_d_loss.append(d_loss)
            # g_loss = sum(total_g_loss) / len(total_g_loss)
            # all_g_loss.append(g_loss)

            print('subset:', subset, 'epoch:', ep, 'ade:', ade, 'fde:', fde, 'min_ade:', min_ade, 'min_fde:', min_fde, min_fde_epoch)
        plot_loss(all_pred_loss, all_d_loss, all_g_loss, f'./checkpoint/checkpoint_DADA/{args.dataset_name}.png')
