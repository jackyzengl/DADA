import argparse
import random
import numpy as np
import torch
import os
import importlib

from dataset import TrajectoryDataset
from torch.utils.data import DataLoader

from model import TrajectoryModel
from torch import optim
import torch.nn.functional as F

from utils import get_motion_modes

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
print(args)

# python train.py --dataset_name sdd --gpu 0 --hp_config config/sdd.py
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


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


if __name__ == "__main__":
    sum_ade, sum_fde = 0, 0
    datasets = ['A2B']

    for subset in datasets:
        args.dataset_name = subset
        args.hp_config = f'config/config_DADA/{subset}.py'
        spec = importlib.util.spec_from_file_location("hp_config", args.hp_config)
        hp_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hp_config)

        test_dataset = TrajectoryDataset(dataset_path=args.dataset_path, dataset_name=args.dataset_name,
                                         dataset_type='test', translation=True, rotation=True,
                                         scaling=False, obs_len=args.obs_len)

        test_loader = DataLoader(test_dataset, collate_fn=test_dataset.coll_fn, batch_size=hp_config.batch_size, shuffle=True,
                                 num_workers=args.num_works)

        motion_modes_file = f'./checkpoint/checkpoint_DADA/{args.dataset_name}_motion_modes.pkl'

        if os.path.exists(motion_modes_file):
            # print('motion modes loading ... ')
            import pickle
            f = open(motion_modes_file, 'rb+')
            motion_modes = pickle.load(f)
            f.close()
            motion_modes = torch.tensor(motion_modes, dtype=torch.float32).cuda()

        model = TrajectoryModel(in_size=2, obs_len=args.obs_len, pred_len=args.pred_len, embed_size=hp_config.model_hidden_dim,
                                enc_num_layers=2, int_num_layers_list=[1, 1], heads=4, forward_expansion=2)
        ckpt = torch.load(f'./checkpoint/checkpoint_DADA/{args.dataset_name}_best.pth')
        model.load_state_dict(ckpt)
        model = model.cuda()

        ade, fde, num_traj = test(model, test_loader, motion_modes)
        print(f'{subset}: ADE={ade}, FDE={fde}, num_traj={num_traj}')
        sum_ade += ade
        sum_fde += fde
    print(f'average ADE={sum_ade/len(datasets)}, average FDE={sum_fde/len(datasets)}')


