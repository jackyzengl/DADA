import argparse
import os
import sys
import torch
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from attrdict import AttrDict

sys.path.append("..")
from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def get_HighDimTensor(args, set_type='train'):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = f'../../../datasets/{args.dataset_name}/{set_type}'
        _, loader = data_loader(_args, path)
        all_hidden = []
        with torch.no_grad():
            for batch in loader:
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
                 non_linear_ped, loss_mask, seq_start_end) = batch

                hidden = generator.get_hidden_state(obs_traj, obs_traj_rel, seq_start_end)
                all_hidden.append(hidden)
        all_hidden = torch.cat(all_hidden, dim=0)
        return all_hidden.detach().cpu().numpy()


def visual(feat):  # [num, dim=32]
    ts = TSNE(n_components=2, random_state=0)
    # ts = PCA(n_components=2)
    x_ts = ts.fit_transform(feat)
    # x_min, x_max = x_ts.min(0), x_ts.max(0)
    # x_final = (x_ts - x_min) / (x_max - x_min)
    x_final = x_ts
    return x_final


if __name__ == '__main__':
    dada_datasets = ['A2B']
    for subset in dada_datasets:
        print(subset)
        args = parser.parse_args()
        args.dataset_name = subset
        args.model_path = f'../checkpoint/checkpoint_DADA/{subset}_with_model.pt'
        torch.manual_seed(72)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        High_Dim_s_real = get_HighDimTensor(args, 'train_origin')
        High_Dim_s_fake = get_HighDimTensor(args, 'train_fake')
        High_Dim_t = get_HighDimTensor(args, 'val')

        LowDim_s_real, LowDim_s_fake, LowDim_t = visual(High_Dim_s_real), visual(High_Dim_s_fake), visual(High_Dim_t)
        len_min = min(LowDim_s_real.shape[0], LowDim_s_fake.shape[0], LowDim_t.shape[0])
        split_s_real = LowDim_s_real.shape[0] // len_min
        split_s_fake = LowDim_s_fake.shape[0] // len_min
        split_t = LowDim_t.shape[0] // len_min

        fig, ax = plt.subplots()
        ax.scatter(LowDim_s_real[:, 0], LowDim_s_real[:, 1], marker='.', c='#E36414', s=35, label='Source')
        # ax.scatter(LowDim_s_fake[:, 0], LowDim_s_fake[:, 1], marker='.', c='#01FF07', s=16, label='DLA')
        ax.scatter(LowDim_t[:, 0], LowDim_t[:, 1], marker='.', c='#3468C0', s=35, label='Target')

        # plt.title(subset, fontsize=12, fontweight='normal')
        ax.tick_params(axis='both', direction='in')
        plt.xticks([])
        plt.yticks([])
        # ax.legend()
        plt.savefig(f'../DomainDiff_visualization/DADA/{subset}.png', bbox_inches='tight', dpi=300)
