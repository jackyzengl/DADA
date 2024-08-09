import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from enum import Enum
import functools
import numpy as np
import math


class ModeKeys(Enum):
    TRAIN = 1
    EVAL = 2
    PREDICT = 3


def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3, decay=1.):
    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize) * decay**it

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x))

    return lr_lambda


def to_one_hot(labels, n_labels):
    return torch.eye(n_labels, device=labels.device)[labels]


def exp_anneal(anneal_kws):
    device = anneal_kws['device']
    start = torch.tensor(anneal_kws['start'], device=device)
    finish = torch.tensor(anneal_kws['finish'], device=device)
    rate = torch.tensor(anneal_kws['rate'], device=device)
    return lambda step: finish - (finish - start)*torch.pow(rate, torch.tensor(step, dtype=torch.float, device=device))


def sigmoid_anneal(anneal_kws):
    device = anneal_kws['device']
    start = torch.tensor(anneal_kws['start'], device=device)
    finish = torch.tensor(anneal_kws['finish'], device=device)
    center_step = torch.tensor(anneal_kws['center_step'], device=device, dtype=torch.float)
    steps_lo_to_hi = torch.tensor(anneal_kws['steps_lo_to_hi'], device=device, dtype=torch.float)
    return lambda step: start + (finish - start)*torch.sigmoid((torch.tensor(float(step), device=device) - center_step) * (1./steps_lo_to_hi))


class CustomLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        super(CustomLR, self).__init__(optimizer, lr_lambda, last_epoch)

    def get_lr(self):
        return [lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]


def mutual_inf_mc(x_dist):
    dist = x_dist.__class__
    H_y = dist(probs=x_dist.probs.mean(dim=0)).entropy()
    return (H_y - x_dist.entropy().mean(dim=0)).sum()


def run_lstm_on_variable_length_seqs(lstm_module, original_seqs, lower_indices=None, upper_indices=None, total_length=None):
    bs, tf = original_seqs.shape[:2]
    if lower_indices is None:
        lower_indices = torch.zeros(bs, dtype=torch.int)
    if upper_indices is None:
        upper_indices = torch.ones(bs, dtype=torch.int) * (tf - 1)
    if total_length is None:
        total_length = max(upper_indices) + 1
    # This is done so that we can just pass in self.prediction_timesteps
    # (which we want to INCLUDE, so this will exclude the next timestep).
    inclusive_break_indices = upper_indices + 1

    pad_list = list()
    for i, seq_len in enumerate(inclusive_break_indices):
        pad_list.append(original_seqs[i, lower_indices[i]:seq_len])

    packed_seqs = rnn.pack_sequence(pad_list, enforce_sorted=False)
    packed_output, (h_n, c_n) = lstm_module(packed_seqs)
    output, _ = rnn.pad_packed_sequence(packed_output,
                                        batch_first=True,
                                        total_length=total_length)

    return output, (h_n, c_n)


def extract_subtensor_per_batch_element(tensor, indices):
    batch_idxs = torch.arange(start=0, end=len(indices))

    batch_idxs = batch_idxs[~torch.isnan(indices)]
    indices = indices[~torch.isnan(indices)]
    if indices.size == 0:
        return None
    else:
        indices = indices.long()
    if tensor.is_cuda:
        batch_idxs = batch_idxs.to(tensor.get_device())
        indices = indices.to(tensor.get_device())
    return tensor[batch_idxs, indices]


def unpack_RNN_state(state_tuple):
    # PyTorch returned LSTM states have 3 dims:
    # (num_layers * num_directions, batch, hidden_size)

    state = torch.cat(state_tuple, dim=0).permute(1, 0, 2)
    # Now state is (batch, 2 * num_layers * num_directions, hidden_size)

    state_size = state.size()
    return torch.reshape(state, (-1, state_size[1] * state_size[2]))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# using wonder's beautiful simplification:
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def relative_to_abs(rel_traj,  # [nums, 20, 2]
                    start_pos):  # [nums, 2]
    displacement = torch.cumsum(rel_traj, dim=1)  # [nums, 20, 2]
    start_pos = torch.unsqueeze(start_pos, dim=1)  # [nums, 1, 2]
    abs_traj = displacement + start_pos  # [nums, 20, 2]
    return abs_traj  # [nums, 20, 2]

def abs_to_relative(abs_traj):  # [nums, 20, 2]
    rel_traj = torch.zeros_like(abs_traj)  # [nums, 20, 2]
    rel_traj[:, 1:, :] = abs_traj[:, 1:, :] - abs_traj[:, :-1, :]
    return rel_traj  # [nums, 20, 2]


def calculate_dx(x,  # [nums, t, 2]
                 dt=0.4):
    dx = torch.zeros_like(x)  # [nums, t, 2]
    length = dx.shape[1]  # t
    dx[:, 0, :] = (x[:, 1, :] - x[:, 0, :]) / dt
    for i in range(1, length):
        dx[:, i, :] = (x[:, i, :] - x[:, i-1, :]) / dt
    return dx  # [nums, t, 2]


def edge_to_abs_traj(neighbors_data_st_s,  # [256, N, 8, 6]
                     x_s):  # [256, 8, 6]
    abs_trajs = []
    nums = len(neighbors_data_st_s)
    for i in range(nums):
        abs_traj_i = []
        nums_neighbors = len(neighbors_data_st_s[i])
        for j in range(nums_neighbors):
            abs_traj_i_j = neighbors_data_st_s[i, j, :2] * 1 + x_s[i, -1, :2]
            abs_traj_i.append(abs_traj_i_j)
        abs_trajs.append(abs_traj_i)
    return abs_trajs

def standardize(x,  # [nums, 20, 2]
                kind='pos'):
    if kind == 'pos':
        mean = x[:, -1, :].unsqueeze(dim=1)  # [nums, 1, 2]
        std = 3
    elif kind == 'vel':
        mean, std = 0, 2
    elif kind == 'acc':
        mean, std = 0, 1
    return (x - mean) / std  # [nums, 20, 2]

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def pre_pad(x,  # [256, 8, 6]
            first_history_index):
    x_ = torch.zeros_like(x)
    nums = x.shape[0]
    for i in range(nums):
        cur_index = int(first_history_index[i])
        x_[i, :cur_index, :] = x[i, cur_index, :].repeat(1, cur_index, 1)
        x_[i, cur_index:, :] = x[i, cur_index:, :]
    return x_  # [256, 8, 6]

def reverse_pad(x,  # [256, 8, 6]
                first_history_index):
    x_ = torch.zeros_like(x)
    dims = x.shape[2]
    pad_tensor = torch.Tensor([np.nan]*dims)  # [dims=6]
    nums = x.shape[0]
    for i in range(nums):
        cur_index = int(first_history_index[i])
        x_[i, :cur_index, :] = pad_tensor.repeat(1, cur_index, 1)  # [1, cur_index, dims=6]
        x_[i, cur_index:, :] = x[i, cur_index:, :]
    return x_  # [256, 8, 6]
