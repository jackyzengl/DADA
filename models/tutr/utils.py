import numpy as np
from sklearn.cluster import KMeans
import os
import pickle

import torch
import random
import torch.nn as nn


def translation(seq, refer_point_index):  # rigid transformation
    # seq [N T 2]
    return seq - seq[:, refer_point_index:refer_point_index + 1]


def get_rot_mats(thea_list):
    num = thea_list.shape[0]
    rot_mat_list = []
    for i in range(num):
        thea = thea_list[i]
        rot_mat_list.append(np.array([[np.cos(thea), -np.sin(thea)],
                                      [np.sin(thea), np.cos(thea)]]))
    return np.stack(rot_mat_list, axis=0)


def rotation(seq, refer_point_index=0):  # rigid transformation

    # seq [N T 2]
    angle = np.arctan2(seq[:, refer_point_index, 1], seq[:, refer_point_index, 0])
    rot_mat = get_rot_mats(angle)
    rot_seq = np.matmul(seq, rot_mat)

    return rot_seq, rot_mat


def simple_moving_average(input, windows_size):
    # input [N T 2]

    x_len = input.shape[1]
    x_list = []
    keep_num = int(np.floor(windows_size / 2))
    for i in range(windows_size):
        x_list.append(input[:, i:x_len - windows_size + 1 + i])
    x = sum(x_list) / windows_size
    x = np.concatenate((input[:, :keep_num], x, input[:, -keep_num:]), axis=1)
    return x


def dy_random_rotation(seqs):
    # seqs [N T 2]
    random_angle = -1 + 2 * np.random.rand(seqs.shape[0])
    random_angle = np.arcsin(random_angle)
    rot_mat = get_rot_mats(random_angle)
    rot_seq = np.matmul(seqs, rot_mat)
    return rot_seq


def kmeans_(seq, n_clusters=100):
    # seq [N T 2] T=pred_len

    input_data = seq.reshape(seq.shape[0], -1)
    # input_data = seq[:, -1] # destination
    clf = KMeans(n_clusters=n_clusters,
                 random_state=1
                 ).fit(input_data)

    centers = clf.cluster_centers_
    centers = centers.reshape(centers.shape[0], -1, 2)

    return centers


def trajectory_motion_modes(all_trajs, obs_len, n_units=120, smooth_size=3, random_rotation=False):
    # full_ego_trajs [B T 2]

    clustering_input = all_trajs[:, obs_len:]
    if smooth_size is not None:
        clustering_input = simple_moving_average(clustering_input, windows_size=smooth_size)
    if random_rotation:
        clustering_input = dy_random_rotation(clustering_input)
    motion_modes = kmeans_(clustering_input, n_units)
    return motion_modes


def get_motion_modes(dataset, obs_len, pred_len, n_clusters, dataset_path, dataset_name, smooth_size, random_rotation,
                     traj_seg=False, motion_modes_file=None):
    trajs_list = []
    index1 = [0, 1, 2, 3, 4, 5]  # make full use of training data
    traj_scenarios = dataset.scenario_list
    for i in range(len(traj_scenarios)):
        curr_ped_obs = traj_scenarios[i][0][:, :2]
        curr_ped_pred = traj_scenarios[i][1]
        curr_traj = np.concatenate((curr_ped_obs, curr_ped_pred), axis=0)  # T 2
        if traj_seg:
            for i in index1:
                seq = curr_traj[i:i + pred_len + 2]
                pre_seq = np.repeat(seq[0:1], obs_len + pred_len - seq.shape[0], axis=0)
                seq = np.concatenate((pre_seq, seq), axis=0)
                trajs_list.append(seq)
        trajs_list.append(curr_traj)

    all_trajs = np.stack(trajs_list, axis=0)  # [B T 2]
    all_trajs = translation(all_trajs, obs_len - 1)
    all_trajs, _ = rotation(all_trajs, 0)
    motion_modes = trajectory_motion_modes(all_trajs, obs_len, n_units=n_clusters,
                                           smooth_size=smooth_size, random_rotation=random_rotation)

    # if not os.path.exists(dataset_path):
    #     os.makedirs(dataset_path)
    save_path_file = motion_modes_file
    f = open(save_path_file, 'wb')
    pickle.dump(motion_modes, f)
    f.close()
    print('Finished')

    return motion_modes


# saving motion modes and closest motion modes
def saving_motion_modes(dataloader, motion_modes, obs_len, dataset_path, dataset_name):
    # motion_modes [K pred_len 2]
    closest_mode_indices_list = []
    cls_soft_label_list = []
    traj_scenes = dataloader.seq_array

    for i in range(traj_scenes.shape[0]):
        curr_scene = traj_scenes[i]  # N T 2
        curr_traj = curr_scene[0:1]  # [1 T 2]
        norm_curr_traj = translation(curr_traj, obs_len - 1)
        norm_curr_traj, _ = rotation(norm_curr_traj, 0)
        norm_curr_traj = norm_curr_traj[:, obs_len:]
        norm_curr_traj = norm_curr_traj.reshape(1, -1)  # [1 pred_len*2]
        norm_curr_traj = np.repeat(norm_curr_traj, motion_modes.shape[0], axis=0)  # [K pred_len*2]
        traj_units_ = motion_modes.reshape(motion_modes.shape[0], -1)  # [K pred_len*2]
        distance = np.linalg.norm(norm_curr_traj - traj_units_, axis=-1)  # [K]
        closest_unit_indices = np.argmin(distance)
        closest_unit_indices = np.expand_dims(closest_unit_indices, axis=0)
        closest_mode_indices_list.append(closest_unit_indices)
        cls_soft_label_list.append(-distance)

    closest_mode_indices_array = np.concatenate(closest_mode_indices_list, axis=0)
    cls_soft_label_array = np.stack(cls_soft_label_list, axis=0)

    np.save(dataset_path + dataset_name + '_motion_modes.npy', motion_modes)
    np.save(dataset_path + dataset_name + '_closest_mode_indices.npy', closest_mode_indices_array)
    np.save(dataset_path + dataset_name + '_cls_soft_label.npy', cls_soft_label_array)


def seed(seed: int):
    rand = seed is None
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = not rand
    torch.backends.cudnn.benchmark = rand


def get_rng_state(device):
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state(device) if torch.cuda.is_available and "cuda" in str(device) else None,
        np.random.get_state(),
        random.getstate(),
    )


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Input:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Output:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def gan_g_loss(scores_fake):
    """
    Input:
    - scores_fake: Tensor of shape (N,) containing scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN generator loss
    """
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):
    """
    Input:
    - scores_real: Tensor of shape (N,) giving scores for real samples
    - scores_fake: Tensor of shape (N,) giving scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN discriminator loss
    """
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake


# # loss of G
# def gan_g_loss(scores_fake,  # [nums, 1]
#                mode='mse'):
#     if mode == 'bce':
#         loss_func = nn.BCEWithLogitsLoss()
#     elif mode == 'mse':
#         loss_func = nn.MSELoss()
#
#     y_real = torch.ones_like(scores_fake)  # * random.uniform(0.7, 1.2)
#     return loss_func(scores_fake, y_real)  # make the fake more real, [1]
#
# # loss of D
# def gan_d_loss(scores_real,  # [nums, 1]
#                scores_fake,  # [nums, 1]
#                mode='mse'):
#     if mode == 'bce':
#         loss_func = nn.BCEWithLogitsLoss()
#     elif mode == 'mse':
#         loss_func = nn.MSELoss()
#
#     y_real = torch.ones_like(scores_real)  # * random.uniform(0.7, 1.2)  # [nums] ~ U(0.7, 1.2)
#     y_fake = torch.zeros_like(scores_fake)  # * random.uniform(0, 0.3)  # [nums] = [0.0, 0.0, ..., 0.0]
#
#     loss_real = loss_func(scores_real, y_real)  # make the real more real
#     loss_fake = loss_func(scores_fake, y_fake)  # make the fake faker
#
#     return loss_real+loss_fake  # [1]+[1]



