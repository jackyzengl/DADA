import torch
import random
import torch.nn as nn


# loss of G
def gan_g_loss(scores_fake,
               mode='bce'):
    if mode == 'bce':
        loss_func = nn.BCEWithLogitsLoss()
    elif mode == 'mse':
        loss_func = nn.MSELoss()
        
    y_real = torch.ones_like(scores_fake)  # * random.uniform(0.7, 1.2)
    return loss_func(scores_fake, y_real)  # make the fake more real, [1]

# loss of D
def gan_d_loss(scores_real, 
               scores_fake,
               mode='bce'):
    if mode == 'bce':
        loss_func = nn.BCEWithLogitsLoss()
    elif mode == 'mse':
        loss_func = nn.MSELoss()
        
    y_real = torch.ones_like(scores_real)  # * random.uniform(0.7, 1.2)  # [nums] ~ U(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake)  # * random.uniform(0, 0.3)  # [nums] = [0.0, 0.0, ..., 0.0]
    
    loss_real = loss_func(scores_real, y_real)  # make the real more real
    loss_fake = loss_func(scores_fake, y_fake)  # make the fake more fake
    
    return loss_real, loss_fake  # [1], [1]

# l2 loss
def l2_loss(pred_traj,
            pred_traj_gt, 
            loss_mask, random=0, mode='average'):
    seq_len, batch, _ = pred_traj.size()
    loss = (loss_mask.unsqueeze(dim=2) * (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2)  # [nums, 20, 2]
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)
