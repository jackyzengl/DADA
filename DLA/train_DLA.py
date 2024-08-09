import argparse
import gc
import logging
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import time


from data_process.loader import data_loader
from models_gat import Generator, Discriminator

from losses import gan_g_loss, gan_d_loss, l2_loss

from utils import bool_flag, relative_to_abs


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = '%(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--subset', default='A2B', type=str)
parser.add_argument('--delim', default='\t')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--skip', default=1, type=int)
parser.add_argument('--batch_size', default=32, type=int)

# Optimization
parser.add_argument('--num_iterations', default=20000, type=int)
parser.add_argument('--num_epochs', default=250, type=int)

# Model Options
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)
parser.add_argument('--gat_headdim_list', default=[64, 32, 64], type=list)
parser.add_argument('--gat_head_list', default=[4, 1], type=list)

# Generator Options
parser.add_argument('--g_conv_dim', default=64, type=int)
parser.add_argument('--g_conv_k_size', default=5, type=int)
parser.add_argument('--g_conv_pad_size', default=2, type=int)
parser.add_argument('--g_learning_rate', default=5e-4, type=float)
parser.add_argument('--clipping_threshold_g', default=0, type=float)

# Discriminator Options
parser.add_argument('--d_conv_dim', default=64, type=int)
parser.add_argument('--d_conv_k_size', default=3, type=int)
parser.add_argument('--d_conv_padding', default=1, type=int)
parser.add_argument('--d_pool_k_size', default=3, type=int)
parser.add_argument('--d_pool_stride', default=2, type=int)
parser.add_argument('--d_pool_padding', default=1, type=int)
parser.add_argument('--d_learning_rate', default=5e-4, type=float)
parser.add_argument('--clipping_threshold_d', default=0, type=float)

# Loss Options
parser.add_argument('--gan_loss_mode', default='mse', type=str, help='mse, bce')
parser.add_argument('--l2_loss_weight', default=1.0, type=float)
parser.add_argument('--lambda_idt_loss', default=1.0, type=float)

# Output
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_loss_every', default=5, type=int)
parser.add_argument('--save_checkpoint_every', default=10, type=int)
parser.add_argument('--checkpoint_name', default='checkpoints')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=True, type=bool)

# Misc
parser.add_argument('--use_gpu', default=True, type=bool)
parser.add_argument('--gpu_num', default="0", type=str)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    
    data_path = '../datasets/' + args.subset
    source_path = os.path.join(data_path, 'train_origin')
    target_path = os.path.join(data_path, 'val')
    
    long_dtype, float_dtype = get_dtypes(args)

    logger.info("Initializing source dataset from: {}.".format(args.subset))
    source_dset, source_loader = data_loader(args, source_path)
    
    logger.info("Initializing target dataset from: {}.".format(args.subset))
    target_dset, target_loader = data_loader(args, target_path)

    iterations_per_epoch = max(len(source_dset), len(target_dset)) / args.batch_size
    
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)
        logger.info('There are {} iterations per epoch.'.format(iterations_per_epoch))
        logger.info('Total epochs: {}.'.format(args.num_epochs))
        logger.info('Total steps: {}.'.format(args.num_iterations))

    # G_S2T
    generator_s2t = Generator(g_conv_dim=args.g_conv_dim,
                              g_conv_k_size=args.g_conv_k_size,
                              g_conv_pad_size=args.g_conv_pad_size,
                              gat_headdim_list=args.gat_headdim_list,
                              gat_head_list=args.gat_head_list).cuda()
    generator_s2t.apply(init_weights)
    generator_s2t.type(float_dtype).train()
    logger.info('Generator_s2t has prepared.')

    # G_T2S
    generator_t2s = Generator(g_conv_dim=args.g_conv_dim,
                              g_conv_k_size=args.g_conv_k_size,
                              g_conv_pad_size=args.g_conv_pad_size,
                              gat_headdim_list=args.gat_headdim_list,
                              gat_head_list=args.gat_head_list).cuda()
    generator_t2s.apply(init_weights)
    generator_t2s.type(float_dtype).train()
    logger.info('Generator_t2s has prepared.')

    # D_T
    discriminator_t = Discriminator(d_conv_dim=args.d_conv_dim,
                                    d_conv_k_size=args.d_conv_k_size,
                                    d_conv_padding=args.d_conv_padding,
                                    d_pool_k_size=args.d_pool_k_size,
                                    d_pool_stride=args.d_pool_stride,
                                    d_pool_padding=args.d_pool_padding,
                                    gat_headdim_list=args.gat_headdim_list,
                                    gat_head_list=args.gat_head_list).cuda()
    discriminator_t.apply(init_weights)
    discriminator_t.type(float_dtype).train()
    logger.info('Discriminator_t has prepared.')

    # D_S
    discriminator_s = Discriminator(d_conv_dim=args.d_conv_dim,
                                    d_conv_k_size=args.d_conv_k_size,
                                    d_conv_padding=args.d_conv_padding,
                                    d_pool_k_size=args.d_pool_k_size,
                                    d_pool_stride=args.d_pool_stride,
                                    d_pool_padding=args.d_pool_padding,
                                    gat_headdim_list=args.gat_headdim_list,
                                    gat_head_list=args.gat_head_list).cuda()
    discriminator_s.apply(init_weights)
    discriminator_s.type(float_dtype).train()
    logger.info('Discriminator_s has prepared.')

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    g_optimizer = optim.Adam(itertools.chain(generator_s2t.parameters(), generator_t2s.parameters()),
                             lr=args.g_learning_rate,
                             )
    d_optimizer = optim.Adam(itertools.chain(discriminator_s.parameters(), discriminator_t.parameters()),
                             lr=args.d_learning_rate,
                             )

    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint:
        restore_path = os.path.join(args.output_dir, 'tools', 'DLA', args.checkpoint_name,
                                    '{}_model.pth'.format(args.subset))

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        generator_s2t.load_state_dict(checkpoint['g_s2t_state'])
        generator_t2s.load_state_dict(checkpoint['g_t2s_state'])
        discriminator_s.load_state_dict(checkpoint['d_s_state'])
        discriminator_t.load_state_dict(checkpoint['d_t_state'])
        g_optimizer.load_state_dict(checkpoint['g_optim_state'])
        d_optimizer.load_state_dict(checkpoint['d_optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        t, epoch = 0, 0
        checkpoint = {'args': args.__dict__,
                      'G_gen_losses': [], 'G_rec_losses': [], 'G_idt_losses': [],
                      'G_losses': [],
                      'D_real_losses': [], 'D_fake_losses': [],
                      'D_losses': [],
                      'losses_ts': [],
                      'sample_ts': [], 'restore_ts': [],
                      'counters': {'t': None, 'epoch': None,},
                      'g_s2t_state': None, 'g_t2s_state': None,
                      'd_s_state': None, 'd_t_state': None,
                      'g_optim_state': None, 'd_optim_state': None,
                      }

    ########## Train DLA ##########
    len_source, len_target = len(source_loader), len(target_loader)
    len_max = max(len_source, len_target)

    while t < args.num_iterations:
        epoch += 1
        logger.info('Starting epoch {}:'.format(epoch))
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        index_batch = 0

        for index_batch in range(len_max):
            if index_batch % len_source == 0:
                del source_iter
                source_iter = iter(source_loader)
                logger.info('** Source iter is reset.')
            if index_batch % len_target == 0:
                del target_iter
                target_iter = iter(target_loader)
                logger.info('** Target iter is reset.')

            batch_s, batch_t = next(source_iter), next(target_iter)
            batch_s = [tensor.cuda() for tensor in batch_s]
            (obs_traj_s, pred_traj_gt_s, whole_traj_s,
             obs_traj_rel_s, pred_traj_gt_rel_s, whole_traj_rel_s,
             non_linear_ped_s, loss_mask_s, seq_start_end_s) = batch_s
            batch_t = [tensor.cuda() for tensor in batch_t]
            (obs_traj_t, pred_traj_gt_t, whole_traj_t,
             obs_traj_rel_t, pred_traj_gt_rel_t, whole_traj_rel_t,
             non_linear_ped_t, loss_mask_t, seq_start_end_t) = batch_t
            logger.info('* batch data loaded.')

            #============ train D ============#
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            # Train D_S, whole_traj_t + obs_traj_s
            traj_rel_fake = generator_t2s(whole_traj_t, whole_traj_rel_t,
                                          seq_start_end_t)  # G_T2S(T)=S', [20, nums, 2]
            traj_fake = relative_to_abs(traj_rel_fake,
                                        whole_traj_t[0])  # S'_abs, [20, nums, 2], X_TS
            scores_real = discriminator_s(obs_traj_s, obs_traj_rel_s,
                                          seq_start_end_s)  # D_S(S), [nums, 8]
            scores_fake = discriminator_s(traj_fake, traj_rel_fake,
                                          seq_start_end_t)  # D_S(S'), [nums, 20]
            d_s_loss_real, d_s_loss_fake = d_loss_fn(scores_real, scores_fake,
                                                     mode=args.gan_loss_mode)
            d_s_loss = d_s_loss_real + d_s_loss_fake
            d_s_loss.backward()
            print('* D_S process has done.')

            # Train D_T, whole_traj_s + obs_traj_t
            traj_rel_fake = generator_s2t(whole_traj_s, whole_traj_rel_s,
                                          seq_start_end_s)   # G_S2T(S)=T', [20, nums, 2]
            traj_fake = relative_to_abs(traj_rel_fake,
                                        whole_traj_s[0])  # T'_abs, [20, nums, 2]
            scores_real = discriminator_t(obs_traj_t, obs_traj_rel_t,
                                          seq_start_end_t)  # D_T(T), [nums, 8]
            scores_fake = discriminator_t(traj_fake, traj_rel_fake,
                                          seq_start_end_s)  # D_T(T'), [nums, 20]
            d_t_loss_real, d_t_loss_fake = d_loss_fn(scores_real, scores_fake,
                                                     mode=args.gan_loss_mode)
            d_t_loss = d_t_loss_real + d_t_loss_fake
            d_t_loss.backward()
            print('* D_T process has done.')

            d_real_loss = d_s_loss_real + d_t_loss_real
            d_fake_loss = d_s_loss_fake + d_t_loss_fake
            d_loss = d_s_loss + d_t_loss
            if args.clipping_threshold_d > 0:
                nn.utils.clip_grad_norm_(discriminator_s.parameters(), args.clipping_threshold_d)
                nn.utils.clip_grad_norm_(discriminator_t.parameters(), args.clipping_threshold_d)
            d_optimizer.step()
            print('* D has been trained.')

            #============ train G ============#
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()

            if args.lambda_idt_loss > 0 and args.l2_loss_weight > 0:
                # G_S2T(T) should be identity if T is fed: ||G_S2T(T) - T||, whole_traj_s + obs_traj_t
                idt_G_S2T = generator_s2t(obs_traj_t, obs_traj_rel_t,
                                          seq_start_end_t)  # [8, nums, 2]
                g_s2t_idt_loss = args.l2_loss_weight * l2_loss(idt_G_S2T, obs_traj_rel_t,
                                                               loss_mask_t[:, :8],
                                                               mode='average')
                g_s2t_idt_loss *= args.lambda_idt_loss

                # G_T2S(S) should be identity if S is fed: ||G_T2S(S) - S||, whole_traj_t + obs_traj_s
                idt_G_T2S = generator_t2s(obs_traj_s, obs_traj_rel_s,
                                          seq_start_end_s)  # [8, nums, 2]
                g_t2s_idt_loss = args.l2_loss_weight * l2_loss(idt_G_T2S, obs_traj_rel_s,
                                                               loss_mask_s[:, :8],
                                                               mode='average')
                g_t2s_idt_loss *= args.lambda_idt_loss
            else:
                g_s2t_idt_loss = 0
                g_t2s_idt_loss = 0

            # Train S-T-S cycle, whole_traj_s + obs_traj_t
            # ||G_S2T(S) - T||
            traj_rel_fake = generator_s2t(whole_traj_s, whole_traj_rel_s,
                                          seq_start_end_s)  # G_S2T(S)=T', [20, nums, 2]
            traj_fake = relative_to_abs(traj_rel_fake,
                                        whole_traj_s[0])  # T'_abs, [20, nums, 2]
            scores_fake = discriminator_t(traj_fake, traj_rel_fake,
                                          seq_start_end_s)  # D_T(T'), [nums, 20]
            g_1_loss = g_loss_fn(scores_fake,
                                 mode=args.gan_loss_mode)
            # ||G_T2S(T') - S||
            reconst_traj_rel = generator_t2s(traj_fake, traj_rel_fake,
                                             seq_start_end_s)  # G_T2S(T')=S', [20, nums, 2]
            if args.l2_loss_weight > 0:
                g_rec1_loss = args.l2_loss_weight * l2_loss(reconst_traj_rel, whole_traj_rel_s,
                                                            loss_mask_s[:, :],
                                                            mode='average')
            print('* G_1 process has done.')

            # Train T-S-T cycle, whole_traj_t + obs_traj_s
            # ||G_T2S(T) - S||
            traj_rel_fake = generator_t2s(whole_traj_t, whole_traj_rel_t,
                                          seq_start_end_t)  # G_T2S(T)=S', [20, nums, 2]
            traj_fake = relative_to_abs(traj_rel_fake,
                                        whole_traj_t[0])  # S'_abs, [20, nums, 2]
            scores_fake = discriminator_s(traj_fake, traj_rel_fake,
                                          seq_start_end_t)  # D_S(S'), [nums, 20]
            g_2_loss = g_loss_fn(scores_fake,
                                 mode=args.gan_loss_mode)
            # ||G_S2T(S') - T||
            reconst_traj_rel = generator_s2t(traj_fake, traj_rel_fake,
                                             seq_start_end_t)  # G_S2T(S')=T', [20, nums, 2]
            if args.l2_loss_weight > 0:
                g_rec2_loss = args.l2_loss_weight * l2_loss(reconst_traj_rel, whole_traj_rel_t,
                                                            loss_mask_t[:, :],
                                                            mode='average')
            print('* G_2 process has done.')

            g_gen_loss = g_1_loss + g_2_loss
            g_rec_loss = g_rec1_loss + g_rec2_loss
            g_idt_loss = g_s2t_idt_loss + g_t2s_idt_loss
            g_loss = g_gen_loss + g_rec_loss + g_idt_loss
            g_loss.backward()
            if args.clipping_threshold_g > 0:
                nn.utils.clip_grad_norm_(generator_s2t.parameters(), args.clipping_threshold_g)
                nn.utils.clip_grad_norm_(generator_t2s.parameters(), args.clipping_threshold_g)
            g_optimizer.step()
            print('* G has been trained.')

            #============ Save ============#
            # Save and print the loss
            if t % args.print_loss_every == 0:
                logger.info('[Step]: {} / {}'.format(t + 1, args.num_iterations))

                logger.info('[D_real_loss]: {:.3f}, [D_fake_loss]: {:.3f}'.format(d_real_loss, d_fake_loss))
                logger.info('[D_loss]: {:.4f}'.format(d_loss))
                checkpoint['D_real_losses'].append(d_real_loss)
                checkpoint['D_fake_losses'].append(d_fake_loss)
                checkpoint['D_losses'].append(d_loss)

                logger.info('[G_gen_loss]: {:.3f}, [G_rec_losses]: {:.3f}, [G_idt_losses]: {:.3f}'.format(g_gen_loss, g_rec_loss, g_idt_loss))
                logger.info('[G_loss]: {:.4f}'.format(g_loss))
                checkpoint['G_gen_losses'].append(g_gen_loss)
                checkpoint['G_rec_losses'].append(g_rec_loss)
                checkpoint['G_idt_losses'].append(g_idt_loss)
                checkpoint['G_losses'].append(g_loss)

                checkpoint['losses_ts'].append(t)

            # Save a checkpoint
            if t > 0 and t % args.save_checkpoint_every == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                checkpoint['g_s2t_state'] = generator_s2t.state_dict()
                checkpoint['g_t2s_state'] = generator_t2s.state_dict()
                checkpoint['g_optim_state'] = g_optimizer.state_dict()

                checkpoint['d_s_state'] = discriminator_s.state_dict()
                checkpoint['d_t_state'] = discriminator_t.state_dict()
                checkpoint['d_optim_state'] = d_optimizer.state_dict()

                checkpoint_path = os.path.join(args.output_dir, args.checkpoint_name,
                                               '{}_dla_model.pth'.format(args.subset))
                logger.info('Saving checkpoint at: {}...'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

            t += 1
            if t >= args.num_iterations:
                break

        del source_iter
        del target_iter

    logger.info('TRAIN finished.')
    
    
    ########## Transfer Data ##########
    # Load model
    checkpoint_path = os.path.join(args.output_dir, args.checkpoint_name,
                                   '{}_dla_model.pth'.format(args.subset))
    logger.info('Loading model from checkpoint {}...'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    generator_s2t.load_state_dict(checkpoint['g_s2t_state'])
    generator_t2s.load_state_dict(checkpoint['g_t2s_state'])
    discriminator_s.load_state_dict(checkpoint['d_s_state'])
    discriminator_t.load_state_dict(checkpoint['d_t_state'])
    t_checkpoint = checkpoint['counters']['t']
    epoch = checkpoint['counters']['epoch']
    logger.info('The checkpoint is from {}th step.'.format(t_checkpoint))
    logger.info('Done.')

    target_traj_fake_list = []  # [[20, nums, 2], [20, nums, 2], ...]
    seq_start_end_s_list = []  # [[bs, 2], [bs, 2], ...]

    len_source, len_target = len(source_loader), len(target_loader)
    len_max = max(len_source, len_target)
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    index_batch = 0

    for index_batch in range(len_max):
        if index_batch % len_source == 0:
            del source_iter
            source_iter = iter(source_loader)
            logger.info('** Source iter is reset.')
        if index_batch % len_target == 0:
            del target_iter
            target_iter = iter(target_loader)
            logger.info('** Target iter is reset.')
        batch_s, batch_t = next(source_iter), next(target_iter)
        logger.info('Starting batch {}:'.format(index_batch))
        # Load data
        batch_s = [tensor.cuda() for tensor in batch_s]
        (obs_traj_s, pred_traj_gt_s, whole_traj_s,
         obs_traj_rel_s, pred_traj_gt_rel_s, whole_traj_rel_s,
         non_linear_ped_s, loss_mask_s, seq_start_end_s) = batch_s
        batch_t = [tensor.cuda() for tensor in batch_t]
        (obs_traj_t, pred_traj_gt_t, whole_traj_t,
         obs_traj_rel_t, pred_traj_gt_rel_t, whole_traj_rel_t,
         non_linear_ped_t, loss_mask_t, seq_start_end_t) = batch_t
        logger.info('* batch data loaded.')

        target_traj_rel_fake = generator_s2t(whole_traj_s, whole_traj_rel_s,
                                             seq_start_end_s)  # T'
        target_traj_fake = relative_to_abs(target_traj_rel_fake,
                                           whole_traj_s[0])  # T'_abs
        logger.info('* target_traj_fake has got.')

        target_traj_fake_list.append(target_traj_fake)
        seq_start_end_s_list.append(seq_start_end_s)

    del source_iter
    del target_iter

    txt_save_path = r'../datasets/' + args.subset
    if not os.path.exists(txt_save_path):
        os.makedirs(txt_save_path)
    txt_target_traj_fake_path = os.path.join(txt_save_path, 'train', 'target_traj_fake_0.txt')
    txt_target_traj_fake = open(txt_target_traj_fake_path, 'w')

    t = 0.0
    dt = 20.0
    idx = 1.0
    frame_x = 0
    if 'C2' in args.subset:
        frame_max = 1000
    else:
        frame_max = 10000
    x = 0
    with torch.no_grad():
        for (target_traj_fake, seq_start_end_s) in zip(target_traj_fake_list, seq_start_end_s_list):
            if frame_x >= frame_max:
                x += 1
                if x >= 12:
                    break
                txt_target_traj_fake_path = os.path.join(txt_save_path, 'train', 'target_traj_fake_{}.txt'.format(x))
                txt_target_traj_fake = open(txt_target_traj_fake_path, 'w')
                frame_x = 0

            for start, end in seq_start_end_s:
                nums_i = int(end-start)
                list2write = target_traj_fake[:, start:end, :].tolist()
                for ti in range(int(dt)):
                    for ni in range(nums_i):
                        str2write = str((ti+t)*10) + '\t' + str(ni+idx) + '\t' +\
                            str(round(list2write[ti][ni][0], 8)) + '\t' + str(round(list2write[ti][ni][1], 8)) + '\n'
                        txt_target_traj_fake.write(str2write)
                idx += nums_i
                t += dt
                frame_x += dt*10
    logger.info('TRANSFER finished.')

if __name__ == '__main__':
    start_time = time.time()
    args = parser.parse_args()
    main(args)
    print(f'\nuse time: {time.time()-start_time}')
