import torch
from torch import nn, optim, utils
import numpy as np
import os
import sys
import time
import dill
import pandas as pd
import json
import random
import pathlib
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append("trajectron")
import evaluation
from argument_parser import args
from model.trajectron import Trajectron
from model.model_registrar import ModelRegistrar
from model.model_utils import cyclical_lr
from model.dataset import EnvironmentDataset, collate

from environment import Environment, Scene, Node
from utils import maybe_makedirs
from environment import derivative_of
from tensorboardX import SummaryWriter


if not torch.cuda.is_available() or args.device == 'cpu':
    args.device = torch.device('cpu')
else:
    if torch.cuda.device_count() == 1:
        args.device = 'cuda:0'
    args.device = torch.device(args.device)

if args.eval_device is None:
    args.eval_device = torch.device('cpu')

torch.cuda.set_device(args.device)

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

args.subset = 'A2B'

def main():
    ########## Train FLA ##########
    # Load hyperparameters from json
    args.conf = os.getcwd() + '/config/config.json'
    if not os.path.exists(args.conf):
        print('Config json not found!')
    with open(args.conf, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams['dynamic_edges'] = args.dynamic_edges
    hyperparams['edge_state_combine_method'] = args.edge_state_combine_method
    hyperparams['edge_influence_combine_method'] = args.edge_influence_combine_method
    hyperparams['edge_addition_filter'] = args.edge_addition_filter
    hyperparams['edge_removal_filter'] = args.edge_removal_filter
    hyperparams['batch_size'] = args.batch_size
    hyperparams['k_eval'] = args.k_eval
    hyperparams['offline_scene_graph'] = args.offline_scene_graph
    hyperparams['incl_robot_node'] = args.incl_robot_node
    hyperparams['node_freq_mult_train'] = args.node_freq_mult_train
    hyperparams['node_freq_mult_eval'] = args.node_freq_mult_eval
    hyperparams['scene_freq_mult_train'] = args.scene_freq_mult_train
    hyperparams['scene_freq_mult_eval'] = args.scene_freq_mult_eval
    hyperparams['scene_freq_mult_viz'] = args.scene_freq_mult_viz
    hyperparams['edge_encoding'] = not args.no_edge_encoding
    hyperparams['use_map_encoding'] = args.map_encoding
    hyperparams['augment'] = args.augment
    hyperparams['override_attention_radius'] = args.override_attention_radius

    print('-----------------------')
    print('| TRAINING PARAMETERS |')
    print('-----------------------')
    print('| batch_size: %d' % args.batch_size)
    print('| device: %s' % args.device)
    print('| eval_device: %s' % args.eval_device)
    print('| Offline Scene Graph Calculation: %s' % args.offline_scene_graph)
    print('| EE state_combine_method: %s' % args.edge_state_combine_method)
    print('| EIE scheme: %s' % args.edge_influence_combine_method)
    print('| dynamic_edges: %s' % args.dynamic_edges)
    print('| robot node: %s' % args.incl_robot_node)
    print('| edge_addition_filter: %s' % args.edge_addition_filter)
    print('| edge_removal_filter: %s' % args.edge_removal_filter)
    print('| MHL: %s' % hyperparams['minimum_history_length'])
    print('| PH: %s' % hyperparams['prediction_horizon'])
    print('-----------------------')

    log_writer = None
    model_dir = None
    args.log_dir = os.getcwd() + 'experiments/saved_models_DADA'
    if not args.debug:
        # Create the log and model directiory if they're not present.
        """
        model_dir = os.path.join(args.log_dir, args.subset,
                                 '_models_' + time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()) + args.log_tag)
        """
        model_dir = os.path.join(args.log_dir, args.subset + '_models')
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

        # Save config to model directory
        with open(os.path.join(model_dir, 'config.json'), 'w') as conf_json:
            json.dump(hyperparams, conf_json)

        log_writer = SummaryWriter(log_dir=model_dir)
    
    # Load training and evaluation environments and scenes
    train_scenes = []
    args.data_dir_transfer = os.getcwd() + '/experiments/processed_data'
    train_data_path = os.path.join(args.data_dir_transfer, args.subset + '_' + args.train_data_dict)
    with open(train_data_path, 'rb') as f:
        train_env = dill.load(f, encoding='latin1')

    for attention_radius_override in args.override_attention_radius:
        node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
        train_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    if train_env.robot_type is None and hyperparams['incl_robot_node']:
        train_env.robot_type = train_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
        for scene in train_env.scenes:
            scene.add_robot_from_nodes(train_env.robot_type)

    train_scenes = train_env.scenes
    train_scenes_sample_probs = train_env.scenes_freq_mult_prop if args.scene_freq_mult_train else None

    train_dataset = EnvironmentDataset(train_env,
                                       hyperparams['state'],
                                       hyperparams['pred_state'],
                                       scene_freq_mult=hyperparams['scene_freq_mult_train'],
                                       node_freq_mult=hyperparams['node_freq_mult_train'],
                                       hyperparams=hyperparams,
                                       min_history_timesteps=hyperparams['minimum_history_length'],
                                       min_future_timesteps=hyperparams['prediction_horizon'],
                                       return_robot=not args.incl_robot_node)
    train_data_loader = dict()
    for node_type_data_set in train_dataset:
        if len(node_type_data_set) == 0:
            continue

        node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                     collate_fn=collate,
                                                     pin_memory=False if args.device is 'cpu' else True,
                                                     batch_size=args.batch_size,
                                                     shuffle=True,
                                                     num_workers=args.preprocess_workers)
        train_data_loader[node_type_data_set.node_type] = node_type_dataloader

    print(f"Loaded training data from {train_data_path}")

    eval_scenes = []
    eval_scenes_sample_probs = None
    if args.eval_every is not None:
        eval_data_path = os.path.join(args.data_dir_transfer, args.subset + '_' + args.eval_data_dict)
        with open(eval_data_path, 'rb') as f:
            eval_env = dill.load(f, encoding='latin1')

        for attention_radius_override in args.override_attention_radius:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            eval_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

        if eval_env.robot_type is None and hyperparams['incl_robot_node']:
            eval_env.robot_type = eval_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
            for scene in eval_env.scenes:
                scene.add_robot_from_nodes(eval_env.robot_type)

        eval_scenes = eval_env.scenes
        eval_scenes_sample_probs = eval_env.scenes_freq_mult_prop if args.scene_freq_mult_eval else None

        eval_dataset = EnvironmentDataset(eval_env,
                                          hyperparams['state'],
                                          hyperparams['pred_state'],
                                          scene_freq_mult=hyperparams['scene_freq_mult_eval'],
                                          node_freq_mult=hyperparams['node_freq_mult_eval'],
                                          hyperparams=hyperparams,
                                          min_history_timesteps=hyperparams['minimum_history_length'],
                                          min_future_timesteps=hyperparams['prediction_horizon'],
                                          return_robot=not args.incl_robot_node)
        eval_data_loader = dict()
        for node_type_data_set in eval_dataset:
            if len(node_type_data_set) == 0:
                continue

            node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                         collate_fn=collate,
                                                         pin_memory=False if args.eval_device is 'cpu' else True,
                                                         batch_size=args.eval_batch_size,
                                                         shuffle=True,
                                                         num_workers=args.preprocess_workers)
            eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

        print(f"Loaded evaluation data from {eval_data_path}")

    # Offline Calculate Scene Graph
    if hyperparams['offline_scene_graph'] == 'yes':
        print(f"Offline calculating scene graphs")
        for i, scene in enumerate(train_scenes):
            scene.calculate_scene_graph(train_env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
            print(f"Created Scene Graph for Training Scene {i}")

        for i, scene in enumerate(eval_scenes):
            scene.calculate_scene_graph(eval_env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
            print(f"Created Scene Graph for Evaluation Scene {i}")

    model_registrar = ModelRegistrar(model_dir, args.device)

    trajectron = Trajectron(model_registrar,
                            hyperparams,
                            log_writer,
                            args.device)

    trajectron.set_environment(train_env)
    trajectron.set_annealing_params()
    print('Created Training Model.')

    eval_trajectron = None
    if args.eval_every is not None or args.vis_every is not None:
        eval_trajectron = Trajectron(model_registrar,
                                     hyperparams,
                                     log_writer,
                                     args.eval_device)
        eval_trajectron.set_environment(eval_env)
        eval_trajectron.set_annealing_params()
    print('Created Evaluation Model.')


    optimizer = dict()
    lr_scheduler = dict()
    for node_type in train_env.NodeType:
        if node_type not in hyperparams['pred_state']:
            continue
        optimizer[node_type] = optim.Adam([
            {'params': model_registrar.get_all_but_namelist_match(
                ['map_encoder', 'discriminator_inlay']).parameters()}, 
            {'params': model_registrar.get_namelist_match(['map_encoder']).parameters(), 'lr':8e-4},
            ], lr=hyperparams['learning_rate'])
        # Set Learning Rate
        if hyperparams['learning_rate_style'] == 'const':
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(
                optimizer[node_type], gamma=1.0
                )
        elif hyperparams['learning_rate_style'] == 'exp':
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(
                optimizer[node_type], gamma=hyperparams['learning_decay_rate']
                )

    optimizer_d_inlay = optim.Adam([
        {'params': model_registrar.get_namelist_match(['discriminator_inlay']).parameters(), 'lr':5e-4},
        ], lr=hyperparams['learning_rate'])
    
    curr_iter_node_type = {node_type: 0 for node_type in train_data_loader.keys()}
    source_loader, target_loader = train_data_loader['PEDESTRIAN'], eval_data_loader['PEDESTRIAN']
    len_source, len_target = len(source_loader), len(target_loader)
    len_max = max(len_source, len_target)
    curr_iter = 0
    max_train_epochs = 1 * args.train_epochs  # args.train_epochs=100
    for epoch in range(1, max_train_epochs + 1):
        print('***** Epoch {}/{} *****'.format(epoch, max_train_epochs))
        #################################
        #           TRAINING            #
        #################################
        model_registrar.to(args.device)
        train_dataset.augment = args.augment
        
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        index_batch = 0
        
        for index_batch in range(len_max):
            print('**** s_real+DA Epoch {}/{}, batch {}/{} ****'.format(epoch, max_train_epochs,
                                                                        index_batch+1, len_max)
                  )
            if index_batch % len_source == 0:
                del source_iter
                source_iter = iter(source_loader)
                print('*** Source Iter is reset ***')
            if index_batch % len_target == 0:
                del target_iter
                target_iter = iter(target_loader)
                print('*** Target Iter is reset ***')
            batch_s, batch_t = next(source_iter), next(target_iter)

            trajectron.set_curr_iter(curr_iter)
            trajectron.step_annealers('PEDESTRIAN')
            # 1. discriminator
            optimizer_d_inlay.zero_grad()
            d_inlay_loss = trajectron.train_loss_D(batch_s, batch_t, 'PEDESTRIAN')
            d_inlay_loss.backward()
            if hyperparams['grad_clip'] is not None:
                nn.utils.clip_grad_value_(model_registrar.parameters(),
                                          hyperparams['grad_clip'])
            optimizer_d_inlay.step()
            # 2. generator
            optimizer['PEDESTRIAN'].zero_grad()
            g_inlay_loss = trajectron.train_loss_G(batch_s, 'PEDESTRIAN')  # batch_t_fake(可选)
            g_inlay_loss.backward()
            if hyperparams['grad_clip'] is not None:
                nn.utils.clip_grad_value_(model_registrar.parameters(), 
                                          hyperparams['grad_clip'])
            optimizer['PEDESTRIAN'].step()
            # 3. trajectron
            optimizer['PEDESTRIAN'].zero_grad()
            pred_loss = trajectron.train_loss_trajectron(batch_s, 'PEDESTRIAN')  # batch
            pred_loss.backward()
            if hyperparams['grad_clip'] is not None:
                nn.utils.clip_grad_value_(model_registrar.parameters(), 
                                          hyperparams['grad_clip'])
            optimizer['PEDESTRIAN'].step()
            
            lr_scheduler['PEDESTRIAN'].step()
            if not args.debug:
                log_writer.add_scalar(f"{'PEDESTRIAN'}/train/learning_rate",
                                      lr_scheduler['PEDESTRIAN'].get_last_lr()[0], curr_iter)
                log_writer.add_scalar(f"{'PEDESTRIAN'}/train/pred_loss", pred_loss, curr_iter)
                log_writer.add_scalar(f"{'PEDESTRIAN'}/train/d_inlay_loss", d_inlay_loss, curr_iter)
                log_writer.add_scalar(f"{'PEDESTRIAN'}/train/g_inlay_loss", g_inlay_loss, curr_iter)
                
            print('** Log has been written **')
            curr_iter += 1
        del source_iter
        del target_iter
        train_dataset.augment = False
         
        if args.eval_every is not None or args.vis_every is not None:
            eval_trajectron.set_curr_iter(epoch)


        #################################
        #           EVALUATION          #
        #################################
        if args.eval_every is not None and not args.debug and epoch % args.eval_every == 0 and epoch > 0:
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            model_registrar.to(args.eval_device)
            with torch.no_grad():
                # Calculate evaluation loss
                for node_type, data_loader in eval_data_loader.items():
                    eval_loss = []
                    print(f"Starting Evaluation @ epoch {epoch} for node type: {node_type}")
                    pbar = tqdm(data_loader, ncols=80)
                    for batch in pbar:
                        eval_loss_node_type = eval_trajectron.eval_loss(batch, node_type)
                        pbar.set_description(f"Epoch {epoch}, {node_type} L: {eval_loss_node_type.item():.2f}")
                        eval_loss.append({node_type: {'nll': [eval_loss_node_type]}})
                        del batch

                    evaluation.log_batch_errors(eval_loss,
                                                log_writer,
                                                f"{node_type}/eval_loss",
                                                epoch)

                # Predict batch timesteps for evaluation dataset evaluation
                eval_batch_errors = []
                for scene in tqdm(eval_scenes, desc='Sample Evaluation', ncols=80):
                    timesteps = scene.sample_timesteps(args.eval_batch_size)

                    predictions = eval_trajectron.predict(scene,
                                                          timesteps,
                                                          ph,
                                                          num_samples=50,
                                                          min_future_timesteps=ph,
                                                          full_dist=False)

                    eval_batch_errors.append(evaluation.compute_batch_statistics(predictions,
                                                                                 scene.dt,
                                                                                 max_hl=max_hl,
                                                                                 ph=ph,
                                                                                 node_type_enum=eval_env.NodeType,
                                                                                 map=scene.map))

                evaluation.log_batch_errors(eval_batch_errors,
                                            log_writer,
                                            'eval',
                                            epoch,
                                            bar_plot=['kde'],
                                            box_plot=['ade', 'fde'])

                # Predict maximum likelihood batch timesteps for evaluation dataset evaluation
                eval_batch_errors_ml = []
                for scene in tqdm(eval_scenes, desc='MM Evaluation', ncols=80):
                    timesteps = scene.sample_timesteps(scene.timesteps)

                    predictions = eval_trajectron.predict(scene,
                                                          timesteps,
                                                          ph,
                                                          num_samples=1,
                                                          min_future_timesteps=ph,
                                                          z_mode=True,
                                                          gmm_mode=True,
                                                          full_dist=False)

                    eval_batch_errors_ml.append(evaluation.compute_batch_statistics(predictions,
                                                                                    scene.dt,
                                                                                    max_hl=max_hl,
                                                                                    ph=ph,
                                                                                    map=scene.map,
                                                                                    node_type_enum=eval_env.NodeType,
                                                                                    kde=False))

                evaluation.log_batch_errors(eval_batch_errors_ml,
                                            log_writer,
                                            'eval/ml',
                                            epoch)

        if args.save_every is not None and args.debug is False and epoch % args.save_every == 0:
            model_registrar.save_models(epoch)


if __name__ == '__main__':
    main()

