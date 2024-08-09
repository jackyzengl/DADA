import sys
import os
import dill
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob

sys.path.append("trajectron")
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
import evaluation


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--subset", type=str, default='A2B')
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int, default=100)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--output_path", help="path to output csv file", type=str)
parser.add_argument("--node_type", help="node type to evaluate", type=str)
args = parser.parse_args()


def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')

    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams


if __name__ == "__main__":
    dataset = ['A2B']
    for subset in dataset:
        args.subset = subset
        eval_path = os.getcwd() + '/experiments/pedestrian/results_DADA/' + args.subset + '_ade_best_of.csv'
        print(eval_path)
        if not os.path.exists(eval_path):
            ######### Test #########
            args.data = os.getcwd() + '/experiments/processed_data/' + args.subset + '_test.pkl'
            with open(args.data, 'rb') as f:
                env = dill.load(f, encoding='latin1')

            args.model = os.getcwd() + '/experiments/saved_models_DADA/' + args.subset + '_models'
            eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)

            if 'override_attention_radius' in hyperparams:
                for attention_radius_override in hyperparams['override_attention_radius']:
                    node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
                    env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

            scenes = env.scenes

            print("-- Preparing Node Graph")
            for scene in tqdm(scenes):
                scene.calculate_scene_graph(env.attention_radius,
                                            hyperparams['edge_addition_filter'],
                                            hyperparams['edge_removal_filter'])

            ph = hyperparams['prediction_horizon']
            max_hl = hyperparams['maximum_history_length']

            with torch.no_grad():
                ############### BEST OF 20 ###############
                eval_ade_batch_errors = np.array([])
                eval_fde_batch_errors = np.array([])
                eval_kde_nll = np.array([])
                print("-- Evaluating best of 20")
                for i, scene in enumerate(scenes):
                    print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
                    for t in tqdm(range(0, scene.timesteps, 10)):
                        timesteps = np.arange(t, t + 10)
                        predictions = eval_stg.predict(scene,
                                                    timesteps,
                                                    ph,
                                                    num_samples=20,
                                                    min_history_timesteps=7,
                                                    min_future_timesteps=12,
                                                    z_mode=False,
                                                    gmm_mode=False,
                                                    full_dist=False)

                        if not predictions:
                            continue

                        batch_error_dict = evaluation.compute_batch_statistics(predictions,
                                                                            scene.dt,
                                                                            max_hl=max_hl,
                                                                            ph=ph,
                                                                            node_type_enum=env.NodeType,
                                                                            map=None,
                                                                            best_of=True,
                                                                            prune_ph_to_future=True)
                        args.node_type = 'PEDESTRIAN'
                        eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
                        eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))

                args.output_path = os.getcwd() + '/experiments/pedestrian/results_DADA'
                pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'best_of'}
                            ).to_csv(os.path.join(args.output_path, args.subset + '_ade_best_of.csv'))
                pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'best_of'}
                            ).to_csv(os.path.join(args.output_path, args.subset + '_fde_best_of.csv'))

    ######### Print Metrics #########
    dataset_names = ['A2B']
    alg_name = "Ours"

    # ADE of All
    perf_df = pd.DataFrame()
    for dataset in dataset_names:
        result_path = os.getcwd() + '/experiments/pedestrian/results_DADA/{}_ade_best_of.csv'.format(dataset)
        for f in glob.glob(result_path):
            dataset_df = pd.read_csv(f)
            dataset_df['dataset'] = dataset
            dataset_df['method'] = alg_name
            perf_df = perf_df.append( dataset_df, ignore_index=True, sort=False)
            del perf_df['Unnamed: 0']

    for dataset in dataset_names:
        print('ADE Best of 20 for ' + dataset + ': ')
        if dataset != 'Average':
            print(f"{perf_df[(perf_df['method'] == alg_name) & (perf_df['dataset'] == dataset)]['value'].mean()}")
        else:
            print(f"{perf_df[(perf_df['method'] == alg_name)]['value'].mean()}")
    del perf_df

    # FDE of All
    perf_df = pd.DataFrame()
    for dataset in dataset_names:
        result_path = os.getcwd() + '/experiments/pedestrian/results_DADA/{}_fde_best_of.csv'.format(dataset)
        for f in glob.glob(result_path):
            dataset_df = pd.read_csv(f)
            dataset_df['dataset'] = dataset
            dataset_df['method'] = alg_name
            perf_df = perf_df.append(dataset_df, ignore_index=True, sort=False)
            del perf_df['Unnamed: 0']

    for dataset in dataset_names:
        print('FDE Best of 20 for ' + dataset + ': ')
        if dataset != 'Average':
            print(f"{perf_df[(perf_df['method'] == alg_name) & (perf_df['dataset'] == dataset)]['value'].mean()}")
        else:
            print(f"{perf_df[(perf_df['method'] == alg_name)]['value'].mean()}")
    del perf_df
