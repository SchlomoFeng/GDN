# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

from sklearn.preprocessing import MinMaxScaler

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.iostream import printsep

from datasets.TimeDataset import TimeDataset


from models.GDN import GDN

from train import train
from test  import test
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import json
import random

class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config['dataset'] 
        train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
        test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)
       
        train, test = train_orig, test_orig

        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)

        self.feature_map = feature_map

        train_dataset_indata = construct_data(train, feature_map, labels=0)
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())


        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)


        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                            shuffle=False, num_workers=0)


        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        self.model = GDN(edge_index_sets, len(feature_map), 
                dim=train_config['dim'], 
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                topk=train_config['topk']
            ).to(self.device)



    def run(self):

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]

            self.train_log = train(self.model, model_save_path, 
                config = train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader, 
                feature_map=self.feature_map,
                test_dataloader=self.test_dataloader,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
                dataset_name=self.env_config['dataset']
            )
        
        # test            
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        # Get feature names for enhanced analysis
        feature_names = list(self.feature_map.keys()) if self.feature_map else None
        experiment_name = f"{self.env_config['dataset']}_seed{self.train_config['seed']}"
        
        print("\n" + "="*60)
        print("Starting enhanced testing with detailed anomaly analysis...")
        print("="*60)

        _, self.test_result = test(best_model, self.test_dataloader, 
                                 feature_names=feature_names, 
                                 enable_detailed_analysis=True,
                                 experiment_name=f"{experiment_name}_test")
        
        _, self.val_result = test(best_model, self.val_dataloader,
                                feature_names=feature_names,
                                enable_detailed_analysis=True, 
                                experiment_name=f"{experiment_name}_val")

        self.get_score(self.test_result, self.val_result)

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)


        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False)

        return train_dataloader, val_dataloader

    def get_score(self, test_result, val_result):

        feature_num = len(test_result[0][0])
        np_test_result = np.array(test_result)
        np_val_result = np.array(val_result)

        test_labels = np_test_result[2, :, 0].tolist()
    
        test_scores, normal_scores = get_full_err_scores(test_result, val_result)

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1) 
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)

        print('\n' + '='*60)
        print('** FINAL RESULTS **')
        print('='*60)

        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
            print('Reporting BEST performance:')
        elif self.env_config['report'] == 'val':
            info = top1_val_info
            print('Reporting VAL-based performance:')

        print(f'F1 score: {info[0]:.6f}')
        print(f'Precision: {info[1]:.6f}') 
        print(f'Recall: {info[2]:.6f}')
        
        # Additional detailed metrics
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(test_labels)) > 1:
                auc_score = roc_auc_score(test_labels, test_scores)
                print(f'AUC Score: {auc_score:.6f}')
        except Exception as e:
            print(f'Could not calculate AUC: {e}')
        
        print(f'Dataset: {self.env_config["dataset"]}')
        print(f'Features: {len(self.feature_map)}')
        print(f'Test samples: {len(test_labels)}')
        print(f'Anomalies in test: {sum(test_labels)} ({100*sum(test_labels)/len(test_labels):.2f}%)')
        print('='*60)
        
        # Save final comprehensive results
        try:
            experiment_name = f"{self.env_config['dataset']}_seed{self.train_config['seed']}"
            results_summary = {
                'experiment_name': experiment_name,
                'dataset': self.env_config['dataset'],
                'config': self.train_config,
                'final_metrics': {
                    'f1_score': float(info[0]),
                    'precision': float(info[1]),
                    'recall': float(info[2])
                },
                'best_performance': {
                    'f1_score': float(top1_best_info[0]),
                    'precision': float(top1_best_info[1]),
                    'recall': float(top1_best_info[2])
                },
                'val_performance': {
                    'f1_score': float(top1_val_info[0]),
                    'precision': float(top1_val_info[1]),
                    'recall': float(top1_val_info[2])
                },
                'test_statistics': {
                    'total_samples': len(test_labels),
                    'anomaly_samples': sum(test_labels),
                    'anomaly_ratio': sum(test_labels)/len(test_labels),
                    'num_features': len(self.feature_map)
                }
            }
            
            # Save to results directory
            results_path = self.get_save_path()[1].replace('.csv', '_final_summary.json')
            with open(results_path, 'w') as f:
                import json
                json.dump(results_summary, f, indent=2)
            print(f'Final results summary saved to: {results_path}')
            
        except Exception as e:
            print(f'Warning: Could not save results summary: {e}')


    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr          

        paths = [
            f'./pretrained/{dir_path}/best_{datestr}.pt',
            f'./results/{dir_path}/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type = int, default=128)
    parser.add_argument('-epoch', help='train epoch', type = int, default=100)
    parser.add_argument('-slide_win', help='slide_win', type = int, default=15)
    parser.add_argument('-dim', help='dimension', type = int, default=64)
    parser.add_argument('-slide_stride', help='slide_stride', type = int, default=5)
    parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='')
    parser.add_argument('-dataset', help='wadi / swat', type = str, default='wadi')
    parser.add_argument('-device', help='cuda / cpu', type = str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type = int, default=0)
    parser.add_argument('-comment', help='experiment comment', type = str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=256)
    parser.add_argument('-decay', help='decay', type = float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type = float, default=0.1)
    parser.add_argument('-topk', help='topk num', type = int, default=20)
    parser.add_argument('-report', help='best / val', type = str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='')
    
    # Enhanced output and visualization options
    parser.add_argument('-enable_detailed_analysis', help='enable detailed anomaly analysis', type=bool, default=True)
    parser.add_argument('-enable_visualizations', help='enable result visualizations', type=bool, default=True)
    parser.add_argument('-log_level', help='logging level (INFO/DEBUG)', type=str, default='INFO')
    parser.add_argument('-save_intermediate_results', help='save intermediate training results', type=bool, default=True)
    parser.add_argument('-visualization_interval', help='epochs between visualization updates', type=int, default=10)

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)


    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
        # Enhanced analysis options
        'enable_detailed_analysis': args.enable_detailed_analysis,
        'enable_visualizations': args.enable_visualizations,
        'log_level': args.log_level,
        'save_intermediate_results': args.save_intermediate_results,
        'visualization_interval': args.visualization_interval,
    }

    env_config={
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path,
        # Enhanced analysis options
        'enable_detailed_analysis': args.enable_detailed_analysis,
        'enable_visualizations': args.enable_visualizations,
    }
    

    main = Main(train_config, env_config, debug=False)
    main.run()





