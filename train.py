import torch
import os
import argparse
from collections import defaultdict

from datasets.utils import create_datasets
from models.model import save_model, load_model
from models.train import train_models
from models.utils import set_random_seed

if __name__ == '__main__':
    # Create configs
    parser = argparse.ArgumentParser(description='Train FixMatch model.')
    parser.add_argument('--purpose', type=str, default='fixmatch', help='model type')
    parser.add_argument('--images_dir', type=str, default='./data/CXR', help='input images data directory')
    parser.add_argument('--data_dir', type=str, default='./data/CXR', help='input data directory')
    parser.add_argument('--num_labeled', type=int, default=25, help='the number of labeled data per class')
    parser.add_argument('--mu', type=int, default=4, help='ratio for unlabeled data')
    parser.add_argument('--fold', type=int, default=5, help='the number of sampled dataset')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='batch size')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed')
    parser.add_argument('--overwrite', action='store_true', help='whether overwrite text files for training and test')
    parser.add_argument('--print_to_file', action='store_true', help='whether all prints are going to write in th file or not')
    parser.add_argument('--metric_types', type=str, nargs='+', default=['acc', 'ppv', 'recall', 'f1'], help='metric types')
    parser.add_argument('--dataset_types', type=str, nargs='+', default=['train', 'test'], help='dataset types')
    parser.add_argument('--is_finetuning',action = 'store_true',help='whether fintunning or transfer learning')
    cfg = vars(parser.parse_args())
    # Set the random seed
    random_seed = cfg['random_seed']
    set_random_seed(random_seed)
    
    # Set print's output stream to the file
    if cfg['print_to_file']:
        from utils import init_file_for_print
        init_file_for_print(cfg)

    # Create datasets
    if cfg['images_dir']==cfg['data_dir']:
        create_datasets(cfg)
    else:
        create_datasets(cfg,cfg['images_dir'])

    # Train the models (index will be used for distributed TPUs)
    trained_models = train_models(index=None, cfg=cfg)
    print('Training is completed.')
    
    # Save the models
    out_dir = f'trained_models/{cfg["purpose"]}'
    save_model(trained_models, out_dir, cfg)
    # trained_models = load_model(cfg)