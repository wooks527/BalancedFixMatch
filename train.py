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
    parser.add_argument('--data_dir', type=str, default='./data/CXR', help='input data directory')
    parser.add_argument('--num_labeled', type=int, default=25, help='the number of labeled data')
    parser.add_argument('--mu', type=int, default=4, help='ratio for unlabeled data')
    parser.add_argument('--fold', type=int, default=5, help='the number of sampled dataset')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='batch size')
    parser.add_argument('--metric_types', type=str, nargs='+', default=['acc', 'ppv', 'recall', 'f1'], help='metric types')
    parser.add_argument('--dataset_types', type=str, nargs='+', default=['train', 'test'], help='dataset types')
    cfg = vars(parser.parse_args())

    # Set the random seed
    random_seed = 0
    set_random_seed(random_seed)

    # Create datasets
    create_datasets(cfg)
    
    # Train the models
    trained_models = train_models(cfg)
    
    # Save the models
    out_dir = f'trained_models/{cfg["purpose"]}'
    save_model(trained_models, out_dir, cfg)
    # trained_models = load_model(cfg)