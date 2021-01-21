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
    parser.add_argument('--out_dir', type=str, default='./trained_models', help='output directory for trained models')
    parser.add_argument('--num_labeled', type=int, default=25, help='the number of labeled data per class')
    parser.add_argument('--scheduler', type=str, default='step', help='learning scheduler for training')
    parser.add_argument('--mu', type=int, default=4, help='ratio for unlabeled data')
    parser.add_argument('--lambda_u', type=float, default=1.0, help='ratio which reflect the unlabeled loss')
    parser.add_argument('--threshold', type=float, default=1.0, help='threshold for the pseudo label')
    parser.add_argument('--fold', type=int, default=5, help='the number of sampled dataset')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='batch size')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed')
    parser.add_argument('--overwrite', action='store_true', help='whether overwrite text files for training and test')
    parser.add_argument('--print_to_file', action='store_true', help='whether all prints are going to write in th file or not')
    parser.add_argument('--use_tpu', action='store_true', help='whether you use tpus or not')
    parser.add_argument('--metric_types', type=str, nargs='+', default=['acc', 'ppv', 'recall', 'f1'], help='metric types')
    parser.add_argument('--dataset_types', type=str, nargs='+', default=['train', 'test'], help='dataset types')
    parser.add_argument('--freeze_conv',action = 'store_true',help='whether fintunning or transfer learning')
    parser.add_argument('--is_old_optimizer',action = 'store_true',help='whether old optimizer or new')
    parser.add_argument('--lr', type=float, default=0.001, help = 'Learning rate of optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help = 'Momentum of optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help = 'Weight decay of optimizer')
    parser.add_argument('--baseline_flag', type=str, default='0', help='Baseline flag for various transformations')
    parser.add_argument('--focal_loss', action = 'store_true', help='whether you use focal loss or previous loss')
    parser.add_argument('--sharpening', action = 'store_true', help='whether you use sharpening or pseudo label')
    parser.add_argument('--gamma', type=float, default=1.0, help='gamma value for the focal loss')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature value for the sharpening')
    parser.add_argument('--opt', type=str, default='SGD', help='optimizer')
    parser.add_argument('--nestrov', type=bool, default=True, help='SGD option')
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
    train_models(index=None, cfg=cfg)
    print('Training is completed.')
