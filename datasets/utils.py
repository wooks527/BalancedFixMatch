import torch
import torchvision
import numpy as np
import shutil
import glob
import os
import random

from torchvision import datasets, transforms
from matplotlib import pyplot as plt

def get_data_transforms(purpose='baseline'):
    '''Data augmentation and normalization
    
    Args:
        purpose (str): the purpose of the model
    
    Returns:
        data_transforms (dict): transformation methods for train, validation and test
        
    '''
    if purpose == 'baseline':
        data_trainsforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    else: # FixMatch
        data_transforms = {
            'train_lb': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'train_ulb': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            # This is based on COVIDNet settings.
            'train_ulb_wa': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomAffine(degrees=10, translate=(0, 0.1), scale=(0.85, 1.15)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=(0.9, 1.1)), #, contrast=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
    return data_transforms

def separate_datasets(data_dir, fold, labeled_num_per_cls, mu):
    '''Separate train datasets into labeled and unlabeled datasets.
    
    Args:
        data_dir (str): the base directory path for the datasets
        fold (int): the number of models which will be trained and tested
        labeled_num_per_cls (int): the number of labeled data in each class
        mu (int): the rate of unlabeled data to labeled data (e.g. labeled:unlabeled = 1:4 -> mu = 4)
        
    Returns:
        nothing
    '''
    # Remove and create directories
    for n in range(fold):
        if os.path.isdir(f'{data_dir}/train_lb/train_lb_{n}'):
            shutil.rmtree(f'{data_dir}/train_lb/train_lb_{n}')
        os.makedirs(f'{data_dir}/train_lb/train_lb_{n}/covid-19')
        os.makedirs(f'{data_dir}/train_lb/train_lb_{n}/pneumonia')
        os.makedirs(f'{data_dir}/train_lb/train_lb_{n}/normal')

        if os.path.isdir(f'{data_dir}/train_ulb/train_ulb_{n}'):
            shutil.rmtree(f'{data_dir}/train_ulb/train_ulb_{n}')
        os.makedirs(f'{data_dir}/train_ulb/train_ulb_{n}/covid-19')
        os.makedirs(f'{data_dir}/train_ulb/train_ulb_{n}/pneumonia')
        os.makedirs(f'{data_dir}/train_ulb/train_ulb_{n}/normal')

        if os.path.isdir(f'{data_dir}/train_ulb/train_ulb_wa_{n}'):
            shutil.rmtree(f'{data_dir}/train_ulb/train_ulb_wa_{n}')
        os.makedirs(f'{data_dir}/train_ulb/train_ulb_wa_{n}/covid-19')
        os.makedirs(f'{data_dir}/train_ulb/train_ulb_wa_{n}/pneumonia')
        os.makedirs(f'{data_dir}/train_ulb/train_ulb_wa_{n}/normal')
        
    # Copy train datasets into labeled and unlabeled directories
    for n in range(fold):
        for img_cls in ('covid-19', 'pneumonia', 'normal'):
            images = glob.glob(f'{data_dir}/train/{img_cls}/*')
            
            # For labeled data
            samples_lb = set(random.sample(images, labeled_num_per_cls))
            for s_img in samples_lb:
                dst_img = s_img.replace('/train/', f'/train_lb/train_lb_{n}/')
                shutil.copyfile(s_img, dst_img)

            # For unlabeled data
            samples_ulb = random.sample(list(filter(lambda ele: ele not in samples_lb, images)), labeled_num_per_cls*mu)
            for s_img in samples_ulb:
                dst_img1 = s_img.replace('/train/', f'/train_ulb/train_ulb_{n}/')
                dst_img2 = s_img.replace('/train/', f'/train_ulb/train_ulb_wa_{n}/')
                shutil.copyfile(s_img, dst_img1)
                shutil.copyfile(s_img, dst_img2)

def get_data_loaders(data_transforms, fold, batch_size, dataset_types, data_dir, lb_partial_dir, purpose='baseline', mu=None):
    '''Create and return data loaders applied transformations, the batch size and so on.
    
    Args:
        data_transforms (dict): transformation methods for train, validation and test
        fold (int): the number of models which will be trained and tested
        batch_size (int): the batch size
        dataset_types (list): dataset types for train and test (e.g. ['train', 'test'] or ['train', 'val', 'test])
        data_dir (str): the base directory path for the datasets
        lb_partial_dir (str): the partial directory path according to datase types or mu
        purpose (str): the purpose of the model
        mu (int): the rate of unlabeled data to labeled data (e.g. labeled:unlabeled = 1:4 -> mu = 4)
        
    Returns:
        data_loaders (list): data loaders applied transformations, the batch size and so on
        dataset_sizes (dict): sizes of train and test datasets
        class_names (list): class names for the dataset
    '''
    f_name = {'train_lb': 'train_lb', 'train_ulb': 'train_ulb', 'train_ulb_wa': 'train_ulb'} # just for fixmatch
    data_loaders = []
    for i in range(fold):
        if purpose == 'baseline':
            image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x if x != 'train'
                                                                               else f'{lb_partial_dir}{i}'),
                                                      data_transforms[x])
                              for x in dataset_types}
            data_loaders.append({x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                                shuffle=True, num_workers=4)
                                for x in dataset_types})
        else: # FixMatch
            image_datasets = {x if 'train' not in x else x[:-2]: datasets.ImageFolder(os.path.join(data_dir,
                                                                                                   x if 'train' not in x
                                                                                                   else f'{f_name[x[:-2]]}/{x}/'),
                                                                 data_transforms[x if 'train' not in x else x[:-2]])
                              for x in [f'train_lb_{i}', f'train_ulb_{i}', f'train_ulb_wa_{i}', 'test']}
            data_loaders.append({x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size if 'ulb' not in x
                                                                                                         else batch_size*mu,
                                                                shuffle=True, num_workers=4)
                                 for x in ['train_lb', 'train_ulb', 'train_ulb_wa', 'test']})
            
    if purpose == 'baseline':
        dataset_sizes = {x: len(image_datasets[x]) for x in dataset_types}
        class_names = image_datasets['train'].classes
    else: # FixMatch
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train_lb', 'train_ulb', 'train_ulb_wa', 'test']}
        class_names = image_datasets['train_lb'].classes
    
    return data_loaders, dataset_sizes, class_names

def show_samples(data_loaders, class_names, dataset_type='train'):
    '''Get a batch of training data and make a grid from batch
    
    Args:
        data_loaders (list): data loaders applied transformations, the batch size and so on
        class_names (list): class names for the dataset
        dataset_type (str): the dataset type which show samples
        
    Returns:
        nothing
    '''
    
    # Extract 4 samples from first data loader
    inputs, classes = next(iter(data_loaders[0][dataset_type]))
    inputs, classes = inputs[:4], classes[:4]
    inp = torchvision.utils.make_grid(inputs)
    title = [class_names[x] for x in classes]
    
    # Plot the samples
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated