import torch
import torchvision
import numpy as np
import os

from torchvision import datasets, transforms
from matplotlib import pyplot as plt

def get_data_transforms():
    '''Data augmentation and normalization for training'''
    return {
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

def get_data_loaders(data_transforms, fold, batch_size, dataset_types, data_dir, lb_partial_dir):
    data_loaders = []
    for i in range(fold):
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x if x != 'train'
                                                                           else f'{lb_partial_dir}{i}'),
                                                  data_transforms[x])
                          for x in dataset_types}
        data_loaders.append({x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                            shuffle=True, num_workers=4)
                            for x in dataset_types})

    dataset_sizes = {x: len(image_datasets[x]) for x in dataset_types}
    class_names = image_datasets['train'].classes
    
    return data_loaders, dataset_sizes, class_names

def show_samples(data_loaders, class_names, dataset_type='train'):
    """Get a batch of training data and make a grid from batch"""
    
    inputs, classes = next(iter(data_loaders[0][dataset_type]))
    inputs, classes = inputs[:4], classes[:4]
    inp = torchvision.utils.make_grid(inputs)
    title = [class_names[x] for x in classes]
    
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated