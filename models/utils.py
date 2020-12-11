import torch
import numpy as np
import random

def set_random_seed(random_seed):
    '''Set random seed for torch, cuda, cudnn, numpy and random.
       This is "referenced by https://hoya012.github.io/blog/reproducible_pytorch/".
    
    Args:
        random_seed (int): random seed for torch, cuda, cudnn, numpy and random
    Returns:
        nothing
    '''
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
#     torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
