import torch
import torchvision
import numpy as np
import shutil
import glob
import os
import random
from torchvision import datasets
from matplotlib import pyplot as plt
from datasets.dataloader import CovidDataLoader
from torch.utils.data import DataLoader

def separate_datasets(data_dir, fold, labeled_num_per_cls, mu,outpath = './data/CXR',class_name=None):
    '''
    Split the dataset randomly by 'fold' parameter.
    Args:
        data_dir: Root path of dataset. ex) ./data/CXR or ./data/CT
        fold: Number of times to split data
        labeled_num_per_cls, mu: Number of labeled image per class. Number of unlabeld image will be labeled_num_per_cls * mu per class.
        outpath: Root path where the text file(train_lb_%.txt, train_ulb_%.txt) will be saved
    '''

    if class_name == None:
        class_name = ['covid-19', 'pneumonia', 'normal']

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    image_paths = [glob.glob(os.path.join(data_dir,'train',cls_name,'*')) for cls_name in class_name]
    unlabeled_image_num = labeled_num_per_cls * len(class_name) * mu # unlabeled_num = labeled_num * mu = labeled_num_per_cls * len(class_name) * mu

    for i, cls_name in enumerate(class_name):  # Check folder.
        assert len(image_paths[i]), '{} does not have a {} folder'.format(os.path.join(data_dir, 'train'),cls_name)

    for n in range(fold): # Make txt file.
        all_image_paths, labeled_image_paths = [], []
        for im_path in image_paths:
            all_image_paths += im_path
            labeled_image_paths.append(np.random.choice(im_path,labeled_num_per_cls,replace=False)) # Random choice image per class.

        file_name = os.path.join(outpath , f'train_lb_{n}.txt')
        with open(file_name, 'w') as f:
            for i in range(labeled_num_per_cls):
                for j,cls_name in enumerate(class_name):
                    f.writelines(labeled_image_paths[j][i] + " " + cls_name+ "\n")
                    all_image_paths.remove(labeled_image_paths[j][i]) # Delete to avoid duplication.
        print('"train_lb_{}.txt" created in {}'.format(n,outpath))

        unlabeled_image_paths = np.random.choice(all_image_paths,unlabeled_image_num,replace=False)
        file_name = os.path.join(outpath, f'train_ulb_{n}.txt')
        with open(file_name, 'w') as f:
            for unlabeled_image in unlabeled_image_paths:
                f.writelines(unlabeled_image + "\n")
        print('"train_ulb_{}.txt" created in {}'.format(n, outpath))
    return True

def make_baseline_dataset(data_dir,labeled_num_per_cls=None,outpath = './data/CXR', dataset_types = None, class_name = None):
    '''
    Make baseline dataset text file
    Args:
        data_dir: Root path of dataset. ex) ./data/CXR or ./data/CT
        labeled_num_per_cls: Number of image per class. It is only used for train dataset
                            if None, The dataset is created over the entire data.
        outpath: Root path where the text file(train.txt, test.txt) will be saved.
                The file name will be dataset_type.txt .
        dataset_types: Type of dataset. ex) ['train', 'test', 'val']
        class_name : It means the subfolder name created with class name. ex)['covid-19', 'pneumonia', 'normal']
    '''

    if dataset_types == None:
        dataset_types = ['train', 'test']

    if class_name == None:
        class_name = ['covid-19', 'pneumonia', 'normal']

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    for dataset_type in dataset_types:
        image_paths = [glob.glob(os.path.join(data_dir,dataset_type,cls_name,'*')) for cls_name in class_name]

        for i, cls_name in enumerate(class_name): # Check folder.
            assert len(image_paths[i]),'{} does not have a {} folder'.format(os.path.join(data_dir,dataset_type),cls_name)

        if dataset_type=='train' and labeled_num_per_cls != None: # In the case of train, randomly extracted as much as labeled_num_per_cls.
            image_paths = [np.random.choice(im_path, labeled_num_per_cls , replace=False) for im_path in image_paths]
        elif dataset_type=='train':
            print('The dataset is created over the entire data.')

        with open(os.path.join(outpath, '{}.txt'.format(dataset_type)), 'w') as f: # Make txt file.
            for i, cls_name in enumerate(class_name):
                for im_path in image_paths[i]:
                    f.writelines(im_path + " " + cls_name+ "\n")
        print('"{}.txt" created in {}'.format(dataset_type,outpath))
    return True

def create_datasets(cfg):
    '''Create datasets into baseline or the seperated type.
    
    Args:
        cfg (dict): The cfg parameter must have purpose, data dir information and mu
                    (Used only when purpose is not baseline).
    Returns:
        nothing
    '''
    # Make baseline datasets
    if not os.path.exists(os.path.join(cfg['data_dir'],'test.txt')):
        from datasets.utils import make_baseline_dataset
        make_baseline_dataset(cfg['data_dir'], cfg['num_labeled'],
                              outpath=cfg['data_dir']) # test는 전부, train은 25개 만큼만
        
    # Make separated datasets
    if cfg['purpose'] == 'fixmatch' \
    and not os.path.exists(os.path.join(cfg['data_dir'],'train_lb_0.txt')):
        from datasets.utils import separate_datasets
        separate_datasets(cfg['data_dir'], cfg['fold'], cfg['epochs'],
                          cfg['mu'],outpath=cfg['data_dir']) # lb는 25개, ulb는 mu*25개
        
def get_data_loaders(dataset_type, cfg, dataset_sizes={}, data_loaders={}, fold_id=None, overwrite=False):
    '''Return data loaders.
    
    Args:
        dataset_type (str): type of dataset. ex) 'train' or 'test'
        cfg (dict): The cfg parameter must have purpose, data dir information and mu
                    (Used only when purpose is not baseline).
        dataset_sizes (dict): the number of images for each class in speicific dataset type.
        data_loaders (dict): data loaders which are already difined
        fold_id (int): fold index for FixMatch
    Returns:
        data_loaders (dict): data loaders for the dataset type
        dataset_sizes (dict): the number of images for each class in speicific dataset type.
        class_names (list): class names for the dataset type
    '''
    # Remove fold_id for baseline
    if cfg['purpose'] == 'baseline':
        fold_id = None
        
    # Create CovidDataLoader
    covid_data_loader = CovidDataLoader(dataset_type, cfg, fold_id=fold_id)
    dataset_sizes[dataset_type] = len(covid_data_loader)
    class_names = list(covid_data_loader.class_names.values())
    
    # Create torch DataLoader
    data_loader = DataLoader(covid_data_loader, batch_size=cfg['batch_size'], num_workers=4, shuffle=True,
                             collate_fn=covid_data_loader.collate_fn)
    data_loaders[dataset_type] = data_loader
    
    return data_loaders, dataset_sizes, class_names

def preprocess(class_names,inputs, classes=None, mu=4):
    '''
    Args:
        class_names: Recommend {0:'Covid-19',1:'Pneumonia',2:'Normal'}
        inputs: Data of image
        classes: Data of label
        mu:

    Returns:
        inp : Preprocessed image (0.0~1.0)
        title : Label data in string format
    '''
    inputs = inputs[:mu]
    inp = torchvision.utils.make_grid(inputs)

    if classes is not None:
        classes = classes[:mu]
        title = [class_names[x] for x in classes.numpy()]
    else:
        title = None

    # Plot the samples
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp, title


def show_samples(data_loader, class_names, iter_num=1, mu=4):
    '''
    This function is used to check if the data loader properly loaded the image.
    Args:
        data_loader: CovidDataLoader class
        class_names: Recommend {0:'covid-19', 1:'pneumonia', 2:'normal'}
        iter_num: This parameter is for how many iterations to check based on batch size 1.
        mu: If the data loader has 2 key values, it is not used.
    Returns:

    '''
    for i, batch in enumerate(data_loader):
        img, labels = preprocess(class_names,batch['img_lb'], batch['label'],mu=mu)

        # Dataset which have labeled and unlabeled data concurrently
        if len(batch.keys()) != 2:
            fig, axes = plt.subplots(nrows=1, ncols=3)
            fig.set_figheight(13)
            fig.set_figwidth(13)
            axes[0].set_title(labels)
            axes[0].imshow(img)

            im_ulb, _ = preprocess(class_names,batch['img_ulb'],mu=mu)
            axes[1].set_title('ulb')
            axes[1].imshow(im_ulb)

            im_ulb_wa, _ = preprocess(class_names,batch['img_ulb_wa'],mu=mu)
            axes[2].set_title('ulb_wa')
            axes[2].imshow(im_ulb_wa)
            fig.tight_layout()

            plt.show()
            
        # Dataset which have labeled data only
        else:
            plt.imshow(img)
            plt.xlabel(labels)
            plt.show()
        if i == iter_num - 1:
            break

    return