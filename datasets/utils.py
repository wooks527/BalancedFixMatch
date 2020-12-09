import torch
import torchvision
import numpy as np
import shutil
import glob
import os
import random
from datasets.transforms import get_data_transforms
from torchvision import datasets
from matplotlib import pyplot as plt
from PIL import Image

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
    inlabeled_image_num = labeled_num_per_cls * mu * len(image_paths)

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

        unlabeled_image_paths = np.random.choice(all_image_paths,inlabeled_image_num,replace=False)
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
        elif dataset_types=='train':
            print('The dataset is created over the entire data.')

        with open(os.path.join(outpath, '{}.txt'.format(dataset_type)), 'w') as f: # Make txt file.
            for i in range(labeled_num_per_cls):
                for j, cls_name in enumerate(class_name):
                    f.writelines(image_paths[j][i] + " " + cls_name+ "\n")
        print('"{}.txt" created in {}'.format(dataset_type,outpath))
    return True

class CovidDataLoader(torch.utils.data.Dataset):
    '''
    Custom data loader for covid data.
    Return:
        new_batch['img_lb'] = [b,224,224,3]
        new_batch['label'] = [b]
        new_batch['img_ulb'] = [b*mu,224,224,3] (Return only when purpose is not baseline)
        new_batch['img_ulb_wa'] = [b*mu,224,224,3] (Return only when purpose is not baseline)
    '''
    def __init__(self,dataset_types,cfg,fold_id=None):
        '''
        Args:
            dataset_types: It distinguishes whether it is a train or a test through the corresponding parameter.
            cfg: The cfg parameter must have purpose, data dir information and mu (Used only when purpose is not baseline).
            fold_id: It is used when the purpose is fixmatch and means fold number.
                    This information is used to read the appropriate txt file.
        '''
        self.type= dataset_types
        self.cfg = cfg
        self.class_names = {0:'Covid-19',1:'Pneumonia',2:'Normal'}
        self.name2label = {'covid-19':0,'pneumonia':1,'normal':2}
        self.transformer = get_data_transforms(cfg['purpose'])
        if 'train' != dataset_types:
            self.image_lb_paths,self.labels = self.load_text(os.path.join(cfg['data_dir'],'{}.txt'.format(dataset_types)))
            return
        elif cfg['purpose'] =='baseline':
            self.image_lb_paths,self.labels = self.load_text(os.path.join(cfg['data_dir'],'train.txt'))
        else:
            assert fold_id!=None,'No fold_id was received.'
            self.image_lb_paths,self.labels = self.load_text(os.path.join(cfg['data_dir'] , f'train_lb_{fold_id}.txt'))
            self.image_ulb_paths = self.load_text(os.path.join(cfg['data_dir'] , f'train_ulb_{fold_id}.txt'),is_labeld=False)

    def load_text(self,txt_path,is_labeld=True):
        f = open(txt_path,'r')
        if is_labeld:
            paths,labels = [],[]
            for l in f.readlines():
                path,label = l.strip('\n').split()
                paths.append(path)
                labels.append(self.name2label[label])
            f.close()
            return paths,labels
        else:
            lines = f.readlines()
            f.close()
            return [l.strip('\n') for l in lines]

    def load_image(self,path):
        img = Image.open(path)
        if img.mode !='RGB':
            return img.convert('RGB')
        return img

    def __len__(self):
        return len(self.image_lb_paths)

    def __getitem__(self, idx):
        img_lb = self.load_image(self.image_lb_paths[idx])
        label = torch.LongTensor([self.labels[idx]])

        if self.cfg['purpose']=='baseline' or 'train' != self.type: # for baseline, test
            img_lb = self.transformer['{}'.format(self.type)](img_lb)
            return {'img_lb':img_lb,'label': label}
        else:
            img_lb = self.transformer['{}'.format('train_lb')](img_lb) # for fixmatch

            img_unlabel = [self.load_image(self.image_ulb_paths[i]) for i in range(idx*self.cfg['mu'],(idx+1)*self.cfg['mu'])]
            img_ulb = torch.cat([self.transformer['train_ulb'](img_u).unsqueeze(0) for img_u in img_unlabel.copy()],0)
            img_ulb_wa =torch.cat([self.transformer['train_ulb_wa'](img_u).unsqueeze(0) for img_u in img_unlabel.copy()],0)
            return {'img_lb': img_lb, 'label': label,'img_ulb':img_ulb,'img_ulb_wa':img_ulb_wa}

    @staticmethod
    def collate_fn(batch):
        if 'img_ulb' in batch[0].keys(): # for fixmatch
            new_batch = {'img_lb' : torch.stack([b['img_lb'] for b in batch],0),
                         'label' : torch.cat([b['label'] for b in batch],0),
                         'img_ulb' : torch.cat([b['img_ulb'] for b in batch],0),
                         'img_ulb_wa' : torch.cat([b['img_ulb_wa'] for b in batch],0),}
        else: # for baseline, test
            new_batch = {'img_lb': torch.stack([b['img_lb'] for b in batch], 0),
                         'label': torch.cat([b['label'] for b in batch], 0)}
        return new_batch


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
        class_names: Recommend {0:'Covid-19',1:'Pneumonia',2:'Normal'}
        iter_num: This parameter is for how many iterations to check based on batch size 1.
        mu: If the data loader has 2 key values, it is not used.
    Returns:

    '''
    for i, batch in enumerate(data_loader):
        img, labels = preprocess(class_names,batch['img_lb'], batch['label'],mu=mu)

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

        else:
            plt.imshow(img)
            plt.xlabel(labels)
            plt.show()
        if i == iter_num - 1:
            break

    return