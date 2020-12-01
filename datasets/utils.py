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

def separate_datasets(data_dir, fold, labeled_num_per_cls, mu,outpath = './data/CXR'):
    class_name = ['covid-19','pneumonia','normal']

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    image_paths = [glob.glob(os.path.join(data_dir,'train',cls_name,'*')) for cls_name in class_name]
    for n in range(fold):
        images = [np.random.choice(im_path,labeled_num_per_cls+labeled_num_per_cls*mu,replace=False) for im_path in image_paths]
        file_name = os.path.join(outpath , f'train_lb_{n}.txt')
        with open(file_name, 'w') as f:
            for i in range(labeled_num_per_cls):
                for j,cls_name in enumerate(class_name):
                    f.writelines(images[j][i] + " " + cls_name+ "\n")
        print('"train_lb_{}.txt" created in {}'.format(n,outpath))
        file_name = os.path.join(outpath, f'train_ulb_{n}.txt')
        with open(file_name, 'w') as f:
            for i in range(labeled_num_per_cls,labeled_num_per_cls*(mu+1)):
                for j, cls_name in enumerate(class_name):
                    f.writelines(images[j][i] + "\n")
        print('"train_ulb_{}.txt" created in {}'.format(n, outpath))
    return True

def make_baseline_dataset(data_dir,labeled_num_per_cls,outpath = './data/CXR'):
    class_name = ['covid-19', 'pneumonia', 'normal']

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    images = [glob.glob(os.path.join(data_dir,'train',cls_name,'*')) for cls_name in class_name]
    images = [np.random.choice(im_path, labeled_num_per_cls , replace=False) for
              im_path in images]

    with open(os.path.join(outpath, 'train.txt'), 'w') as f:
        for i in range(labeled_num_per_cls):
            for j, cls_name in enumerate(class_name):
                f.writelines(images[j][i] + " " + cls_name+ "\n")
    print('"train.txt" created in {}'.format(outpath))
    images = [glob.glob(os.path.join(data_dir, 'test', cls_name, '*')) for cls_name in class_name]

    with open(os.path.join(outpath, 'test.txt'), 'w') as f:
        for j, cls_name in enumerate(class_name):
            for i in range(len(images[j])):
                f.writelines(images[j][i] + " " + cls_name + "\n")
    print('"test.txt" created in {}'.format(outpath))
    return True

class Data_loader(torch.utils.data.Dataset):
    def __init__(self,dataset_types,cfg,fold_id=None):
        self.type= dataset_types
        self.cfg = cfg
        self.class_names = {0:'Covid-19',1:'Normal',2:'Pneumonia'}
        self.name2label = {'covid-19':0,'normal':1,'pneumonia':2}
        self.transformer = get_data_transforms(cfg['purpose'])
        if 'train' not in dataset_types:
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

        if self.cfg['purpose']=='baseline' or 'train' not in self.type:
            img_lb = self.transformer['{}'.format(self.type)](img_lb)
            return {'img_lb':img_lb,'label': label}
        else:
            img_lb = self.transformer['{}'.format('train_lb')](img_lb)

            img_unlabel = [self.load_image(self.image_ulb_paths[i]) for i in range(idx*self.cfg['mu'],(idx+1)*self.cfg['mu'])]
            img_ulb = torch.cat([self.transformer['train_ulb'](img_u).unsqueeze(0) for img_u in img_unlabel.copy()],0)
            img_ulb_wa =torch.cat([self.transformer['train_ulb_wa'](img_u).unsqueeze(0) for img_u in img_unlabel.copy()],0)
            return {'img_lb': img_lb, 'label': label,'img_ulb':img_ulb,'img_ulb_wa':img_ulb_wa}

    @staticmethod
    def collate_fn(batch):
        if 'img_ulb' in batch[0].keys():
            new_batch = {'img_lb' : torch.stack([b['img_lb'] for b in batch],0),
                         'label' : torch.cat([b['label'] for b in batch],0),
                         'img_ulb' : torch.cat([b['img_ulb'] for b in batch],0),
                         'img_ulb_wa' : torch.cat([b['img_ulb_wa'] for b in batch],0),}
        else:
            new_batch = {'img_lb': torch.stack([b['img_lb'] for b in batch], 0),
                         'label': torch.cat([b['label'] for b in batch], 0)}
        return new_batch


def preprocess(class_names,inputs, classes=None, mu=4):
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