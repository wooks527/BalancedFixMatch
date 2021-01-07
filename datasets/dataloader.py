import torch
import os
from PIL import Image
# from datasets.transforms import get_data_transforms
from datasets.transforms_hwkim import get_data_transforms


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
        self.class_names = {0:'covid-19',1:'pneumonia',2:'normal'}
        self.name2label = {'covid-19':0,'pneumonia':1,'normal':2}
        # self.transformer = get_data_transforms(cfg['purpose'])
        self.transformer = get_data_transforms(cfg['purpose'], cfg['baseline_flag'])
        if 'train' != dataset_types:
            self.image_lb_paths,self.labels = self.load_text(os.path.join(cfg['data_dir'],'{}.txt'.format(dataset_types)))
            return
        elif cfg['purpose'] =='baseline':
            if fold_id==None:
                print('The data loader did not receive any information about the fold id.\nSo the dataset is loaded over the entire train data.')
                self.image_lb_paths,self.labels = self.load_text(os.path.join(cfg['data_dir'], 'train.txt'))
            else:
                self.image_lb_paths,self.labels = self.load_text(os.path.join(cfg['data_dir'] , f'train_lb_{fold_id}.txt'))
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
