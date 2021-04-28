from torchvision import transforms
from datasets.randaugment import *

def get_data_transforms(purpose='baseline'):
    '''Data augmentation and normalization
    
    Args:
        purpose (str): the purpose of the model
    
    Returns:
        data_transforms (dict): transformation methods for train, validation and test
        
    '''
    if purpose == 'baseline':
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomAffine(degrees=0, translate=(0.125, 0.125)),
                transforms.RandomHorizontalFlip(p=0.5),
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
    elif purpose == 'fixmatch': # FixMatch
        data_transforms = {
            'train_lb': transforms.Compose([
                transforms.RandomAffine(degrees=0, translate=(0.125, 0.125)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'train_ulb': transforms.Compose([
                transforms.RandomAffine(degrees=0, translate=(0.125, 0.125)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'train_ulb_wa': transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
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
        data_transforms['train_ulb_wa'].transforms.insert(0,RandAugment(5, purpose='focal_loss'))
        
    return data_transforms