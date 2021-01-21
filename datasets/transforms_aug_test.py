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
    elif purpose == 'fixmatch': # FixMatch (직관)
        data_transforms = {
            'train_lb': transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'train_ulb': transforms.Compose([
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
    # elif purpose == 'fixmatch': # FixMatch (nl50)
    #     data_transforms = {
    #         'train_lb': transforms.Compose([
    #             transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)), # bf13
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]),
    #         'train_ulb': transforms.Compose([
    #             transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)), # bf13
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]),
    #         'train_ulb_wa': transforms.Compose([
    #             transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)), # bf13
    #             transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # bf9
    #             transforms.RandomAffine(degrees=(-10, 10)), # bf8
    #             transforms.ColorJitter(brightness=(0.7, 1.3)), # bf3
    #             transforms.RandomHorizontalFlip(p=0.5), # bf16
    #             transforms.ColorJitter(contrast=(0.8, 1.2)), # bf5
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]),
    #         'val': transforms.Compose([
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]),
    #         'test': transforms.Compose([
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]),
    #     }
    # elif purpose == 'fixmatch': # FixMatch (nl100)
    #     data_transforms = {
    #         'train_lb': transforms.Compose([
    #             transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # bf9
    #             transforms.RandomHorizontalFlip(p=0.5),
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]),
    #         'train_ulb': transforms.Compose([
    #             transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # bf9
    #             transforms.RandomHorizontalFlip(p=0.5),
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]),
    #         'train_ulb_wa': transforms.Compose([
    #             transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # bf9
    #             transforms.RandomAffine(degrees=0, scale=(0.85, 1.15)), # bf14
    #             transforms.ColorJitter(brightness=(0.7, 1.3)), # bf3
    #             transforms.ColorJitter(contrast=(0.7, 1.3)), # bf6
    #             transforms.RandomAffine(degrees=(-5, 5)), # bf7
    #             transforms.RandomHorizontalFlip(p=0.5), # bf16
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]),
    #         'val': transforms.Compose([
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]),
    #         'test': transforms.Compose([
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]),
    #     }
    # elif purpose == 'fixmatch': # FixMatch (nl150)
    #     data_transforms = {
    #         'train_lb': transforms.Compose([
    #             transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)), # bf13
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]),
    #         'train_ulb': transforms.Compose([
    #             transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)), # bf13
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]),
    #         'train_ulb_wa': transforms.Compose([
    #             transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)), # bf13
    #             transforms.ColorJitter(brightness=(0.7, 1.3)), # bf3
    #             transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # bf9
    #             transforms.ColorJitter(contrast=(0.7, 1.3)), # bf6
    #             transforms.RandomAffine(degrees=(-10, 10)), # bf8
    #             transforms.RandomHorizontalFlip(p=0.5), # bf16
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]),
    #         'val': transforms.Compose([
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]),
    #         'test': transforms.Compose([
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]),
    #     }
    elif purpose == 'fixaug1':  # FixMatch
        data_transforms = {
            'train_lb': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomAffine(degrees=(-5,5), translate=(0.1, 0.1), scale=(0.925, 1.075)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=(0.95, 1.05)),  # , contrast=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'train_ulb': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomAffine(degrees=(-5,5), translate=(0.1, 0.1), scale=(0.925, 1.075)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=(0.95, 1.05)),  # , contrast=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            # This is based on COVIDNet settings.
            'train_ulb_wa': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomAffine(degrees=(-10,10), translate=(0.1, 0.1), scale=(0.85, 1.15)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=(0.9, 1.1)),  # , contrast=(0.9, 1.1)),
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
    elif purpose == 'fixaug2':  # FixMatch
        data_transforms = {
            'train_lb': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomAffine(degrees=(-10,10), translate=(0.1, 0.1), scale=(0.85, 1.15)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=(0.9, 1.1)),  # , contrast=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'train_ulb': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomAffine(degrees=(-10,10), translate=(0.1, 0.1), scale=(0.85, 1.15)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=(0.9, 1.1)),  # , contrast=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'train_ulb_wa': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomAffine(degrees=(-10,10), translate=(0.1, 0.1), scale=(0.85, 1.15)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=(0.9, 1.1)),  # , contrast=(0.9, 1.1)),
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
        data_transforms['train_ulb_wa'].transforms.insert(0,RandAugment(3))
    elif purpose == 'fixaug3':  # FixMatch
        data_transforms = {
            'train_lb': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomAffine(degrees=(-10,10), translate=(0.1, 0.1), scale=(0.85, 1.15)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=(0.9, 1.1)),  # , contrast=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'train_ulb': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomAffine(degrees=(-10,10), translate=(0.1, 0.1), scale=(0.85, 1.15)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=(0.9, 1.1)),  # , contrast=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'train_ulb_wa': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomAffine(degrees=(-10,10), translate=(0.1, 0.1), scale=(0.85, 1.15)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=(0.9, 1.1)),  # , contrast=(0.9, 1.1)),
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
        data_transforms['train_ulb_wa'].transforms.insert(0,RandAugment(3,purpose ='fixaug3'))

    return data_transforms