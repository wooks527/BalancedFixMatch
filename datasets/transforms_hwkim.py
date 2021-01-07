from torchvision import transforms
from datasets.randaugment import *
from itertools import permutations

def get_data_transforms(purpose='baseline', baseline_flag=0, num_labeled=50):
    '''Data augmentation and normalization
    
    Args:
        purpose (str): the purpose of the model
    
    Returns:
        data_transforms (dict): transformation methods for train, validation and test
        
    '''
    # TODO: Add transformation methods which wasn't added below.
    transforms_dict = {
        '1': transforms.ColorJitter(brightness=(0.9, 1.1)),
        '2': transforms.ColorJitter(brightness=(0.8, 1.2)),
        '3': transforms.ColorJitter(brightness=(0.7, 1.3)),
        '4': transforms.ColorJitter(contrast=(0.9, 1.1)),
        '5': transforms.ColorJitter(contrast=(0.8, 1.2)),
        '6': transforms.ColorJitter(contrast=(0.7, 1.3)),
        '7': transforms.RandomAffine(degrees=(-5, 5)),
        '8': transforms.RandomAffine(degrees=(-10, 10)),
        '9': transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        '10': transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
        '11': transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        '12': transforms.RandomAffine(degrees=0, scale=(0.95, 1.05)),
        '13': transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        '14': transforms.RandomAffine(degrees=0, scale=(0.85, 1.15)),
        '15': transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
        '16': transforms.RandomHorizontalFlip(p=0.5),
        '17': transforms.Normalize([0.493, 0.493, 0.493], [0.246, 0.246, 0.246]),
        '18': transforms.Resize(272),
        '19': transforms.Resize(288),
        '20': transforms.Resize(304),
    }

    if purpose == 'baseline':
        data_transforms = {
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
        if baseline_flag == '0':
            data_transforms['train'] = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            if ',' not in baseline_flag: # Single case
                if baseline_flag == '17': # Normalization mean/std change
                    for dtype in ('train', 'val', 'test'):
                        data_transforms[dtype] = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms_dict[baseline_flag]
                        ])
                elif baseline_flag == '18' or baseline_flag == '19' or baseline_flag == '20': # Resize
                    data_transforms['train'] = transforms.Compose([
                        transforms_dict[baseline_flag],
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                else: # baseline_flag: 1 ~ 16
                    data_transforms['train'] = transforms.Compose([
                        transforms_dict[baseline_flag],
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
            else: # Multiple case without normalization cases
                bflags = baseline_flag.split(',')
                final_transforms = [transforms_dict[bflag] for bflag in bflags]
                final_transforms.extend([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                data_transforms['train'] = transforms.Compose(final_transforms)
                
    elif purpose == 'fixmatch': # FixMatch
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