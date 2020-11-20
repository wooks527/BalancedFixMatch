import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models

def get_model(device, fine_tuning=True):
    '''Create and return the model based on ResNet-50.
    
    Args:
        device (obj): the device where the model will be trained (e.g. cpu or gpu)
        fine_tuning (bool): whether fine-tuning or not
        
    Returns:
        model_ft (obj): the model which will be trained
        criterion (obj): the loss function (e.g. cross entropy)
        optimizer_ft (obj): the optimizer (e.g. Adam)
        exp_lr_scheduler (obj): the learning scheduler (e.g. Step decay)
    '''
    # Get the pre-trained model
    model_ft = models.resnet50(pretrained=True)
    for param in model_ft.parameters():
        param.requires_grad = fine_tuning
        
    # Change fully connected layer
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 3)
    model_ft = model_ft.to(device)
    
    # Set loss function, optimizer and learning scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    return model_ft, criterion, optimizer_ft, exp_lr_scheduler