import torch
import time
import copy
import numpy as np
import torch

from collections import defaultdict
from models.metrics import update_batch_metrics, get_epoch_metrics, print_metrics

def train_model(model, criterion, optimizer, scheduler, i, cls_names, metric_types,
                dataset_types, data_loaders, dataset_sizes, device, num_epochs=25, batch_size=4, patience=5):
    '''Train the model'''
    
    since = time.time()

#     best_model_wts = copy.deepcopy(model.state_dict())
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    metrics = {m_type: defaultdict(float) for m_type in metric_types}
    
    print(f'{"-"*20}\nModel {i+1}\n{"-"*20}\n')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in dataset_types:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            batch_metrics = {'tp': defaultdict(int), 'size': defaultdict(int),
                             'fp': defaultdict(int), 'fn': defaultdict(int)}

            # Iterate over data.
            for inputs, labels in data_loaders[i][phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                batch_metrics = update_batch_metrics(batch_metrics, preds, labels)

            if phase == 'train':
                scheduler.step()

            epoch_metrics = get_epoch_metrics(running_loss, dataset_sizes, phase,
                                              running_corrects, batch_metrics, metric_types)
            print_metrics(epoch_metrics, cls_names, phase=phase)
            # deep copy the model

        if phase == 'test':
            early_stopping(epoch_metrics['loss']['All'], model)
            if early_stopping.early_stop:
                print("Early stopping!!")
                break

    for metric_type in metric_types:
        for img_cls in cls_names:
            metrics[metric_type][img_cls] = epoch_metrics[metric_type][img_cls]
            
    print_metrics(metrics, cls_names, phase='Best results')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('-' * 20, '\n')

    return model, metrics


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience (Source: https://github.com/Bjarten/early-stopping-pytorch)."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.update_val_loss_min(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.update_val_loss_min(val_loss, model)
            self.counter = 0

    def update_val_loss_min(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).\n')
            
        self.val_loss_min = val_loss