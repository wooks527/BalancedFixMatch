import torch
import torch.nn.functional as F
import time
import copy
import numpy as np
import torch

from collections import defaultdict
from models.metrics import update_batch_metrics, get_epoch_metrics, print_metrics

def train_model(model, criterion, optimizer, scheduler, i, cls_names, metric_types, dataset_types,
                data_loaders, dataset_sizes, device, num_epochs=25, batch_size=4, patience=5,
                lambda_u=1.0, threshold=0.95, purpose='baseline', is_early=True):
    '''Train the model.
    
    Args:
        model (obj): the model which will be trained
        criterion (obj): the loss function (e.g. cross entropy)
        optimizer (obj): the optimizer (e.g. Adam)
        scheduler (obj): the learning scheduler (e.g. Step decay)
        i (int): the number indicating which model it is
        cls_names (list): class names to calculate performance metrics of the model including "All"
                          (e.g. ['All', 'COVID-19', 'Pneumonia', 'Normal'])
        metric_types (list): the performance metrics of the model (e.g. Accuracy, F1-Score and so on)
        dataset_types (list): dataset types for train and test (e.g. ['train', 'test'] or ['train', 'val', 'test])
        data_loaders (list): data loaders applied transformations, the batch size and so on
        dataset_sizes (dict): sizes of train and test datasets
        device (obj): the device where the model will be trained (e.g. cpu or gpu)
        num_epochs (int): the number of epochs
        batch_size (int): the batch size
        patience (int): the number of patience times for early stopping
        lambda_u (float): the ratio of reflect unlabeled loss
        threshold (float): the treshold for predicted results for unlabeled data
        purpose (str): the purpose of the model
    
    Returns:
        model (obj): the model which was trained
        metrics (dict): the results of the performance metrics after training the model
    '''
    
    since = time.time()
    if is_early:
        early_stopping = EarlyStopping(patience=patience, verbose=True)
    metrics = {m_type: defaultdict(float) for m_type in metric_types}
    
    print(f'{"-"*20}\nModel {i+1}\n{"-"*20}\n')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and test phase
        for phase in dataset_types:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            batch_metrics = {'tp': defaultdict(int), 'size': defaultdict(int),
                             'fp': defaultdict(int), 'fn': defaultdict(int)}
            mask_ratio = [] # just for fixmatch

            phase_for_data_loader = phase
            if purpose == 'fixmatch' and phase == 'train':
                phase_for_data_loader = 'train_lb'
                
            # Iterate over data.
            for inputs, labels in data_loaders[i][phase_for_data_loader]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                if purpose == 'fixmatch' and phase == 'train':
                    inputs_lb, labels_lb = inputs, labels
                    inputs_ulb, _ = next(iter(data_loaders[i]['train_ulb']))
                    inputs_ulb_wa, _ = next(iter(data_loaders[i]['train_ulb_wa']))
                    inputs_ulb = inputs_ulb.to(device)
                    inputs_ulb_wa = inputs_ulb_wa.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if purpose == 'baseline' or phase == 'test':
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    else: # fixmatch in train
                        outputs_lb = model(inputs_lb)
                        _, preds_lb = torch.max(outputs_lb, 1)
                        loss_lb = criterion(outputs_lb, labels_lb)

                        loss_ulb, masks = 0, 0.0
                        outputs_ulb = model(inputs_ulb)
                        pseudo_label = torch.softmax(outputs_ulb, dim=-1)
                        probs_ulb, preds_ulb = torch.max(pseudo_label, 1)
                        mask = probs_ulb.ge(threshold).float()

                        outputs_ulb_wa = model(inputs_ulb_wa)
                        loss_ulb = (F.cross_entropy(outputs_ulb_wa, preds_ulb, reduction='none') * mask).mean()
                        mask_ratio.append(mask.mean().item())
                        loss = loss_lb + loss_ulb * lambda_u

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # Calculate loss and metrics per the batch
                if purpose == 'baseline' or phase == 'test':
                    running_loss += loss.item() * inputs.size(0)
                else: # FixMatch
                    running_loss += loss_lb.item() * inputs_lb.size(0) \
                                 + loss_ulb * lambda_u * inputs_ulb.size(0)
                    
                if phase == 'train' and purpose == 'fixmatch':
                    preds, labels = preds_lb, labels_lb
                running_corrects += torch.sum(preds == labels.data)
                batch_metrics = update_batch_metrics(batch_metrics, preds, labels)
                
            if phase == 'train':
                scheduler.step()
            
            # Calcluate the metrics (e.g. Accuracy) per the epoch
            phase_for_epoch_metrics = phase
            if phase == 'train' and purpose == 'fixmatch':
                phase_for_epoch_metrics = 'train_lb'
            epoch_metrics = get_epoch_metrics(running_loss, dataset_sizes, phase_for_epoch_metrics,
                                              running_corrects, batch_metrics, metric_types)
            print_metrics(epoch_metrics, cls_names, phase=phase, mask_ratio=mask_ratio)

        # Check early stopping
        if phase == 'test' and is_early:
            early_stopping(epoch_metrics['loss']['All'], model)
            if early_stopping.early_stop:
                print("Early stopping!!")
                break

    # Set best metrics based on epoch metrics
    # This best metrics can be changed by how calculate the best metrics
    for metric_type in metric_types:
        if metric_type == 'acc':
            metrics[metric_type]['All'] = epoch_metrics[metric_type]['All']
        else:
            for img_cls in cls_names:
                metrics[metric_type][img_cls] = epoch_metrics[metric_type][img_cls]
            
    print_metrics(metrics, cls_names, phase='Best results')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('-' * 20, '\n')

    return model, metrics


class EarlyStopping:
    '''Early stops the training if validation loss doesn't improve after a given patience.
       This code is referenced in the GitHub repository, "https://github.com/Bjarten/early-stopping-pytorch".
    '''
    
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        '''Init EarlyStopping object.
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print            
        '''
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model):
        '''Call function to check the early stopping case
        Args:
            val_loss (folat): the validation loss
            model (obj): the model which is training
        '''
        score = -val_loss

        # First update
        if self.best_score is None:
            self.best_score = score
            self.update_val_loss_min(val_loss)
        # Early stopping case when the loss is still decreasing
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        # The case which still need to be updated
        else:
            self.best_score = score
            self.update_val_loss_min(val_loss)
            self.counter = 0

    def update_val_loss_min(self, val_loss):
        '''Update the minimum validation loss when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).\n')
            
        self.val_loss_min = val_loss