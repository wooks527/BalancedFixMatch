import torch
import torch.nn.functional as F
# import torch_xla.core.xla_model as xm
# import torch_xla.distributed.parallel_loader as pl
import time
import copy
import os
import numpy as np
import torch

from collections import defaultdict
from datasets.utils import get_data_loaders
from models.model import get_model, save_model
from models.metrics import update_batch_metrics, get_epoch_metrics, update_mean_metrics, print_metrics

def train_model(model, criterion, optimizer, scheduler, i, class_names, metric_targets, metric_types,
                dataset_types, data_loaders, dataset_sizes, device, cfg, num_epochs=25, batch_size=4,
                patience=5, lambda_u=1.0, threshold=0.95, purpose='baseline', is_early=True):
    '''Train the model.

    Args:
        model (obj): the model which will be trained
        criterion (obj): the loss function (e.g. cross entropy)
        optimizer (obj): the optimizer (e.g. Adam)
        scheduler (obj): the learning scheduler (e.g. Step decay)
        i (int): the number indicating which model it is
        class_names (dict): class names for images (e.g. {0: 'covid-19', 1: 'pneumonia', 2: 'normal'})
        metric_targets (list): metric targets to calculate performance metrics of the model
                               (e.g. ['all', 'covid-19', 'pneumonia', 'normal'])
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
        best_metrics (dict): the results of the best performance metrics after training the model
    '''

    since = time.time()
    if is_early:
        early_stopping = EarlyStopping(patience=patience, verbose=True)
    best_metrics = {m_type: defaultdict(float) for m_type in metric_types}
    epoch_metrics_list = []

    print(f'{"-" * 20}\nModel {i + 1}\n{"-" * 20}\n')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and test phase
        for phase in dataset_types:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            epoch_loss = 0.0
            batch_metrics = {'tp': defaultdict(int), 'size': defaultdict(int),
                             'fp': defaultdict(int), 'fn': defaultdict(int)}
            mask_ratio = []  # just for fixmatch

            # Create a pareallel loader
            if cfg['use_tpu']:
                # if phase == 'train':
                #     data_loaders[phase].sampler.set_epoch(epoch)

                final_data_loader = pl.ParallelLoader(data_loaders[phase], [device]).per_device_loader(device)
            else:
                final_data_loader = data_loaders[phase]


            # Iterate over data.
            for batch in final_data_loader:
                # Load batches
                inputs_lb, labels = batch['img_lb'], batch['label']
                inputs_lb = inputs_lb.to(device)
                labels = labels.to(device)
                if purpose != 'baseline' and phase == 'train':
                    inputs_ulb, inputs_ulb_wa = batch['img_ulb'], batch['img_ulb_wa']
                    inputs_ulb = inputs_ulb.to(device)
                    inputs_ulb_wa = inputs_ulb_wa.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # Forward the model
                with torch.set_grad_enabled(phase == 'train'):
                    # Calculate labeled loss
                    outputs = model(inputs_lb)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_lb = criterion(outputs, labels)
                    
                    # Calculate unlabeled loss for FixMatch
                    if purpose != 'baseline' and phase == 'train':
                        outputs_ulb = model(inputs_ulb)
                        probs_ulb = torch.softmax(outputs_ulb, dim=-1)
                        probs_ulb, preds_ulb = torch.max(probs_ulb, 1)
                        mask = probs_ulb.ge(threshold).float()

                        outputs_ulb_wa = model(inputs_ulb_wa)
                        loss_ulb = (F.cross_entropy(outputs_ulb_wa, preds_ulb, reduction='none') * mask).mean()
                        mask_ratio.append(mask.mean().item())
                        loss += loss_ulb * lambda_u

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()

                        if cfg['use_tpu']:
                            xm.optimizer_step(optimizer)
                        else:
                            optimizer.step()

                # Calculate loss and metrics per the batch
                if purpose == 'baseline' or phase == 'test':
                    epoch_loss += loss.item() * inputs_lb.size(0)
                else:  # FixMatch
                    epoch_loss += loss_lb.item() * inputs_lb.size(0) \
                                    + loss_ulb * lambda_u * inputs_ulb.size(0)

                batch_metrics = update_batch_metrics(batch_metrics, preds, labels, class_names)

            # if phase == 'train':
            #     scheduler.step()

            # Calcluate the metrics (e.g. Accuracy) per the epoch
            epoch_metrics = get_epoch_metrics(epoch_loss, dataset_sizes, phase,
                                              class_names, batch_metrics, metric_types)
            print_metrics(epoch_metrics, metric_targets, cfg, phase=phase, mask_ratio=mask_ratio)
            if phase != 'train':
                epoch_metrics_list.append(epoch_metrics)

        # Check early stopping
        if phase == 'test' and is_early:
            early_stopping(epoch_metrics['loss']['all'], model)
            if early_stopping.early_stop:
                print("Early stopping!!")
                break

    # Set best metrics based on recent 5 epochs metrics
    for metric_type in metric_types: # e.g. ['acc', 'ppv', ...]
        for metric_target in metric_targets: # e.g. ['all', 'covid-19', ...]
            # Accuracy couldn't calculate for each class
            if metric_type == 'acc' and metric_target in class_names:
                continue

            best_mean = np.array([em[metric_type][metric_target]
                                    for em in epoch_metrics_list[-5:]]).mean()
            best_std = np.array([em[metric_type][metric_target]
                                for em in epoch_metrics_list[-5:]]).std()
            best_metrics[metric_type][metric_target] = (best_mean, best_std)

    print_metrics(best_metrics, metric_targets, cfg, phase='Best results')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('-' * 20, '\n')

    return model, best_metrics

def train_models(index, cfg):
    '''Train models for fold times.
    
    Args:
        cfg (dict): The cfg parameter must have purpose, data dir information and mu
                    (Used only when purpose is not baseline).
    Returns:
        trained_models (list): trained models
    '''
    # Set print's output stream to the file
    from utils import set_print_to_file
    global print
    print = set_print_to_file(print, cfg)

    # Set parameters
    if cfg['use_tpu']:
        # Acquires the (unique) Cloud TPU core corresponding to this process's index
        device = xm.xla_device()
        print("Process", index ,"is using", xm.xla_real_devices([str(device)])[0])
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        data_loaders, dataset_sizes, class_names = get_data_loaders(dataset_type='val', cfg=cfg)
    except:
        data_loaders, dataset_sizes, class_names = get_data_loaders(dataset_type='test', cfg=cfg)
    mean_metrics = {m_type: defaultdict(tuple) for m_type in cfg['metric_types']}
    metric_targets = ['all'] + class_names

    # Train the models
    for i in range(cfg['fold']):
        data_loaders, dataset_sizes, class_names = get_data_loaders(dataset_type='train', cfg=cfg,
                                                                    dataset_sizes=dataset_sizes,
                                                                    data_loaders=data_loaders, fold_id=i)

        model_ft, criterion, optimizer_ft, exp_lr_scheduler = get_model(device, fine_tuning=cfg['is_finetuning'],
                                                                        scheduler=cfg['scheduler'])
        model, metrics = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, i, class_names, metric_targets,
                                     cfg['metric_types'], cfg['dataset_types'], data_loaders, dataset_sizes, device, cfg,
                                     num_epochs=cfg['epochs'], lambda_u=cfg['lambda_u'], threshold=cfg['threshold'],
                                     purpose=cfg['purpose'], is_early=False)

        # save_model(model, cfg)
        del model_ft, model
        mean_metrics = update_mean_metrics(metric_targets, mean_metrics, metrics, status='training')

    # Calculate mean metrics
    mean_metrics = update_mean_metrics(metric_targets, mean_metrics, status='final', fold=cfg['fold'])
    print_metrics(mean_metrics, metric_targets, cfg, phase='Mean results')


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