import sys
from collections import defaultdict

def update_batch_metrics(batch_metrics, preds, labels):
    '''Update the batch metrics.
    
    Args:
        batch_metrics (dict): performance metrics for batch (e.g. Accuracy)
        preds (obj): the prediction results
        labels (obj): the labels
    
    Returns:
        batch_metrics (dict): performance metrics for batch (e.g. Accuracy)
    '''
    CLS = {0: 'COVID-19', 1: 'Normal', 2: 'Pneumonia'}
    for pred, label in zip(preds, labels.data):
        batch_metrics['size'][CLS[label.item()]] += 1
        if pred == label.data:
            batch_metrics['tp'][CLS[label.item()]] += 1
        else:
            batch_metrics['fp'][CLS[pred.item()]] += 1
            batch_metrics['fn'][CLS[label.item()]] += 1

    return batch_metrics

def get_epoch_metrics(running_loss, dataset_sizes, phase, running_corrects, batch_metrics,
                      metric_types, mask_ratio=None, epsilon=sys.float_info.epsilon):
    '''Calculate the performance metrics per the epoch.
    
    Args:
        running_loss (float): the loss per epoch
        dataset_sizes (dict): sizes of train and test datasets
        phase (str): current status which hadle the model (e.g. train or test)
        running_corrects (double): TP + TN
        batch_metrics (dict): performance metrics for batch (e.g. Accuracy)
        metric_types (list): the performance metrics of the model (e.g. Accuracy, F1-Score and so on)
        mask_ratio (float): the mask ratio to mask unlabeled loss in prediction results
                            which are smaller than threshold
        epsilon (float): the small number to prevent the situation
                         that the number is devided by zero in calculating the F1-score
    
    Returns:
        
    '''
    epoch_metrics = defaultdict(dict)
    epoch_metrics['loss']['All'] = running_loss / dataset_sizes[phase]
    epoch_metrics['acc']['All'] = running_corrects.double() / dataset_sizes[phase]
    
    cls_names = ('COVID-19', 'Normal', 'Pneumonia')
    if 'ppv' in metric_types:
        epoch_metrics['ppv'] = defaultdict(float, {c: round(float(batch_metrics['tp'][c])
                                                            / (batch_metrics['tp'][c]
                                                               + batch_metrics['fp'][c]
                                                               + epsilon), 4)
                                                   for c in cls_names})
    if 'recall' in metric_types:
        epoch_metrics['recall'] = defaultdict(float, {c: round(float(batch_metrics['tp'][c])
                                                                / (batch_metrics['tp'][c]
                                                                   + batch_metrics['fn'][c]
                                                                   + epsilon), 4)
                                                      for c in cls_names})
    if 'f1' in metric_types:
        epoch_metrics['f1'] = {c: round(2 * epoch_metrics['ppv'][c]
                                          * epoch_metrics['recall'][c]
                                          / (epoch_metrics['ppv'][c]
                                             + epoch_metrics['recall'][c]
                                             + epsilon), 4)
                               for c in cls_names}
        epoch_metrics['f1']['All'] = sum([f for f in epoch_metrics['f1'].values()]) / 3
    
    return epoch_metrics

def update_mean_metrics(cls_names, mean_metrics, metrics=None, status='training', fold=None):
    '''Update mean metrics among all models.
    
    Args:
        cls_names (list): class names to calculate performance metrics of the model including "All"
                          (e.g. ['All', 'COVID-19', 'Pneumonia', 'Normal'])
        mean_metrics (dict): mean performance metrics among all models
        metrics (dict): best performance metrics per the model
        status (str): current status (e.g. training or final)
        fold (int): the number of models which will be trained and tested
    
    Returns:
        mean_metrics (dict): mean performance metrics among all models
    '''
    if status == 'training':
        for metric_type, targets in metrics.items():
            if metric_type == 'acc':
                mean_metrics[metric_type]['All'] += targets['All']
            else:
                for img_cls in cls_names:
                    if targets.get(img_cls):
                        mean_metrics[metric_type][img_cls] += targets[img_cls]
    else: # final
        for metric_type, targets in mean_metrics.items():
            if metric_type == 'acc':
                mean_metrics[metric_type]['All'] /= fold
            else:
                for img_cls in cls_names:
                    if targets.get(img_cls):
                        mean_metrics[metric_type][img_cls] /= fold

    return mean_metrics
    

def print_metrics(epoch_metrics, cls_names, phase='', mask_ratio=None):
    '''Print performance metrics.
    
    Args:
        epoch_metrics (dict): performance metrics for epoch (e.g. Accuracy)
        cls_names (list): class names to calculate performance metrics of the model including "All"
                          (e.g. ['All', 'COVID-19', 'Pneumonia', 'Normal'])
        phase (str): current status which hadle the model (e.g. train or test)
        mask_ratio (float): the mask ratio to mask unlabeled loss in prediction results
                            which are smaller than threshold
    
    Returns:
        nothing
    '''
    if 'Best' in phase:
        print(f'\n{"-"*20}')

    print(f'[{phase}]')
    for metric_type, targets in epoch_metrics.items():
        results = f'{metric_type.upper()} -'
        if metric_type == 'loss' or metric_type == 'acc':
            results += f' {targets["All"]:.4f}'
        else:
            for img_cls in cls_names:
                results += f' {img_cls}: {targets[img_cls]:.4f} '
        print(results)
        
    if mask_ratio:
        print(f'Mask ratio\'s range: {1 - max(mask_ratio)} ~ {1 - min(mask_ratio)}')
    print()