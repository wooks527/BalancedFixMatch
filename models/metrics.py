import sys
from collections import defaultdict

def update_batch_metrics(batch_metrics, preds, labels, class_names):
    '''Update the batch metrics.
    
    Args:
        batch_metrics (dict): performance metrics for batch (e.g. Accuracy)
        preds (obj): the prediction results
        labels (obj): the labels
        class_names (dict): class names for images (e.g. {0: 'covid-19', 1: 'pneumonia', 2: 'normal'})
    
    Returns:
        batch_metrics (dict): performance metrics for batch (e.g. Accuracy)
    '''
    for pred, label in zip(preds, labels.data):
        if pred == label.data:
            batch_metrics['tp'][class_names[label.item()]] += 1
        else:
            batch_metrics['fp'][class_names[pred.item()]] += 1
            batch_metrics['fn'][class_names[label.item()]] += 1
    return batch_metrics

def get_epoch_metrics(epoch_loss, dataset_sizes, phase, class_names, batch_metrics,
                      metric_types, mask_ratio=None, epsilon=sys.float_info.epsilon):
    '''Calculate the performance metrics per the epoch.
    
    Args:
        epoch_loss (float): the loss per epoch
        dataset_sizes (dict): sizes of train and test datasets
        phase (str): current status which hadle the model (e.g. train or test)
        class_names (dict): class names for images (e.g. {0: 'covid-19', 1: 'pneumonia', 2: 'normal'})
        batch_metrics (dict): performance metrics for batch (e.g. Accuracy)
        metric_types (list): the performance metrics of the model (e.g. Accuracy, F1-Score and so on)
        mask_ratio (float): the mask ratio to mask unlabeled loss in prediction results
                            which are smaller than threshold
        epsilon (float): the small number to prevent the situation
                         that the number is devided by zero in calculating the F1-score
    
    Returns:
        
    '''
    epoch_metrics = defaultdict(dict)
    epoch_metrics['loss']['all'] = epoch_loss / dataset_sizes[phase]
    epoch_metrics['acc']['all'] = sum(batch_metrics['tp'].values()) / dataset_sizes[phase]
    
    if 'ppv' in metric_types:
        epoch_metrics['ppv'] = defaultdict(float, {c: round(float(batch_metrics['tp'][c])
                                                            / (batch_metrics['tp'][c]
                                                               + batch_metrics['fp'][c]
                                                               + epsilon), 4)
                                                   for c in class_names})
        epoch_metrics['ppv']['all'] = sum([f for f in epoch_metrics['ppv'].values()]) / 3
    if 'recall' in metric_types:
        epoch_metrics['recall'] = defaultdict(float, {c: round(float(batch_metrics['tp'][c])
                                                                / (batch_metrics['tp'][c]
                                                                   + batch_metrics['fn'][c]
                                                                   + epsilon), 4)
                                                      for c in class_names})
        epoch_metrics['recall']['all'] = sum([f for f in epoch_metrics['recall'].values()]) / 3
    if 'f1' in metric_types:
        epoch_metrics['f1'] = {c: round(2 * epoch_metrics['ppv'][c]
                                          * epoch_metrics['recall'][c]
                                          / (epoch_metrics['ppv'][c]
                                             + epoch_metrics['recall'][c]
                                             + epsilon), 4)
                               for c in class_names}
        epoch_metrics['f1']['all'] = sum([f for f in epoch_metrics['f1'].values()]) / 3
    return epoch_metrics

def update_mean_metrics(metric_targets, mean_metrics, metrics=None, status='training', fold=None):
    '''Update mean metrics among all models.
    
    Args:
        metric_targets (list): metric targets to calculate performance metrics of the model
                               (e.g. ['all', 'covid-19', 'pneumonia', 'normal'])
        mean_metrics (dict): mean performance metrics among all models
        metrics (dict): best performance metrics per the model
        status (str): current status (e.g. training or final)
        fold (int): the number of models which will be trained and tested
    
    Returns:
        mean_metrics (dict): mean performance metrics among all models
    '''
    if status == 'training':
        for metric_type, targets in metrics.items():
            for metric_target in metric_targets:
                if targets.get(metric_target):
                    cur_best_mean, cur_best_std = targets[metric_target]

                    # Initial state of mean_metrics
                    if not mean_metrics[metric_type].get(metric_target):
                        best_mean, best_std = 0.0, 0.0
                    else: # There are values in mean_metrics
                        best_mean, best_std = mean_metrics[metric_type][metric_target]

                    mean_metrics[metric_type][metric_target] = (cur_best_mean + best_mean,
                                                                cur_best_std + best_std)
    else: # final
        for metric_type, targets in mean_metrics.items():
            for metric_target in metric_targets:
                if targets.get(metric_target):
                    sum_best_mean, sum_best_std = mean_metrics[metric_type][metric_target]
                    mean_metrics[metric_type][metric_target] = (sum_best_mean / fold,
                                                                sum_best_std / fold)

    return mean_metrics
    

def print_metrics(epoch_metrics, metric_targets, phase='', mask_ratio=None):
    '''Print performance metrics.
    
    Args:
        epoch_metrics (dict): performance metrics for epoch (e.g. Accuracy)
        metric_targets (list): metric targets to calculate performance metrics of the model
                               (e.g. ['all', 'covid-19', 'pneumonia', 'normal'])
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
            # Do not use for best or mean results because it consist of the mean and std
            if phase == 'Best results' or phase == 'Mean results':
                best_mean, best_std = targets["all"]
                results += f' {best_mean:.4f} (±{best_std:.2f})'
            else:
                results += f' {targets["all"]:.4f}'
        else:
            for metric_target in metric_targets:
                # Do not use for best or mean results because it consist of the mean and std
                if phase == 'Best results' or phase == 'Mean results':
                    best_mean, best_std = targets[metric_target]
                    results += f' {metric_target.upper()}: {best_mean:.4f} (±{best_std:.2f}) '
                else:
                    results += f' {metric_target.upper()}: {targets[metric_target]:.4f} '
        print(results)
        
    if mask_ratio:
        print(f'Mask ratio\'s range: {1 - max(mask_ratio)} ~ {1 - min(mask_ratio)}')
    print()