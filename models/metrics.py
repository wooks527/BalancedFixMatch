import sys
from collections import defaultdict

def update_batch_metrics(batch_metrics, preds, labels):
    CLS = {0: 'COVID-19', 1: 'Pneumonia', 2: 'Normal'}
    for pred, label in zip(preds, labels.data):
        batch_metrics['size'][CLS[label.item()]] += 1
        if pred == label.data:
            batch_metrics['tp'][CLS[label.item()]] += 1
        else:
            batch_metrics['fp'][CLS[pred.item()]] += 1
            batch_metrics['fn'][CLS[label.item()]] += 1
            
    return batch_metrics

def get_epoch_metrics(running_loss, dataset_sizes, phase, running_corrects, batch_metrics, metric_types, epsilon=sys.float_info.epsilon):
    epoch_metrics = defaultdict(dict)
    epoch_metrics['loss']['All'] = running_loss / dataset_sizes[phase]
    epoch_metrics['acc'] = defaultdict(float, {c: round(float(n) / batch_metrics['size'][c], 4)
                                                for c, n in batch_metrics['tp'].items()})
    epoch_metrics['acc']['All'] = running_corrects.double() / dataset_sizes[phase]
    if 'f1' in metric_types:
        epoch_metrics['ppv'] = defaultdict(float, {c: round(float(n) / (batch_metrics['tp'][c]
                                                                        + sum([s for c_temp, s in batch_metrics['fp'].items()
                                                                               if c_temp != c])), 4)
                                                   for c, n in batch_metrics['tp'].items()})
        epoch_metrics['recall'] = defaultdict(float, {c: round(float(n) / (batch_metrics['tp'][c]
                                                                           + sum([s for c_temp, s in batch_metrics['fn'].items()
                                                                                  if c_temp != c])), 4)
                                                      for c, n in batch_metrics['tp'].items()})
        epoch_metrics['f1'] = {c: round(2 * epoch_metrics['ppv'][c] * epoch_metrics['recall'][c] / (epoch_metrics['ppv'][c]
                                                                                                    + epoch_metrics['recall'][c]
                                                                                                    + epsilon), 4)
                               for c in ('COVID-19', 'Pneumonia', 'Normal')}
        epoch_metrics['f1']['All'] = sum([f for f in epoch_metrics['f1'].values()]) / 3
    
    return epoch_metrics

def update_mean_metrics(cls_names, mean_metrics, metrics=None, status='training', fold=None):
    if status == 'training':
        for metric_type, targets in metrics.items():
            results = f'{metric_type.upper()} -'
            for img_cls in cls_names:
                if targets.get(img_cls):
                    mean_metrics[metric_type][img_cls] += targets[img_cls]
    else:
        for metric_type, targets in mean_metrics.items():
            results = f'{metric_type.upper()} -'
            for img_cls in cls_names:
                if targets.get(img_cls):
                    mean_metrics[metric_type][img_cls] /= fold

    return mean_metrics
    

def print_metrics(epoch_metrics, cls_names, phase=''):
    if 'Best' in phase:
        print(f'\n{"-"*20}')

    print(f'[{phase}]')
    for metric_type, targets in epoch_metrics.items():
        results = f'{metric_type.upper()} -'
        if metric_type == 'loss':
            results += f' {targets["All"]:.4f}'
        else:
            for img_cls in cls_names:
                results += f' {img_cls}: {targets[img_cls]:.4f} '
        print(results)
    print()