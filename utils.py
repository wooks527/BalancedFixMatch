import os

def init_file_for_print(cfg):
    '''Init results.txt file.'''
    if not os.path.isdir('results/'):
            os.mkdir('results/')
    fname = f"results/{cfg['purpose']}-b{cfg['batch_size']}-nl{cfg['num_labeled']}-m{cfg['mu']}-lb{cfg['lambda_u']}-th{cfg['threshold']}-sharp{cfg['sharpening']}-T{cfg['temperature']}-focal{cfg['focal_loss']}-fg{cfg['gamma']}-opt{cfg['opt']}-lr{cfg['lr']}-mom{cfg['momentum']}-sc:{cfg['scheduler']}-r{cfg['random_seed']}.txt"
    with open(fname, 'w'):
        pass

def set_print_to_file(print, cfg):
    '''Set print's output stream to the file.
       This codes are referenced by https://stackoverflow.com/a/27622201.

    Args:
        cfg (dict): The cfg parameter must have purpose, data dir information and mu
                    (Used only when purpose is not baseline).
    Returns:
        nothing
    '''
    if cfg['print_to_file']:
        fname = f"results/{cfg['purpose']}-b{cfg['batch_size']}-nl{cfg['num_labeled']}-m{cfg['mu']}-lb{cfg['lambda_u']}-th{cfg['threshold']}-sharp{cfg['sharpening']}-T{cfg['temperature']}-focal{cfg['focal_loss']}-fg{cfg['gamma']}-opt{cfg['opt']}-lr{cfg['lr']}-mom{cfg['momentum']}-sc:{cfg['scheduler']}-r{cfg['random_seed']}.txt"
        def file_print(func):
            def wrapped_func(*args,**kwargs):
                kwargs['file'] = open(fname, 'a')
                return func(*args,**kwargs)
            return wrapped_func
        
        return file_print(print)
    elif cfg['use_tpu']:
        import torch_xla.core.xla_model as xm
        return xm.master_print
    return print