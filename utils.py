import os

def init_file_for_print():
    '''Init results.txt file.'''
    fname = f'results/results.txt'
    with open(fname, 'w'):
        pass

def set_print_to_file(print, print_to_file):
    '''Set print's output stream to the file.
       This codes are referenced by https://stackoverflow.com/a/27622201.

    Args:
        cfg (dict): The cfg parameter must have purpose, data dir information and mu
                    (Used only when purpose is not baseline).
    Returns:
        nothing
    '''
    if print_to_file:
        if not os.path.isdir('results/'):
            os.mkdir('results/')

        fname = f'results/results.txt'
        def file_print(func):
            def wrapped_func(*args,**kwargs):
                kwargs['file'] = open(fname, 'a')
                return func(*args,**kwargs)
            return wrapped_func
        
        return file_print(print)
    return print