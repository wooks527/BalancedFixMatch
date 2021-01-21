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



def create_experiment_data(cfg):
    if cfg['images_dir'] == None:
        cfg['images_dir'] = cfg['data_dir']
    
    if 'class_name' not in cfg.keys():
        cfg['class_name'] = ['covid-19', 'pneumonia', 'normal']
        cls2id = {'covid-19':0,'pneumonia':1,'normal':2}
    
    if not os.path.exists(cfg['data_dir']):
        os.makedirs(cfg['data_dir'])

    for dataset_type in cfg['dataset_types']:
        image_paths = [glob.glob(os.path.join(cfg['images_dir'],dataset_type,cls_name,'*')) for cls_name in cfg['class_name']]
        for i in range(len(image_paths)):
            for j in range(len(image_paths[i])):
                image_paths[i][j]= image_paths[i][j].replace('\\','/')
               

        for i, cls_name in enumerate(cfg['class_name']): # Check folder.
            assert len(image_paths[i]),'{} does not have a {} folder'.format(os.path.join(cfg['images_dir'],dataset_type),cls_name)

        if dataset_type=='train':
            for i in range(cfg['fold']):
                info=[dict() for _ in range(len(cfg['experiment_numlabel']))]
                all_image_paths = copy.deepcopy(image_paths)

                # init data
                for j in range(len(cfg['experiment_numlabel'])):
                    info[j]['train_lb']=dict()
                    for cls_name in cfg['class_name']:
                        info[j]['train_lb'][cls_name] = []
    
                    info[j]['train_ulb'] = dict()
                    for m in range(len(cfg['experiment_mu'])):
                        info[j]['train_ulb'][cfg['experiment_mu'][m]]=[]
                
                # load fixed data
                for j in range(len(cfg['fixed_numlabel'])): 
                    for m in range(len(cfg['fixed_mu'])):
                        assert os.path.exists(os.path.join(cfg['data_dir'],f"nl{cfg['fixed_numlabel'][j]}",f"{cfg['fixed_mu'][m]}" ,f"train_lb_{i}.txt"))
                        assert os.path.exists(os.path.join(cfg['data_dir'],f"nl{cfg['fixed_numlabel'][j]}",f"{cfg['fixed_mu'][m]}" ,f"train_ulb_{i}.txt"))
                        f = open(os.path.join(cfg['data_dir'],f"nl{cfg['fixed_numlabel'][j]}",f"{cfg['fixed_mu'][m]}" ,f"train_lb_{i}.txt"),'r')
                        lines = f.readlines()
                        f.close()
                        
                        if m == len(cfg['fixed_mu'])-1:
                            for line in lines:
                                line = line.rstrip('\n').split()
                                if j==len(cfg['fixed_numlabel'])-1:
                                    all_image_paths[cls2id[line[1]]].remove(line[0])
                                info[j]['train_lb'][line[1]].append(line[0])
                        
                        f = open(os.path.join(cfg['data_dir'],f"nl{cfg['fixed_numlabel'][j]}",f"{cfg['fixed_mu'][m]}" ,f"train_ulb_{i}.txt"),'r')
                        lines = f.readlines()
                        f.close()
                    
                        for line in lines:
                            line = line.rstrip('\n')
                            if j==len(cfg['fixed_numlabel'])-1 and m == len(cfg['fixed_mu'])-1:
                                for cls_name in cfg['class_name']:
                                    if cls_name in line:
                                        all_image_paths[cls2id[cls_name]].remove(line)
                                        break
                            info[j]['train_ulb'][cfg['fixed_mu'][m]].append(line)
                        print(f"Load train_ulb nl = {cfg['fixed_numlabel'][j]}, mu = {cfg['fixed_mu'][m]}")8

                if not os.path.exists(os.path.join(cfg['data_dir'],f"nl{cfg['experiment_numlabel'][0]}",f"{cfg['experiment_mu'][0]}" ,f"train_lb_{i}.txt")):
                    print("There is no pivot file")
                    # Make labeled data pivot
                    for c,cls_name in enumerate(cfg['class_name']):
                        choiced_paths = list(np.random.choice(all_image_paths[c],cfg['experiment_numlabel'][0],replace=False))
                        for im_path in choiced_paths:
                            all_image_paths[c].remove(im_path)
                            info[0]['train_lb'][cls_name].append(im_path)
                        
                    for m in range(len(cfg['experiment_mu'])):
                        if not os.path.exists(os.path.join(cfg['data_dir'],f"nl{cfg['experiment_numlabel'][0]}",f"{cfg['experiment_mu'][m]}")):
                            os.makedirs(os.path.join(cfg['data_dir'],f"nl{cfg['experiment_numlabel'][0]}",f"{cfg['experiment_mu'][m]}"))

                        with open(os.path.join(cfg['data_dir'],f"nl{cfg['experiment_numlabel'][0]}",f"{cfg['experiment_mu'][m]}" ,f"train_lb_{i}.txt"),'w') as f:
                            for n in range(cfg['experiment_numlabel'][0]):
                                for cls_name in cfg['class_name']:
                                    im_path = info[0]['train_lb'][cls_name][n]
                                    f.writelines(im_path + " " + cls_name+ "\n")
                    print('"{}" created in {}'.format(f"train_lb_{i}.txt",os.path.join(cfg['data_dir'],f"nl{cfg['experiment_numlabel'][0]}",f"{cfg['experiment_mu'][m]}")))

                
                # Make labeled data
                for j in range(len(cfg['experiment_numlabel'])):
                    
                    if not len(info[j]['train_lb'][cfg['class_name'][0]]):
                        for c,cls_name in enumerate(cfg['class_name']):
                            need_num = cfg['experiment_numlabel'][j]-cfg['experiment_numlabel'][j-1]
                            choiced_paths = list(np.random.choice(all_image_paths[c],need_num,replace=False))
                            
                            for im_path in choiced_paths:
                                all_image_paths[c].remove(im_path)
                                
                            choiced_paths = info[j-1]['train_lb'][cls_name] + choiced_paths

                            for im_path in choiced_paths:
                                info[j]['train_lb'][cls_name].append(im_path)
                                          

                    for m in range(len(cfg['experiment_mu'])):
                        if os.path.exists(os.path.join(cfg['data_dir'],f"nl{cfg['experiment_numlabel'][j]}",f"{cfg['experiment_mu'][m]}" ,f"train_lb_{i}.txt")):
                            continue

                        if not os.path.exists(os.path.join(cfg['data_dir'],f"nl{cfg['experiment_numlabel'][j]}",f"{cfg['experiment_mu'][m]}")):
                            os.makedirs(os.path.join(cfg['data_dir'],f"nl{cfg['experiment_numlabel'][j]}",f"{cfg['experiment_mu'][m]}"))

                        with open(os.path.join(cfg['data_dir'],f"nl{cfg['experiment_numlabel'][j]}",f"{cfg['experiment_mu'][m]}" ,f"train_lb_{i}.txt"),'w') as f:
                            for n in range(cfg['experiment_numlabel'][j]):
                                for cls_name in cfg['class_name']:
                                    im_path = info[j]['train_lb'][cls_name][n]
                                    f.writelines(im_path + " " + cls_name+ "\n")
                        print('"{}" created in {}'.format(f"train_lb_{i}.txt",os.path.join(cfg['data_dir'],f"nl{cfg['experiment_numlabel'][j]}",f"{cfg['experiment_mu'][m]}")))


                # Convert txt per class to all txt
                temp_list = []
                for im_paths in all_image_paths:
                    temp_list+=im_paths
                all_image_paths = temp_list

                # Make unlabeled pivot
                if not os.path.exists(os.path.join(cfg['data_dir'],f"nl{cfg['experiment_numlabel'][0]}",f"{cfg['experiment_mu'][0]}" ,f"train_ulb_{i}.txt")):
                    need_num = cfg['experiment_numlabel'][0]*cfg['experiment_mu'][0]*len(cfg['class_name'])
                    choiced_paths = list(np.random.choice(all_image_paths,need_num,replace=False))
                     
                    with open(os.path.join(cfg['data_dir'],f"nl{cfg['experiment_numlabel'][0]}",f"{cfg['experiment_mu'][0]}" ,f"train_ulb_{i}.txt"),'w') as f:
                        for im_path in choiced_paths:
                            f.writelines(im_path+ "\n")
                            all_image_paths.remove(im_path)
                            info[0]['train_ulb'][cfg['experiment_mu'][0]].append(im_path)
                    print('"{}.txt" created in {}'.format(f"train_ulb_{i}.txt",os.path.join(cfg['data_dir'],f"nl{cfg['experiment_numlabel'][0]}",f"{cfg['experiment_mu'][0]}")))

                
                # Make unlabeled data
                for j in range(len(cfg['experiment_numlabel'])):

                    for m in range(len(cfg['experiment_mu'])):
                        if len(info[j]['train_ulb'][cfg['experiment_mu'][m]]): continue

                        if j==0:
                            pre_list = info[j]['train_ulb'][cfg['experiment_mu'][m-1]]
                        elif m==0:
                            pre_list = info[j-1]['train_ulb'][cfg['experiment_mu'][m]]
                        else:
                            pre_list = list(set(info[j-1]['train_ulb'][cfg['experiment_mu'][m]]+info[j]['train_ulb'][cfg['experiment_mu'][m-1]]))

                        need_num = (cfg['experiment_numlabel'][j]*cfg['experiment_mu'][m]*len(cfg['class_name']))-len(pre_list)
                        choiced_paths = list(np.random.choice(all_image_paths,need_num,replace=False))
                        
                        for im_path in choiced_paths:
                            all_image_paths.remove(im_path)
                        
                        choiced_paths = pre_list + choiced_paths
                        with open(os.path.join(cfg['data_dir'],f"nl{cfg['experiment_numlabel'][j]}",f"{cfg['experiment_mu'][m]}" ,f"train_ulb_{i}.txt"),'w') as f:
                            for im_path in choiced_paths:
                                f.writelines(im_path+ "\n")
                                info[j]['train_ulb'][cfg['experiment_mu'][m]].append(im_path)
                        print('"{}.txt" created in {}'.format(f"train_ulb_{i}.txt",os.path.join(cfg['data_dir'],f"nl{cfg['experiment_numlabel'][j]}",f"{cfg['experiment_mu'][m]}")))

        else:
            for num_label in cfg['experiment_numlabel']:
                for mu in cfg['experiment_mu']:
                    folder = os.path.join(cfg['data_dir'],f"nl{num_label}",f"{mu}")
                    if os.path.exists(os.path.join(folder, '{}.txt'.format(dataset_type))): continue
                    if not os.path.exists(folder):
                        os.makedirs(folder)

                    with open(os.path.join(folder, '{}.txt'.format(dataset_type)), 'w') as f: # Make txt file.
                        for i, cls_name in enumerate(cfg['class_name']):
                            for im_path in image_paths[i]:
                                f.writelines(im_path + " " + cls_name+ "\n")
                    print('"{}.txt" created in {}'.format(dataset_type,folder))


if __name__ == '__main__':
    import argparse
    import numpy as np
    import glob
    import random
    import copy

    # Create configs
    parser = argparse.ArgumentParser(description='Make experiment data')
    parser.add_argument('--images_dir', type=str, default='./data/CXR', help='input images data directory')
    parser.add_argument('--data_dir', type=str, default='./data/CXR', help='input data directory')
    parser.add_argument('--fold', type=int, default=4, help='the number of sampled dataset')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument('--dataset_types', type=str, nargs='+', default=['train', 'test'], help='dataset types')
    parser.add_argument('--experiment_numlabel', type=int, nargs='+',default=[25,50,100,150], help='')
    parser.add_argument('--experiment_mu', type=int, nargs='+',default=[1,2,3], help='')
    parser.add_argument('--fixed_numlabel', type=int, nargs='+',default=[0], help='')
    parser.add_argument('--fixed_mu', type=int, nargs='+',default=[0], help='')
    cfg = vars(parser.parse_args())
    print(cfg)
    create_experiment_data(cfg)
    exit()