
'''
Two types of DataLoader are needed
- source only 
- transfer learning (DA, source and target datasets)

The source only DataLoader is straightforward.
The transfer learning DataLoader is more complex. 
The target dataset is split into train and validation sets.
The source dataset is split into train and validation sets.
The train loaders are combined during training.
The validation loaders are combined during validation.

The source and target dataset is split using stratified k-fold cross-validation.
'''

import json
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from torch.utils.data import Dataset
import glob
from natsort import natsorted
from AT_Great.experiment_configs.utils import get_configs
from sklearn.model_selection import KFold
from AT_Great.datautil.great_base import *
from AT_Great.datautil.util import *
from AT_Great.datautil.great_dataset import *
from easyfsl.samplers.task_sampler import TaskSampler
# from AT_Great.datautil.temp_redundant.SubsetAug import SubsetAug
from pathlib import Path


def fold_dataset_load_lda(data_path, sub, pos, args, fold, mean=None, std=None, feats='aug', compute_norms=False,
                     fold_idx_set_train='train', fold_idx_set_test='test'):
    data_path = Path(data_path)
    if mean is None and std is None:
        if compute_norms == False:
            mean, std = load_fold_norm_params(args, pos, fold)
        else:
            # If you want to comupte mena and std on the fly
            mean, std = None, None

    if fold_idx_set_test == 'all':
        # we dont need to split the data into train and test sets
        dataset = AT_GreatFolded(args, data_path, sub, pos, transform=True, target_transform=True, 
                             fold=fold, datafold_path=args['fold_config'], get_fold_set=fold_idx_set_test, 
                             mean=mean, std=std, feats=feats)
        
    else:
        
        train_dataset = AT_GreatFolded(args, data_path, sub, pos, transform=True, target_transform=True, 
                                fold=fold, datafold_path=args['fold_config'], get_fold_set=fold_idx_set_train, 
                                mean=mean, std=std, feats=feats)
        # if mean ad std would be None at the beinning of hte functin, train_dataset shoul calculate these and tey should be passes to the val_datatset
        mean = train_dataset.data_mean
        std = train_dataset.data_std
        
        val_dataset = AT_GreatFolded(args, data_path, sub, pos, transform=True, target_transform=True,
                                    fold=fold, datafold_path=args['fold_config'], get_fold_set=fold_idx_set_test, 
                                    mean=mean, std=std, feats=feats)
        
    if args['lda'] == True and fold_idx_set_test == 'all':
        print('return mode: all')
        return dataset, mean, std
        # pass
    elif args['lda'] == True:
        return train_dataset, val_dataset, mean, std

def load_fold_norm_params(args, pos, k_folds):
    # datafile_path = os.path.abspath(args['fold_params']) # previously, before optimising the yamls
    datafile_path = args['fold_params']
    args = get_configs(datafile_path)
    mean, std = args[int(pos[0])][int(k_folds)]['train_mean'], args[int(pos[0])][int(k_folds)]['train_std']
    return mean, std
    

def fold_loader_init(data_path, sub, pos, args, fold, mean=None, std=None, feats='aug', compute_norms=False,
                     fold_idx_set_train='train', fold_idx_set_test='test', data_mode='source'):
    data_path = Path(data_path)
    if mean is None and std is None:
        if compute_norms == False:
            mean, std = load_fold_norm_params(args, pos, fold)
        else:
            # If you want to comupte mena and std on the fly
            mean, std = None, None
        
    train_dataset = AT_GreatFolded(args, data_path, sub, pos, transform=True, target_transform=True, 
                            fold=fold, datafold_path=args['fold_config'], get_fold_set=fold_idx_set_train, 
                            mean=mean, std=std, feats=feats)

    # if its val set, we should be gettings statisctics from this set
    # TODO check it!
    if data_mode == 'source':
        mean = train_dataset.data_mean
        std = train_dataset.data_std
    
    val_dataset = AT_GreatFolded(args, data_path, sub, pos, transform=True, target_transform=True,
                                fold=fold, datafold_path=args['fold_config'], get_fold_set=fold_idx_set_test, 
                                mean=mean, std=std, feats=feats)
    
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'])

    if args['norm_stats'] == 'independant' or data_mode == 'target':
        return train_loader, val_loader
    elif args['norm_stats'] == 'source_domain_statisitcs':
        return train_loader, val_loader,  mean, std 
   

'''
def fold_loader_init(data_path, sub, pos, args, fold, mean=None, std=None, feats='aug', compute_norms=False,
                     fold_idx_set_train='train', fold_idx_set_test='test'):
    data_path = Path(data_path)
    if mean is None and std is None:
        if compute_norms == False:
            mean, std = load_fold_norm_params(args, pos, fold)
        else:
            # If you want to comupte mena and std on the fly
            mean, std = None, None

    if fold_idx_set_test == 'all':
        # we dont need to split the data into train and test sets
        dataset = AT_GreatFolded(args, data_path, sub, pos, transform=True, target_transform=True, 
                             fold=fold, datafold_path=args['fold_config'], get_fold_set=fold_idx_set_test, 
                             mean=mean, std=std, feats=feats)
        
    else:
        
        train_dataset = AT_GreatFolded(args, data_path, sub, pos, transform=True, target_transform=True, 
                                fold=fold, datafold_path=args['fold_config'], get_fold_set=fold_idx_set_train, 
                                mean=mean, std=std, feats=feats)
        
        val_dataset = AT_GreatFolded(args, data_path, sub, pos, transform=True, target_transform=True,
                                    fold=fold, datafold_path=args['fold_config'], get_fold_set=fold_idx_set_test, 
                                    mean=mean, std=std, feats=feats)
        
        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args['batch_size'])



    if args['lda'] == True and fold_idx_set_test == 'all':
        print('return mode: all')
        return dataset, mean, std
        # pass
    elif args['lda'] == True:
        return train_dataset, val_dataset, mean, std
    elif args['norm_stats'] == 'independant':
        return train_loader, val_loader
    elif args['norm_stats'] == 'source_domain_statisitcs':
        return train_loader, val_loader,  mean, std 
'''



def get_foldedtrialdata_loaders(args, 
                                fold, 
                                feats='aug', 
                                compute_norms=False, 
                                fold_idx_set_train=['train','train'],
                                fold_idx_set_test=['test', 'test'], 
                                **kwargs):
    
    if args['task'] == 'source_only':
        s_train_loader, s_test_loader= fold_loader_init(args['source']['data_path'], 
                                                    args['source']['sub'], args['source']['pos'], 
                                                    args, fold, feats=feats, mean=None, std=None,
                                                    compute_norms=compute_norms,fold_idx_set_train=fold_idx_set_train[0], 
                                                    fold_idx_set_test=fold_idx_set_test[0], data_mode='source')
        return s_train_loader, s_test_loader


    elif args['task'] == 'transfer_learning':
        s_train_loader, s_test_loader, mean, std  = fold_loader_init(args['source']['data_path'], 
                                                                    args['source']['sub'], args['source']['pos'], 
                                                                    args, fold, mean=None, std=None, feats=feats,
                                                                    compute_norms=compute_norms,fold_idx_set_train=fold_idx_set_train[0], 
                                                                    fold_idx_set_test=fold_idx_set_test[0], data_mode='source')
        
        t_train_loader, t_test_loader   = fold_loader_init(args['target']['data_path'], args['target']['sub'],
                                                            args['target']['pos'], args, fold, mean, std, feats=feats,
                                                            compute_norms=compute_norms, fold_idx_set_train=fold_idx_set_train[1], 
                                                            fold_idx_set_test=fold_idx_set_test[1], data_mode='target') 

        return (s_train_loader, s_test_loader, t_train_loader, t_test_loader)
    


#--------------------------------------------------------------------------------
# --------------------- FOR PROTOTYPICAL NEURAL NETWORK -------------------------
def get_subset_loaders(config, dataset, file, fold=0): # config['fold'] might want tot include it
    '''
    Retrieve loaders for subset of the dataset depending on the k-fold.
    '''

    idxs = read_fold_data(file=file)
    # change this: config['k_folds_file']
    # in teh fn above, inject the fold file into the config for Source only and transfer learning

    # TODO: decide where I want to override it, in the fn param or in the config
    config['fold'] = fold 

    # Sample elements randomly from a given list of ids, no replacement.
    train_source_dataset = Subset(dataset, idxs[str(config['fold'])]['train_ids'])
    val_source_dataset = Subset(dataset, idxs[str(config['fold'])]['test_ids'])

    loader = DataLoader(train_source_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_source_dataset, batch_size=config['batch_size'])
    return loader, val_loader

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------


'''
def loader_init(data_path, sub, pos, args, dataset=None):
    data_path = Path(data_path)
    if dataset is None:
        dataset = AT_Great(args, data_path, sub, pos, transform=True, target_transform=None)
    # technically, once I have the dataset init, I could just run subsetloaders ona fold... 

    train_loader, test_loader = get_fold_subset_loaders(args, dataset) 
    dataset.data_mean, dataset.data_std = dataset.get_norms()
    return train_loader, test_loader, dataset


def get_trialdata_loaders(args, datasets:list):
    if args['task'] == 'source_only':
        dataset = datasets[0]
        s_train_loader, s_test_loader, dataset=  loader_init(args['source']['data_path'], 
                                                    args['source']['sub'], args['source']['pos'], args, 
                                                    dataset=dataset)
        return s_train_loader, s_test_loader, dataset
    
    elif args['task'] == 'transfer_learning':
        s_dataset = datasets[0]
        t_dataset = datasets[1]
        s_train_loader, s_test_loader, s_dataset = loader_init(args['source']['data_path'], 
                                                    args['source']['sub'], args['source']['pos'], args, s_dataset)
        t_train_loader, t_test_loader, t_dataset = loader_init(args['target']['data_path'], args['target']['sub'],
                                                    args['target']['pos'], args, t_dataset) 
        return(s_train_loader, s_test_loader, t_train_loader, t_test_loader, s_dataset, t_dataset)
    

def calculate_mean_std(dataset, indices):
    """
    Calculates mean and standard deviation from a subset of the dataset.
    """
    torch.mean(dataset[indices[0]][0], dim=0)
    
    data_sum = torch.zeros_like(dataset[indices[0]][0])  # Initialize with same shape as a data sample
    
    for idx in indices:
        sample, _ = dataset[idx]
        data_sum += sample

    mean = data_sum / len(indices)
    std = torch.std(data_sum, dim=0) / len(indices)  # Calculate per-channel std
    print('lol mean', mean.shape, 'std', std.shape)
    return mean, std



def get_fold_subset_loaders(config, dataset, file=None, fold=0): # config['fold'] might want tot include it
    
    # Retrieve loaders for subset of the dataset depending on the k-fold.
    
    
    # change this: config['k_folds_file']
    # in teh fn above, inject the fold file into the config for Source only and transfer learning

    if 'fold' in config:
        fold = config['fold']   
    else:
        fold = fold


    if file is not None:
        idxs = read_fold_data(file=file)
        train_idx = idxs[str(config['fold'])]['train_ids']
        test_idx = idxs[str(config['fold'])]['test_ids']
        
    else:
        train_idx, test_idx = dataset.get_fold_idxs(fold=fold)

    
    dataset.data_mean, dataset.data_std = calculate_mean_std(dataset, train_idx)
    # does this overwrite the mena and std on the dataset?
    train_source_dataset = Subset(dataset, train_idx)
    val_source_dataset = Subset(dataset, test_idx)
    print('length', train_source_dataset.__len__(), val_source_dataset.__len__() )

    # Sample elements randomly from a given list of ids, no replacement.
    # either make my wn subset or make a parent class
    # mean_train, std_train = calculate_mean_std(dataset, train_idx)
    # the mean and std are calculated on the entire dataset instead of only training set
    # it will be an issue for source_only setting
    # train_source_dataset_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=dataset.data_mean, std=dataset.data_std)
    # ])
    # val_source_dataset_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=dataset.data_mean, std=dataset.data_std)
    # ])

 
    loader = DataLoader(train_source_dataset, batch_size=config['batch_size'], shuffle=True )
    val_loader = DataLoader(val_source_dataset, batch_size=config['batch_size'])
    return loader, val_loader




# need to import easyfsl for PNN
# def task_sampler_loaders(args):
#     train_sampler = TaskSampler(train_set1, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS)
#     target_sampler = TaskSampler(target_set1, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS)
# ____________________________________________________________________________________
# ____________________________________________________________________________________
# ____________________________________________________________________________________
# def get_data_loaders(args):
#     
#     Retrieve loaders for source only or transfer learning setting depending on the k-fold.
#    
#     #TODO: Add the fold to the config file
#     fold=args['fold']

#     if args['task'] == 'source_only':
#         s_data_path = Path(args['source']['data_path'])
#         dataset = Base_Great(args, s_data_path, args['source']['sub'], 
#                             args['source']['pos'], transform=None, target_transform=None)
#         return get_subset_loaders(args, dataset, args['sourceonly_kfolds_file'])
    
#     elif args['task'] == 'transfer_learning':
#         s_data_path = Path(args['source']['data_path'])
#         t_data_path = Path(args['target']['data_path'])
#         train_dataset = Base_Great(args, s_data_path, args['source']['sub'], 
#                                  args['source']['pos'], transform=None, target_transform=None)
#         val_dataset = Base_Great(args, t_data_path, args['target']['sub'], 
#                                args['source']['pos'], transform=None, target_transform=None)
#         source_train_loader, source_test_loader = get_subset_loaders(args, train_dataset, args['sourceonly_kfolds_file'])
#         target_train_loader, target_test_loader = get_subset_loaders(args, val_dataset, args['tl_kfolds_file'])
#         return(source_train_loader, source_test_loader, target_train_loader, target_test_loader)
        

# def get_data_subset_idxs(args:dict):

#     if args['task'] == 'source_only':
#         s_data_path = Path(args['source']['data_path'])
#         dataset = Base_Great(args, s_data_path, args['source']['sub'], 
#                             args['source']['pos'], transform=None, target_transform=None)
#         subset_idxs_kfold_save(dataset, args, s_data_path, file=args['sourceonly_kfolds_file'])
    
#     elif args['task'] == 'transfer_learning':
#         s_data_path = Path(args['source']['data_path'])
#         t_data_path = Path(args['target']['data_path'])
#         train_dataset = Base_Great(args, s_data_path, args['source']['sub'], 
#                                  args['source']['pos'], transform=None, target_transform=None)
#         val_dataset = Base_Great(args, t_data_path, args['target']['sub'], 
#                                 args['target']['pos'], transform=None, target_transform=None)
#         subset_idxs_kfold_save(train_dataset, args, s_data_path, file=args['sourceonly_kfolds_file'])
#         subset_idxs_kfold_save(val_dataset, args, t_data_path, file=args['tl_kfolds_file'])
    

# from pathlib import Path
# if __name__ == '__main__':
    
#     args = get_configs(os.path.join(os.getcwd(), 'great', 'experiment_configs', 'data_configs', 'great.yaml'))
#     current_dir = os.getcwd()

#     # TODO
#     # save the fold ids for the kfold for source only or transfer learning setting
#     get_data_subset_idxs(args)

#     # retieve the fold ids for the kfold for source only or transfer learning setting
#     loaders = get_data_loaders(args)

#     # TODO: seperate model and data yaml


    # k_folds = args['k_folds']
    # skf = stratkfold(dataset, args, data_path=data_path)

    # for fold, (train_ids, test_ids) in enumerate(skf): # .split(X, y)
    #     # Print
    #     print(f'FOLD {fold}')
    #     print('--------------------------------')
        
    #     # Sample elements randomly from a given list of ids, no replacement.
    #     train_source_dataset = Subset(dataset, train_ids)
    #     val_source_dataset = Subset(dataset, test_ids)
    #     # print(torch.unique(y[train_ids], return_counts=True))
    #     # print(torch.unique(y[test_ids], return_counts=True))


    # # Use labels from both target components for stratified sampling
    # target_labels = torch.cat((target_dataset.y_grasp, target_dataset.y_pos), dim=1)

    # skf = StratifiedKFold(n_splits=5)

    # for train_index, val_index in skf.split(target_dataset, target_labels):
    #     # Subset both source and target datasets based on indices
    #     train_source_dataset = Subset(source_dataset, train_index)
    #     val_source_dataset = Subset(source_dataset, val_index)
        
    #     train_target_dataset = Subset(target_dataset, train_index)
    #     val_target_dataset = Subset(target_dataset, val_index)

    #     # Create data loaders for both source and target
    #     train_source_loader = DataLoader(train_source_dataset, batch_size=32, shuffle=True)
    #     val_source_loader = DataLoader(val_source_dataset, batch_size=32)

    #     train_target_loader = DataLoader(train_target_dataset, batch_size=32, shuffle=True)
    #     val_target_loader = DataLoader(val_target_dataset, batch_size=32)

    #     # Combine loaders during training (if applicable)
    #     combined_train_loader = (train_source_loader, train_target_loader)
    #     combined_val_loader = (val_source_loader, val_target_loader)

    #     # Train and evaluate your model with combined loaders
    #     # ...

    #     # Track and average performance metrics across folds
    #     # ...







# def loader_init(data_path, sub, pos, args):
#     data_path = Path(data_path)
#     dataset = AT_Great(args, data_path, sub, pos, transform=True, target_transform=None)
#     # technically, once I have the dataset init, I could just run subsetloaders ona fold... 
    
#     train_loader, test_loader = get_fold_subset_loaders(args, dataset) 
#     return train_loader, test_loader

# def get_trialdata_loaders(args):
#     if args['task'] == 'source_only':
#         return loader_init(args['source']['data_path'], 
#                            args['source']['sub'], args['source']['pos'], args)
    
#     elif args['task'] == 'transfer_learning':
#         s_train_loader, s_test_loader = loader_init(args['source']['data_path'], 
#                                                     args['source']['sub'], args['source']['pos'], args)
#         t_train_loader, t_test_loader = loader_init(args['target']['data_path'], args['target']['sub'],
#                                                     args['target']['pos'], args) 
#         return(s_train_loader, s_test_loader, t_train_loader, t_test_loader)
    
# def get_trialdata_loaders(args):
#    
#     #TODO: Add the fold to the config file
#     fold=args['fold']

#     if args['task'] == 'source_only':
#         s_data_path = Path(args['source']['data_path'])
#         dataset = AT_Great(args, s_data_path, args['source']['sub'], 
#                             args['source']['pos'], transform=True, target_transform=None)
#         return get_fold_subset_loaders(args, dataset) #, args['sourceonly_kfolds_file'])
    
#     elif args['task'] == 'transfer_learning':
#         s_data_path = Path(args['source']['data_path'])
#         t_data_path = Path(args['target']['data_path'])
#         train_dataset = AT_Great(args, s_data_path, args['source']['sub'], 
#                                  args['source']['pos'], transform=True, target_transform=None)
#         val_dataset = AT_Great(args, t_data_path, args['target']['sub'], 
#                                args['target']['pos'], transform=None, target_transform=None)
#         source_train_loader, source_test_loader = get_fold_subset_loaders(args, train_dataset, fold=0) # args['sourceonly_kfolds_file'])
#         target_train_loader, target_test_loader = get_fold_subset_loaders(args, val_dataset, fold=0) # args['tl_kfolds_file'])
#         return(source_train_loader, source_test_loader, target_train_loader, target_test_loader)
'''