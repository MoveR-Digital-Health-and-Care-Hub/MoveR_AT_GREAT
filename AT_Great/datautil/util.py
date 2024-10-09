import glob
import json
from natsort import natsorted
import os
import torch
import numpy as np

from sklearn.model_selection import StratifiedKFold

def get_files(args, data_path):
    
    file_list = glob.glob(data_path + '/*')
    #seperate x and y position files
    position_y_files = natsorted([file for file in file_list if os.path.basename(file).split('_')[-1][0] == 'y'])
    position_files = natsorted([file for file in file_list if os.path.basename(file).split('_')[-1][0] != 'y'])

    filtered_y_files = [file for file in position_y_files if os.path.basename(file).split("_")[3] in args['pos']] 
    filtered_x_files =[file for file in position_files if os.path.basename(file).split("_")[3][0] in args['pos']] 
    
    if 'block' in args:
        filtered_y_files = [file for file in filtered_y_files if os.path.basename(file).split("_")[1] in args['block']]
        filtered_x_files = [file for file in filtered_x_files if os.path.basename(file).split("_")[1] in args['block']]

    if 'day' in args:
        filtered_y_files = [file for file in filtered_y_files if os.path.basename(file).split("_")[0] in args['day']]
        filtered_x_files = [file for file in filtered_x_files if  os.path.basename(file).split("_")[0] in args['day']]

    return filtered_x_files, filtered_y_files


def read_fold_data(file='fold_data.json'):
    with open(file, 'r') as file:
        fold_data = json.load(file)
    return fold_data


def subset_idxs_kfold_save(dataset, config, data_path, file='fold_data.json'):
    skf = stratkfold(dataset, config['k_folds'], data_path=data_path)
    fold_data = {}
    for fold, (train_ids, test_ids) in enumerate(skf):
        print(f'FOLD {fold}')
        print('--------------------------------')
        #fold_data[fold] = {'train_ids': train_ids, 'test_ids': test_ids}
        fold_data[fold] = {'train_ids': train_ids.tolist(), 'test_ids': test_ids.tolist()}

    with open(file, 'w') as file:
        json.dump(fold_data, file)


def stratkfold(dataset, folds, data_path):
    skf = StratifiedKFold(n_splits=folds, random_state=None, shuffle=False)
    X = dataset.get_data(data_path)
    y, _ = dataset.get_lbl(data_path)
    return skf.split(X, y)

def flatten(lst):
    result = []
    for i in lst:
        if isinstance(i, list):
            result.extend(flatten(i))
        else:
            result.append(i)
    return result



def update_lbl_trials(data_path, config, position):
    '''
    Update the trial values in the lbl tensor. The trial values are updated to be unique across all the blocks 
    (2 blocks in each position have hte same trial keys - w.r.t. data recording protocol).
    The counter is then reset and start from 0 for the next position. 
    This is important for the fold generation as the trial values are used as keys to split the data into folds.
    The lbl values are still unique, having [trial, label, position] as the key.
    '''
    
    final_lbl = torch.empty(0, 3) # dim: trial, label, position

    # Can extend to multiple subjects if needed
    for pos in position:
        global_index = 0
        for block in config['pos_day_block'][pos]:
            lbl = torch.load(os.path.join(data_path, f'{block}_position_{pos}_y.pt'))
            
            temp_arr = lbl.clone() 
            temp_trials = torch.zeros(temp_arr.shape[0])
            unique_values = torch.unique(temp_arr[:, 0]) 
            no_unique_trials = len(unique_values)
            # print('Block:', block, 'Unique trials:', no_unique_trials)

            for value in unique_values:
                indices = torch.where(temp_arr[:,0] == value)[0]
                temp_trials[indices] = global_index
                global_index += 1

            temp_arr[:,0] = temp_trials
            final_lbl = torch.cat([final_lbl, temp_arr], dim=0)
            # print('final lbl shape', final_lbl.shape, 'final lbl trials:', len(torch.unique(final_lbl[:, 0])))
            # print('final unique:', torch.unique(final_lbl[:, 0], return_counts=True))
    return final_lbl


def data_modif(inputs, labels, device, config):
    '''
    Depending on the tpe of input, the iput tensor is modified in trainig and testing fn().
    Can be later integrated into the dataset class.
    '''
    
    if config['feats'] == 'ftd_base':
        # 1D input
        # inputs = inputs.view(inputs.size(0), -1).
        inputs = inputs.to(device)
    elif config['feats'] == 'aug':
        # 2D input
        inputs = inputs.unsqueeze(3).to(device)
    
    labels = labels.to(device)
    return inputs, labels



def map_pos(arr):
    '''
    Function to map the position values to unique integers. Used in AT_GREAT repo.
    '''
    unique_values = np.unique(arr)
    mapping = {value: i+1 for i, value in enumerate(unique_values)}
    arr = np.vectorize(mapping.get)(arr)

    # print(arr)
    return arr
