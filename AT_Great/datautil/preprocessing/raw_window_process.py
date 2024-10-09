'''
This script when the extract_data_oldwindower.py is used to extarct feats into separate .npy files
This script merges them into a single tensor and saves them to a single .pt file.
It also creates a dictionary with labels and extra information.

'''

import os
import pickle
import torch
import numpy as np
import os
import h5py
import fnmatch
# import sys
# sys.path.append('source/configs/')
from AT_Great.experiment_configs.utils import get_configs
import torch



args = get_configs(os.path.join(os.getcwd(), 'AT_Great', 'experiment_configs', 'data_configs', 'great.yaml'))

# scrape files from the folder for a single subejct
def scrape_files(path, pattern):
    # Use to find the fiels from a predefined position and list them
    return [f for f in os.listdir(path) if fnmatch.fnmatch(f, f'*{pattern}*')]


# load meta data and labels to dictionary
def load_todict(files, dataset={}, idx = 0):
    '''
    Load to the dictionary, the key is the global index. 
    Global since, we combine two sessions together.

    '''
    for filename in files:
        # block = NotImplemented 
        print('filename: ', filename)
        if filename.endswith(".npy"):
            data = np.load(filename)

            key = os.path.split(filename)[-1].split('.')[0]
            trial_key, counter, temp_lbl, position = key.split('_')
            dataset[idx] = {
                'data': data,
                'trial_key': trial_key,
                'counter': int(counter),
                'temp_lbl': int(temp_lbl),
                'position': position, 
                'block': filename.split(os.sep)[-3],
            }
        print('dataset: ', dataset)
        idx += 1 
        
    return dataset, idx 

# save data to h5py
def save_dict_to_hdf5(dataset, filename):
    with h5py.File(filename, 'w') as f:
        for key, value in dataset.items():
            if isinstance(value, dict):
                group = f.create_group(key)
                for sub_key, sub_value in value.items():
                    group.create_dataset(sub_key, data=sub_value)
            else:
                f.create_dataset(key, data=value)



# find files
def find(pattern, path):
    # Use to find the fiels from a predefined position and list them
    return [f for f in os.listdir(path) if fnmatch.fnmatch(f, f'*{pattern}*')]



# def sub_dict(path):

#     dict_sub = {}
#     for day in var:
#         block_list = None
#         for block in day:
#             print('block: ', block)
#             for pos, val in var_pos.items():
#                 path = ''
#                 for p in val:
#                     temp_pos = 'position_' + str(p)
#                     path = os.path.join(path, block, temp_pos)
#                     print('pos: ', pos, 'path: ', path)
#                     if not os.path.exists(path):
#                         print(f'Path does not exist: {path}\n')
#                         continue



'''
                #     if block == pos:
                #         print('block: ', block, 'pos: ', pos)
                # temp_pos = 'position_' + str(pos)
                
                # path = os.path.join(path, block, temp_pos)
                
                # if not os.path.exists(path):
                # #     print(f'Path does not exist: {path}\n')
                #      continue
                # # else:
                    

                # if block_list is None:
                #     block_list = find(str(pos), path)
                # else:
                #     block_list.extend(find(str(pos), path))

                # print('block_list: ', block_list)
                # dict_sub.update({pos: load_todict(block_list, path)})

    # save_dict_to_hdf5(dataset, block+'.h5')

# def list_posfiles(path):
#     _list = []
#     for v in var:
#         dir_path = os.path.join(path, v)
#         for pos in var_pos[v]:
#             temp_p = 'position_' + str(pos)
#             dir_pos = os.path.join(dir_path, temp_p)
#             print(dir_pos)
#             _list.append(dir_pos)
#     return _list
'''


def write_pickle(_dict, path):
    # Save
    with open(path, 'wb') as f:
        pickle.dump(_dict, f)

def read_pickle(path):
    # Load the dictionary
    with open(path, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

def list_subdirs(subs_path):
    return [os.path.join(subs_path, d) for d in os.listdir(subs_path) if os.path.isdir(os.path.join(subs_path, d))]


def merge_pos_data(path, sub_save_dir, sub_name):
    pos_list = args['var_pos'][os.path.basename(path)]
    print('pos_list: ', pos_list)
    positions = list_subdirs(path)
    block = os.path.basename(path)
    for poss in positions:
        get_pos_data(poss,  sub_save_dir, block)


# save position files to a single tensor
def get_pos_data(path, sub_save_dir, block):
    # Load the numpy files and convert them to tensors
    file_list = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.npy')]
    
    tensors = [torch.from_numpy(np.load(os.path.join(path, file))) for file in os.listdir(path) if file.endswith('.npy')]
    # Stack the tensors along a new dimension
    stacked_tensors = torch.stack(tensors)
    save_to = os.path.join(sub_save_dir, block+'_'+os.path.basename(path)+'.pt')
    torch.save(stacked_tensors, save_to)

    # save the lbl of the data 
    get_pos_lbl_data(file_list, sub_save_dir, block, path)


def get_pos_lbl_data(file_names, sub_save_dir, block, pos):

    # Split each file name, remove the extension, take the last two parts, and convert to tensor
    tensors = [torch.tensor([int(part) for part in os.path.splitext(file_name)[0].split('_')[-2:]]) for file_name in file_names]
    stacked_tensors = torch.stack(tensors)
    save_to = os.path.join(sub_save_dir, block+'_'+os.path.basename(pos)+'_y.pt')
    torch.save(stacked_tensors, save_to)


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)    


def main():
    # root dir
    root_dir = os.path.split(os.getcwd())[0]
    
    #subject path
    subs_path = os.path.join(root_dir, 'PycharmProjects','arm_translation_dataset','processed_data','hudgins256_100')
    subs = list_subdirs(subs_path)
    #print('subs: ', subs)

    # single subjects blocks
    # sub1 = subs[0]
    for sub1 in subs:
        sub_name = os.path.basename(sub1)
        sub_save_dir = os.path.join(root_dir, 'robustlearn', 'diversify','data','arm_translation', sub_name)
        check_dir(sub_save_dir)

        block_dirlist = list_subdirs(sub1)
        for block in block_dirlist:
            merge_pos_data(block, sub_save_dir, sub_name)

        # print('subs: ', block_dirlist)
        # print('var: ', main_configs.var)

    
    
if __name__ == '__main__':
    main()

