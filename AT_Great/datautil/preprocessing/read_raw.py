from operator import itemgetter
import numpy as np
import pandas as pd
import os
import glob
from natsort import natsorted
from AT_Great.datautil.preprocessing.dataIO import DataReadWrite as dataIO
import h5py
from AT_Great.datautil.preprocessing.util import flatten
# def flatten(xss):
#     return [x for xs in xss for x in xs]


def get_data_paths(data_dir = 'data'):
    participants = []

    # Loop through each participant directory
    for participant_folder in os.listdir(data_dir):
        if participant_folder.startswith('participant_'):
            participant_path = os.path.join(data_dir, participant_folder)
            blocks = []

            # Find all directories inside the participant folder that contain 'block' in their name
            block_folders = glob.glob(os.path.join(participant_path, '*block*'))

            # Loop through each block directory inside the participant folder
            for block_folder in block_folders:
                if os.path.isdir(block_folder):
                    # Check if the required files exist in the block directory
                    if 'emg_raw.hdf5' in os.listdir(block_folder) and 'trials.csv' in os.listdir(block_folder):
                        block_files = {
                            'emg_raw': os.path.join(block_folder, 'emg_raw.hdf5'),
                            'trials': os.path.join(block_folder, 'trials.csv')
                        }
                        blocks.append(block_files)

            # Add the participant's blocks to the list of all participants
            participants.append({
                participant_folder: blocks
            })
    print(participants)
    # Print the list of participants and their blocks
    return participants


def get_files_list_NEW(args, file_extension='.npy', merged_files_only=False, lbl_simple_count=False):
    data_files = None
    for dirr in args:
        # dirr_positions = [os.path.join(dirr, 'pos_{}'.format(i)) for i in positions]
        data_sub = [os.path.join(dirr, elem) for elem in os.listdir(dirr) if elem[-4:] == file_extension]

        data_sub = natsorted(data_sub)  # make sure if only one position taken, the array is flatten
        if data_files is None:
            data_files = data_sub
        else:
            data_files = data_files.extend(data_sub)
        print('data sub', data_sub)
    return data_files


def get_trial_pos_grasp(path): # used to be called get_lbls
    '''
    Read position and grasp mapping from the raw data
    :param path:
    :return: dict
    '''
    dataIO().check_dir_exist(path)
    lbls = dataIO().read_csv(path)

    lbls_dict = {}

    # get each trial block
    for i in range(0, lbls.shape[0]):
        lbls_dict.update({i: {'position': lbls['target_position'][i],
                              'grasp': lbls['grasp'][i]}})

    return lbls_dict



def get_subdirs(path):
    subdirs = []
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_dir():
                subdirs.append(entry.name)
    return subdirs


def check_dir(_path):
    if dataIO().check_dir_exist(_path) is False:
        os.mkdir(_path)


def make_participants_dirs(process_parent_dir):
    data_parent_dir = r"/data"
    participants = get_subdirs(data_parent_dir)
    blocks = ['block1', 'block2']

    for i in range(len(participants)):
        for j in range(len(blocks)):
            _path = os.path.join(process_parent_dir, participants[i], blocks[j])
            check_dir(_path)


def check_trials(x_path, y_labels, no_raw_samples=9000):
    # Check if there is N=5 trials for each grip and
    # there is approximately 10K samples (can't be less than 9K of
    # raw sEMG data)
    trials_keys, data = get_data(x_path)
    lbls = dataIO().read_csv(y_labels)
    lbls_grasp, lbls_position = [], []

    # get each trial block
    for i in range(0, lbls.shape[0]):
        lbls_position.append(lbls['target_position'][i])
        lbls_grasp.append(lbls['grasp'][i])

    position_check = np.unique(np.array(lbls_position), return_counts=True)
    grasp_check = np.unique(np.array(lbls_grasp), return_counts=True)
    print(f' position_check {position_check}, grasp_check {grasp_check}')
    # raw_len = data['150'][:].shape[1]
    for key in trials_keys:
        raw_len = data[key][:].shape[1]
        # print(key, 'raw data len:',  raw_len)
        if raw_len < no_raw_samples:
            raise ValueError(f'Data at {key} key from {x_path} '
                             f'has less than 9K samples for 2kHz and 5sec recording data.')

    # np.unique(trials_keys)
    return len(trials_keys), position_check, grasp_check


def list_files(subject_dir, raw_emg_file='emg_raw.hdf5', label_trials_file='trials.csv'):
    blocks = get_subdirs(subject_dir)
    list_femg, list_flbl = [], []
    for block in blocks:
        temp_path = os.path.join(subject_dir, block)
        emg_file = os.path.join(temp_path, raw_emg_file)
        lbl_file = os.path.join(temp_path, label_trials_file)
        list_femg.append(emg_file)
        list_flbl.append(lbl_file)
    return list_femg, list_flbl


def list_allsub_files(_dir):
    # from all sub dirs, list files for all subjects
    subjects = get_subdirs(_dir)
    list_all = []
    for elem in subjects:
        temp_path = os.path.join(_dir, elem)
        list_all.append(list_files(temp_path))
    return list_all


def get_data(path):
    '''
    Read data from the raw data and sort them using the keys
    :param path:
    :return:
    '''
    dataIO().check_dir_exist(path)
    data = h5py.File(path, "r")
    trial_keys = np.array(natsorted(data.keys()))
    return trial_keys, data

def get_pos_trialkeys(position, lbl_dict):
    trial_blocks, trial_grasps = [], []
    for key, val in lbl_dict.items():
        if lbl_dict[key]['position'] == position:
            trial_blocks.append(key)
            trial_grasps.append(lbl_dict[key]['grasp'])
    return trial_blocks, trial_grasps

