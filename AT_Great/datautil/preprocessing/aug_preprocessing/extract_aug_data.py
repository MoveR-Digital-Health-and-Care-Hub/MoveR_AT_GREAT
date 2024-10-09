import os
import numpy as np
from AT_Great.datautil.preprocessing.dataIO import DataReadWrite as dataIO
from AT_Great.datautil.preprocessing.read_raw import get_data, get_trial_pos_grasp
from AT_Great.experiment_configs.utils import get_configs
from scipy.signal import welch
import torch
from AT_Great.datautil.preprocessing.aug_preprocessing.features import *
from AT_Great.datautil.preprocessing.aug_preprocessing.windower import *
from scipy.signal import iirnotch, lfilter

ROOT_DIR = os.getcwd()
config_path = os.path.join(ROOT_DIR, 'AT_Great', 'experiment_configs', 'config_extract_feats.yaml')
# /home/kasia/AT_Great/AT_Great/experiment_configs/config_extract_feats.yaml
configs = get_configs(config_path)
print('Current working directory:', os.getcwd(), ROOT_DIR)


def run_trial_preprocess(data_dict, y_dict, configs, _dir, 
                                                        day_block, sub_prime, no_feats=9, no_sensors=16, 
                                                        rolling_window=False):
    
    assert _dir == 'processed_data_144x256feats_augstride10and10' if rolling_window else 'processed_data_16x9feats'

    unique_positions = np.unique([y_dict[i]['position'] for i in y_dict]).tolist()
    sorted_keys = sorted(y_dict.keys(), 
                                     key=lambda x: unique_positions.index(y_dict[x]['position']))
    
    for position in unique_positions:
        position_keys = [key for key in sorted_keys if y_dict[key]['position'] == position]

        # for each position, create a file and save data
        if rolling_window:
            final_data, y = np.empty((0, (no_sensors*no_feats), configs['window_size'])), np.empty((0,3))  # 144x256
        else: 
            final_data, y = np.empty((0, no_sensors, no_feats)), np.empty((0,3)) # 16x9
        # for each trial, slide window and extract feats
        for i in position_keys:
            temp_dir = day_block + '_position_' + str(y_dict[i]['position'])
            temp_x = data_dict[str(i)][:]
            

            if rolling_window:
                # this is for rolling window for augmented data - 144x256
                up_data = extract_feats(temp_x, configs['window_size'], stepsize=configs['stride2'], padded=False, copy=True)
               
                up_data = up_data.reshape(up_data.shape[0], -1)
                X = sliding_window(up_data, configs['window_size'], stepsize=configs['stride2'], padded=False, axis=0, copy=True)
                print('key:', i, ' X shape:', X.shape)
            else:
                up_data = extract_feats(temp_x, configs['window_size'], stepsize=configs['stride'], padded=False, copy=True)
                X = up_data

            # create labels
            y_temp = get_lbl(X, i, y_dict)
            y = np.vstack((y, y_temp))
            final_data = np.vstack((final_data, X))
            
            
        # save the extracted data and labels
        save_path = os.path.join(ROOT_DIR,'AT_Great', 'data',  _dir, sub_prime)
        print('----------------- final data shape:', final_data.shape, 'label data shape:', y.shape, '----------------- ')
        # save_data(torch.tensor(final_data), torch.tensor(y), save_path, temp_dir)


def get_lbl(X, pos_key, y_dict):
    y_temp = np.zeros((X.shape[0], 3))
    y_temp[:,0] = pos_key
    y_temp[:,1] = y_dict[pos_key]['grasp']
    y_temp[:,2] = y_dict[pos_key]['position']
    return y_temp


def save_data(data: torch.Tensor, y: torch.Tensor, save_path: str, filename: str):
    if dataIO().check_dir_exist(save_path) is False:
        os.mkdir(save_path)

    torch.save(torch.tensor(data), os.path.join(save_path, filename + '.pt'))
    torch.save(torch.tensor(y), os.path.join(save_path, filename + '_y.pt'))


def roll_tile_window(x_data, num_steps=256):

    num_samples, num_features = x_data.shape
    X, y = np.empty((0, num_steps, num_features)), np.empty((0,))
    for i in range(num_samples - num_steps + 1): 
        end_ix = i + num_steps
        seq_X = x_data[i:end_ix]
        # seq_y = y_data[end_ix - 1]
        X = np.vstack((X, seq_X[np.newaxis, ...]))
        # y = np.append(y, seq_y)
    return X


def notch_filter(x, quality_factor=30, fs=2000.0):
    freq_filterout = 50
    b,a = iirnotch(freq_filterout, quality_factor, fs=fs)
    # filter signal out
    signal = lfilter(b, a, x, axis=-1) # check what axis to apply this to!
    return signal


def extract_feats(data, window_size, stepsize=1, padded=False, copy=True):
        # data cleaning
        data = notch_filter(data)
        data = Offset_Correction(data.T)
  
        # extract feats
        param1 = MAV(data, window_size, stepsize)  
        param2 = ZC(data, window_size, stepsize)  
        param3 = SSC(data, window_size, stepsize)  
        param4 = WL(data, window_size, stepsize)  
        param5 = LV(data, window_size, stepsize)
        param6 = SKW(data, window_size, stepsize)
        param7 = MNF(data, window_size, stepsize)
        param8 = PKF(data, window_size, stepsize)
        param9 = VCF(data, window_size, stepsize)
        extracted = np.dstack((param1, param2, param3, param4, param5, param6, param7, param8, param9))
        return extracted
    
def main():
    project_path = os.path.join(os.getcwd(), 'AT_Great') 
    print('project_path', project_path)
    for i in configs['subjects']: 

        # do only sub 7!
        if int(i) != int(7): 
            continue
        k=0
        print('Subject :', i)
        for day in configs['days']: 
            for block in configs['blocks']:
                print('K:', k, 'Day:', day, 'Block:', block)  
                sub_prime = 'participant_' + str(i)
                sub = 'participant' + str(i)+'_'
                d = 'day'+str(day) + '_block' + str(block)

                read_path = os.path.join("data", 'data', sub_prime, str(sub+d))
                

                # 1. Load data - single subject/participant
                p = os.path.join(project_path, read_path, "emg_raw.hdf5")
                print('p:', p)
                p_label = os.path.join(project_path, read_path, "trials.csv")
                trials_keys, data = get_data(p)
                lbls_dict = get_trial_pos_grasp(p_label)
                print(f'read_path: {read_path}') #, trial keys {trials_keys}')
            

                folder_name =  'processed_data_144x256feats_augstride10and10'  # 'processed_data_16x9feats' # c
                run_trial_preprocess(data, lbls_dict, configs, folder_name, 
                                     day_block = d, sub_prime=sub_prime,
                                     rolling_window=True) # False)

                k+=1
                



if __name__ == "__main__":
    print('os.path.split(os.getcwd())[0]', os.path.split(os.getcwd())[0])
    main()

# if __name__ == '__main__':
#     import h5py
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from AT_Great.datautil.preprocessing.aug_preprocessing import *
#     from AT_Great.datautil.preprocessing.read_raw import get_data, get_trial_pos_grasp
#     import os
#     data_path = r'/home/kasia/AT_Great/AT_Great/data/data/participant_1/participant1_day1_block1/emg_raw.hdf5'
#     p_label = r'/home/kasia/AT_Great/AT_Great/data/data/participant_1/participant1_day1_block1/trials.csv'

#     ROOT_DIR = os.getcwd()
#     config_path = os.path.join(ROOT_DIR, 'AT_Great', 'experiment_configs', 'config_extract_feats.yaml')
#     # /home/kasia/AT_Great/AT_Great/experiment_configs/config_extract_feats.yaml
#     configs = get_configs(config_path)

#     trials_keys, data = get_data(data_path)
#     win_size = 256
#     win_stride = 100
#     lbls_dict = get_trial_pos_grasp(p_label)

#     run_trial_preprocess_PROCESSDATA_NOEXTRA_SLIDINGWIN(data, lbls_dict, configs, 
#                                                         None, day_block = 'l', sub_prime=None,)