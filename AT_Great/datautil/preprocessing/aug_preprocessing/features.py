import numpy as np
from scipy import signal
from AT_Great.datautil.preprocessing.aug_preprocessing.windower import *
from scipy.stats import skew
import math
import scipy

def Normalise(x):
    # return (x-np.min(x))/(np.max(x)-np.min(x))
    return (x - np.mean(x)) / np.std(x)  # tested better

def Offset_Correction(x):
    return x - np.mean(x, axis=0)
    
def slice(x, win_size, win_stride, axis=0): # it was  axis=-1
    # x = np.vstack((np.zeros(((win_size - 1), x.shape[-1])), x))
    # print('input shape:', x.shape, 'win_size:', win_size, 'win_stride:', win_stride, 'axis:', axis)
    x = sliding_window(x, size=win_size, stepsize=win_stride, axis=axis)
    # print('sliding window shape:', x.shape)
    return x
def RAW(x, win_size, win_stride):
    return x
def RFT(x, win_size, win_stride):
    return np.abs(x)
def WL(x, win_size, win_stride):
    x = slice(x, win_size, win_stride)
    return np.sum(np.abs(np.diff(x, axis=-1)), axis=-1) #/win_size
def LV(x, win_size, win_stride):
    x = slice(x, win_size, win_stride)
    return np.log10(np.var(x, axis=-1)) #/win_size
def RMS(x, win_size, win_stride):
    x = slice(x, win_size, win_stride)
    return np.sqrt(np.mean(np.square(x), axis=-1))

def MAV(x, win_size, win_stride):
    x = slice(x, win_size, win_stride)
    # print('MAV shape:', x.shape)
    return np.mean(np.abs(x), axis=-1)

def ZC(x, win_size, win_stride, thr=0.00):
    x = slice(x, win_size, win_stride)
    return np.sum(((np.diff(np.sign(x), axis=-1) != 0) & (np.abs(np.diff(x, axis=-1)) >= thr)), axis=-1)
def SSC(x, win_size, win_stride, thr=0):
    x = slice(x, win_size, win_stride)
    return np.sum((((x[:,:,1:-1] - x[:,:,:-2]) * (x[:,:,1:-1] - x[:,:,2:])) > thr), axis=-1)
def SKW(x, win_size, win_stride):
    x = slice(x, win_size, win_stride)
    return skew(x, axis=-1, bias=True, nan_policy='propagate')
def MNF(x, win_size, win_stride):
    x = slice(x, win_size, win_stride)
    nfft_value = int(np.power(2, np.ceil(math.log(win_size, 2))))
    f, pxx = scipy.signal.welch(x, 2000, window='hamming', nperseg=win_size, noverlap=None, nfft=nfft_value, detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')
    return np.sum(f[None,None,:] * pxx, axis=-1) / np.sum(pxx, axis=-1)

def MDF(x, win_size, win_stride):
    x = slice(x, win_size, win_stride)
    nfft_value = int(np.power(2, np.ceil(math.log(win_size, 2))))
    f, pxx = scipy.signal.welch(x, 2000, window='hamming', nperseg=win_size, noverlap=None, nfft=nfft_value, detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')
    mda = 1/2 * np.sum(pxx, axis=-1)
    pxx_cumsum = np.cumsum(pxx, axis=-1)
    mdi = np.argmin(np.abs(pxx_cumsum-mda[:,:,None]), axis=-1)
    return f[mdi]
def PKF(x, win_size, win_stride):
    x = slice(x, win_size, win_stride)
    nfft_value = int(np.power(2, np.ceil(math.log(win_size, 2))))
    f, pxx = scipy.signal.welch(x, 2000, window='hamming', nperseg=win_size, noverlap=None, nfft=nfft_value, detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')
    mxi = np.argmax(pxx, axis=-1)
    return f[mxi]
def VCF(x, win_size, win_stride):
    x = slice(x, win_size, win_stride)
    nfft_value = int(np.power(2, np.ceil(math.log(win_size, 2))))
    f, pxx = scipy.signal.welch(x, 2000, window='hamming', nperseg=win_size, noverlap=None, nfft=nfft_value, detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')
    sm0 = np.sum(pxx * np.power(f, 0)[None,None,:], axis=-1)
    sm1 = np.sum(pxx * np.power(f, 1)[None,None,:], axis=-1)
    sm2 = np.sum(pxx * np.power(f, 2)[None,None,:], axis=-1)
    return sm2/sm0-np.power((sm1/sm0),2)
def Envelope(x, win_size, win_stride):
    # need an analog filter
    # x = slice(x, win_size, win_stride)
    x = np.abs(x)
    # o = []
    # for i in range(int((len(x)-win_size)//win_stride+1)):
    #     o.append(np.max(x[i*win_stride:i*win_stride+win_size], axis=0))
    # o = np.vstack(o)
    b, a = signal.butter(1, 2, 'lowpass', output='ba', fs=2000) #//win_stride
    x = signal.filtfilt(b, a, x, axis=0)
    return x

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
             
    # print('data shape:',  data['0'][:].shape, data['0'][:].T.shape)
    # print(f'read_path: {data_path}, trial keys {trials_keys}')
    # print(f'lbls_dict: {lbls_dict}')


    # unique_positions = np.unique([lbls_dict[i]['position'] for i in lbls_dict]).tolist()
    # sorted_keys = sorted(lbls_dict.keys(), 
    #                                  key=lambda x: unique_positions.index(lbls_dict[x]['position']))
    # print('sorted keys:', sorted_keys)
    # print('unique positions:', unique_positions)
    # for i in range(0, len(sorted_keys), 1):
    #     key = sorted_keys[i]
    #     print()
    #     print('key:', key, 'lbls_dict[key]:', lbls_dict[key])
    

    # for position in unique_positions:
    #     position_keys = [key for key in sorted_keys if lbls_dict[key]['position'] == position]

    #     # for each position, create a file and save data
    #     final_data, y = np.empty((0, 16, 9)), np.empty((0,3))
    #     # for each trial, slide window and extract feats
    #     for i in position_keys:
    #         print('Key', i)
    #         temp_dir = day_block + '_position_' + str(y_dict[i]['position'])
    #         temp_x = data_dict[str(i)][:].T 
    #         up_data = extract_feats(temp_x, configs['window_size'], stepsize=configs['stride1'], padded=False, copy=True)


    # slice(data, win_size, win_stride, axis=0)
    print('lalla')
    # import matplotlib.pyplot as plt
    # from sklearn.preprocessing import normalize
    # emg = np.load('/Users/owl/Database/Database_Nina/data2/release/s01/emg.npy')
    # input = emg[:20000, :]*10000

    # # func = [WL,LV,RMS,MAV,ZC,SSC,SKW,MNF,MDF,PKF,VCF]
    # func = [RAW]

    # res = [normalize(f(input, 200, 10)) for f in func]

    # t = np.linspace(0, len(input), len(res[0]))

    # print([res[n].shape for n in range(len(res))])
    # print([(np.max(res[n]), np.min(res[n]))for n in range(len(res))])

    # plt.figure(figsize=(20,15))

    # plt.plot(input[:, 7]-1)

    # for i in range(len(res)):
    #     plt.plot(t, res[i][:, 7]+i, label=str(func[i]))
    # plt.legend()
    # plt.show()
    