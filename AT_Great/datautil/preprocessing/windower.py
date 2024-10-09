import torch
import torchaudio.transforms as T
from scipy.signal import iirnotch, lfilter
from numpy.lib.stride_tricks import sliding_window_view as sliding_win
import os
import numpy as np
from AT_Great.datautil.preprocessing.dataIO import DataReadWrite as dataIO
from AT_Great.datautil.preprocessing.timefeats import waveform_length, wilson_amplitude, logvar, ar, mean_value, zero_crossings, \
    slope_sign_changes, root_mean_square, sample_entropy



class Windower():
    def __init__(self, mean=None, std=None, freq_filterout=50):
        self.mean = mean
        self.std = std
        self.freq_filterout = freq_filterout

    def slide_window(self, x, y, extract_function, rootdir=None, trial_key=0, position=None,
                     segment_size=600, step=300, if_norm=False, if_offset_corr=False):
        channel, counter, begin = 0, 0, 0
        class_labels = []

        # Segmented window indices for window_beginning and window_end
        end = x.shape[1]
        id1 = np.arange(begin, end, step)
        id2 = np.arange((begin + segment_size), end, step)
        diff = len(id1) - len(id2)
        id1 = id1[:-diff]

        # apply notch filter
        x = self.notch_filter(x)
        # print('Notch Filtered.')
        # print(f'x shape: {x[:8,:8]}')

        if if_norm is True:
            x = self.norm_rec(x)
            print('Normalised.')

        for j, (idx1, idx2) in enumerate(zip(id1, id2)):
            if j != 0:
                break
            # Offset correction: non-applicable when using normalisation on the entire dataset with mean_0 and std_1
            if if_offset_corr is True:
                temp_x = self.offset_correction(x, idx1, idx2)
                feats, temp_lbl = self.pipeline(extract_function, temp_x, y[idx1:idx2])
                print('Extracted')
                print(feats)
                print('transposed:\n ', feats.T)  
            else:
                feats, temp_lbl = self.pipeline(extract_function, x[:, idx1:idx2], y[idx1:idx2])  #self.pipeline(x[:, idx1:idx2], y[idx1:idx2])

            class_labels.append(temp_lbl)

            # Save data
            save_path = str(int(trial_key))+'_'+str(counter)+'_'+str(int(temp_lbl))+'_'+str(int(position))
            save_path_spec = os.path.join(rootdir, save_path)
            self.write_in(save_path_spec,'.npy', feats)

            print(j, begin, end, idx1, idx2, feats.shape)
            counter += 1

    def pipeline(self, extract_function, data, y):  # *args, **kwargs):
        '''
        write preprocessing step on (samples, channels) data
        '''
        channels = data.shape[0]
        feats = []

        for i in range(channels):
            if hasattr(self, extract_function):
                func = getattr(self, extract_function)
                feats.append(func(data[i, :]))  # *args, **kwargs)
            else:
                raise ValueError('Function not found.')

            # feats.append(self.hudgins_feats(data[i, :]))
        segment_data = (np.array(feats), max(y))
        return segment_data

    def notch_filter(self, x, quality_factor=30, fs=2000.0):
        b,a = iirnotch(self.freq_filterout, quality_factor, fs=fs)
        # filter signal out
        signal = lfilter(b, a, x, axis=-1) # check what axis to apply this to!

        return signal



    def hudgins_feats(self, data):
        # Mean Absolute Value, Zero Crossing, Slope Sign Changes and Waveform Length
        param1 = mean_value(data)  # check axis
        param2 = zero_crossings(data)  # check axis
        param3 = slope_sign_changes(data)  # check axis
        param4 = waveform_length(data)  # check axis
        args = (param1, param2, param3, param4)
        # X = np.concatenate(args)
        X = np.array(args)
        return X

    def sampEn_pipeline(self, data):
        # Sample Entropy, Cepstral Coefficients, Root Mean Square and Waveform Length.
        param1 = sample_entropy(data)
        # param2 = Cepstral Coefficients
        param3 = root_mean_square(data)
        param4 = waveform_length(data)
        args = (param1, param3, param4)  # add param2
        X = np.concatenate(args)
        return X

    def _feat_pchannel(self, data):
        # Pipeline module block to get all the feats from a single channel
        # As an example -> still to choose
        p1_emg = waveform_length(data)
        p2_emg = wilson_amplitude(data)
        p3_emg = logvar(data)
        p4_emg = ar(data, order=4, axis=0)
        args = (p1_emg, p2_emg, p3_emg, p4_emg)
        # X = np.concatenate(args)
        X = np.array(args)
        return X


    def write_in(self, rootdir, f_temp, data):
        fname = rootdir + f_temp
        write = dataIO()
        write.write_npy(fname, data)

    def offset_correction(self, x, idx1, idx2):
        temp_mean = np.mean(x, axis=1)
        temp_x = x[:, idx1:idx2] - temp_mean[:, np.newaxis]
        return temp_x

    def norm_rec(self, x):
        # norm = normalizer()
        # norm.fit(x)
        # return norm.transform(x)
        temp_x = (x[:,:] - self.mean[:, np.newaxis])/self.std[:, np.newaxis]
        print(f'temp_x, {temp_x.shape}')
        return temp_x

    def get_specs1(self, data):
        # rewrite: init spectrogram once and then loop through channels
        n_fft = 256  # 1024
        win_length = None
        hop_length = 100
        # Define transform
        spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )
        spec = spectrogram(torch.from_numpy(data))
        return spec

    def get_specs2(self, data):
        pass
