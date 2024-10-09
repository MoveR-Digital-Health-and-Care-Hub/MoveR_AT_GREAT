import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np
from AT_Great.datautil.great_base import *
from sklearn.model_selection import StratifiedKFold
from AT_Great.datautil.util import *
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
import yaml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
from AT_Great.datautil.util import *

class AT_GreatFolded():
    def __init__(self, 
                 config, 
                 data_path, 
                 sub, 
                 pos, 
                 transform=None, 
                 target_transform=None, 
                 fold=0, 
                 datafold_path=None,
                 get_fold_set='train', 
                 norm_dim=0, 
                 mean=None, 
                 std=None, 
                 feats='aug'): # 'aug' or 'ftd_base'
        super().__init__()# config, data_path, sub, pos, transform, target_transform, aug_data)
        self.position = pos
        self.get_fold_set = get_fold_set
        self.config= config
        self.sub = sub
        self.transform = transform
        self.target_transform = target_transform
        self.fold = fold
        self.feats = feats
        self.norm_dim = norm_dim
       
        # load all data
        self.data = self.agg_data(data_path)
        self.lbl = self.__agg_lbl(data_path)
        
        # get fold data 
        self.fold_dict = self.read_fold_idxs(datafold_path) 
        self.fold_data, self.fold_lbl = self.get_fold_data(self.get_fold_set)   
        
        # if 'pos_clf' in config:
        #     self.pos_clf = config['pos_clf']

        # else:
        #     self.pos_clf = False


        # get the mean and std for the fold
        if mean is None or std is None:
            self.data_mean, self.data_std = self.get_norms(dim=norm_dim, feats=feats)
        else:
            self.data_mean = mean if isinstance(mean, torch.Tensor) else torch.tensor(mean)
            self.data_std = std if isinstance(std, torch.Tensor) else torch.tensor(std)
    
        
    def data_check(self):
        unique_trials = torch.unique(self.lbl[:, 0])
        print("Unique Trials:", len(unique_trials))
        assert len(unique_trials) == 60, 'Need 60 trials in each position'
        for cl in range(1,7): # 6 classes
            print('unique trial, grasp:', cl, 'trial count', len(torch.where(self.lbl[:, 1]==cl)[0])) #, return_counts=True))
        '''# check  how many trials per grasp
        # unique_grasps = torch.unique(self.lbl[:, 1], return_counts=True)
        # unique_positions = torch.unique(self.lbl[:, 2])
        # print('Lbl shape', self.lbl.shape, 
        #       '\n No of trials', len(torch.unique(self.lbl[:,0])))
        # print("Unique Positions:", unique_positions[0])
        # print("Unique Grasps:", unique_grasps)'''
        
        
    # --------------------------------------------------
    # ----------------- aggregate data -----------------
    def agg_data(self, data_path):
        tensors = []
        for pos in self.position:
            # data = torch.cat([torch.load(os.path.join(data_path, f'{block}_position_{pos}.pt')) 
            #                           for block in self.config['pos_day_block'][pos]], dim=0)
            data = []
            for block in self.config['pos_day_block'][pos]:
                tensor = torch.load(os.path.join(data_path, f'{block}_position_{pos}.pt'))
                # print('agg data location ', tensor.device)
                data.append(tensor)
            data = torch.cat(data, dim=0) # dim=0)
            tensors.append(data)
        self.data = torch.cat(tensors, dim=0) #torch.stack(tensors, dim=0).squeeze()

        return self.data
    

    def __agg_lbl(self, data_path):
        # self.lbl = self.__update_lbl_trials(data_path)
        self.lbl = update_lbl_trials(data_path, self.config, self.position)
        return self.lbl
    

    # def __update_lbl_trials(self, data_path):
    #     '''
    #     Update the trial values in the lbl tensor. The trial values are updated to be unique across all the blocks.
    #     The counter is then resetn and start from 0 for the next position. 
    #     This is important for the fold generation as the trial values are used as keys to split the data into folds.
    #     The lbl values are still unique, having [trial, label, position] as the key.
    #     '''
        
    #     final_lbl = torch.empty(0, 3) # dim: trial, label, position

    #     # Can extend to multiple subjects if needed
    #     for pos in self.position:
    #         global_index = 0
    #         for block in self.config['pos_day_block'][pos]:
    #             lbl = torch.load(os.path.join(data_path, f'{block}_position_{pos}_y.pt'))
                
    #             temp_arr = lbl.clone() # make a copy of an arr
    #             temp_trials = torch.zeros(temp_arr.shape[0])
    #             unique_values = torch.unique(temp_arr[:, 0]) #, return_inverse=True, return_inverse=True)
    #             no_unique_trials = len(unique_values)
    #             #print('Block:', block, 'Unique trials:', no_unique_trials)

    #             for value in unique_values:
    #                 indices = torch.where(temp_arr[:,0] == value)[0]
    #                 temp_trials[indices] = global_index
    #                 global_index += 1

    #             temp_arr[:,0] = temp_trials
    #             final_lbl = torch.cat([final_lbl, temp_arr], dim=0)
    #             # print('final lbl shape', final_lbl.shape, 'final lbl trials:', len(torch.unique(final_lbl[:, 0])))
    #             # print('final unique:', torch.unique(final_lbl[:, 0], return_counts=True))
    #     return final_lbl
    
    def get_uniqmap_pos_lbl(self, lbl=None):
        if lbl is None:
            lbl= self.lbl
        # get the unique mappings for the position and label
        unique_mappings, indices = np.unique(lbl[:, [1, 2]].to('cpu').numpy(), axis=0, return_index=True)

        temp_arr = torch.zeros(lbl.shape[0])
        glob_val = 0
        for i in range(len(unique_mappings)):
           
            idxs = torch.where(torch.logical_and(lbl[:,1] == unique_mappings[i][0], lbl[:,2] == unique_mappings[i][1]))[0]
            temp_arr[idxs] = glob_val
            glob_val += 1

        # add temp_Arr as lbl[:,3]
        lbl = torch.cat([lbl, temp_arr.reshape(-1,1)], dim=1)

        return lbl




    # --------------------------------------------------
    # ----------------- get fold idxs, data ------------
    def read_fold_idxs(self, datafile_path=None):
        _dict={}
        for pos in self.position:
            pos = int(pos)
            if datafile_path is None:
                datafile_path = os.path.abspath(self.config["fold_config"])
            fold_config = get_configs(datafile_path)
            
            
            train_trial_ids = np.array(fold_config[pos][self.fold]['train_trials'])[:,0]
            test_trial_ids = np.array(fold_config[pos][self.fold]['test_trials'])[:,0] # [trial, grip_lbl]
            _dict.update({pos: [{'train_trial_ids': train_trial_ids, 'test_trial_ids': test_trial_ids}]})
                
        return _dict #train_trial_ids, test_trial_ids, 
    

    def _get_idxs(self, _dict, fold_mode='train_trial_ids', pos=[0]):
        # sample_idxs = flatten([np.where(self.lbl[:,0].to('cpu').numpy() == i)[0].tolist() for i in fold_idxs])
        # sample_idxs = flatten([torch.where(self.lbl[:,0].to('cpu') == i)[0].tolist() for i in fold_idxs])
        combined_idxs = []
        for p in pos:
            p = int(p)
            for i in _dict[p][0][fold_mode]: # why 0?
                condition1 = self.lbl[:,0].to('cpu') == i
                condition2 = self.lbl[:,2].to('cpu') == p
                idxs = torch.where(torch.logical_and(condition1, condition2))[0].tolist()
                combined_idxs.extend(idxs)
        sample_idxs = flatten(combined_idxs)
        return sample_idxs


    def get_fold_data(self, fold_idx_set='train'):
        '''
        Function returns the fold data and labels based on the fold_idx_set
        '''
        if fold_idx_set=='train':
            idxs = self._get_idxs(self.fold_dict, 'train_trial_ids', self.position)
        elif fold_idx_set=='test':
            idxs = self._get_idxs(self.fold_dict, 'test_trial_ids', self.position)
        elif fold_idx_set=='all':
            idxs = self._get_idxs(self.fold_dict, 'train_trial_ids', self.position) + self._get_idxs(self.fold_dict, 'test_trial_ids', self.position)
        else:
            raise ValueError('fold_idx_set should be either train or test')
        
        fold_data = self.data[idxs]
        fold_lbl = self.lbl[idxs]
        return fold_data, fold_lbl

    
    def get_fold_transformed_data(self):
        '''
        Get fold data transformed based on the mean and std of the fold
        '''
        assert isinstance(self.data_mean, torch.Tensor), "mean must be a torch tensor"
        assert isinstance(self.data_std, torch.Tensor), "std must be a torch tensor"

        if self.feats == 'aug':
            fold_data = (self.fold_data - self.data_mean.reshape(-1,1)) / self.data_std.reshape(-1, 1)
        elif self.feats == 'ftd_base':
            fold_data = (self.fold_data - self.data_mean) / self.data_std
        return fold_data
    
    
    # --------------------------------------------------
    # ----------------- get norms params ------------
    def get_norms(self, dim=0, feats = 'aug', fnames=None):
        # norms are calculated on the train split
        # if statement to assert it is only calculated on the train split
        if self.get_fold_set != 'train':
            fold_data, _ = self.get_fold_data('train')
            temp_data = fold_data.detach().clone()
        else: 
            temp_data = self.fold_data.detach().clone()


        if feats == 'aug': # 144 x 256
            dim = 0
            temp_data = temp_data.transpose(1, 2).reshape(-1, self.data.shape[1])
            
        elif feats == 'ftd_base': # 16 x 9
            dim = 0
            temp_data = temp_data.reshape(-1, self.data.shape[2])
            

        data_mean = torch.mean(temp_data, dim=dim) #.reshape(-1,1) # should be for each feature meaning (144 shape of data)
        data_std = torch.std(temp_data, dim=dim) #.reshape(-1,1)

        if fnames is not None:
            pass
            # save the params for that fold
        return data_mean, data_std 


    # --------------------------------------------------
    # ----------------- pytorch override ---------------
    def __len__(self):
        return len(self.fold_data)

    def __getitem__(self, index): 
        sample = self.fold_data[index]
        label = self.fold_lbl[index]

        assert isinstance(self.data_mean, torch.Tensor), "mean must be a torch tensor"
        assert isinstance(self.data_std, torch.Tensor), "std must be a torch tensor"
       
        if self.transform:
            if self.feats == 'aug':
                sample = (sample - self.data_mean.reshape(-1,1)) / self.data_std.reshape(-1, 1)
                sample = sample.unsqueeze(2)
                print('sample shape', sample.shape)

            elif self.feats == 'ftd_base':
                sample = (sample - self.data_mean) / self.data_std
                # print('sample shape', sample.shape)
                # sample = sample[:,:4]

        if self.target_transform:
            # label = torch.tensor(label[1]) # self.target_transform(label)
            label = label[1].long()
            label -= 1
        
        return sample, label # TODO possibly split into sample, label, position


    # --------------------------------------------------
    # --------------------- aux fns --------------------
    def get_lbl_data(self):
        self.trial = self.lbl[:,0]   
        self.lbl = self.lbl[:,1]
        self.pos = self.lbl[:,2]
        return self.trial, self.lbl, self.pos


    def get_fold_idxs(self, fold):
        train_trial_ids = self.fold_data[fold]['train_trial_ids']
        test_trial_ids = self.fold_data[fold]['test_trial_ids']
        train_idx = flatten([np.where(self.lbl[:,0].to('cpu').numpy() == i)[0].tolist() for i in train_trial_ids])
        test_idx = flatten([np.where(self.lbl[:,0].to('cpu').numpy() == i)[0].tolist() for i in test_trial_ids])
        return train_idx, test_idx




# ------------------------------------------------------------------------------
# ----------------- Save fold generator parameters -----------------------------
# ------------------------------------------------------------------------------
def save_fold_gen_params(data, filename, save_path):
    # Convert tensors to lists
    for key, value in data.items():
        if isinstance(value, dict):
            save_fold_gen_params(value, filename, save_path)
        elif isinstance(value, torch.Tensor):
            data[key] = value.tolist()

    # Save data to YAML file
    with open(os.path.join(save_path, filename), 'w') as file:
        yaml.dump(data, file)


def run_great_getmean_std_singlepos(current_dir, args, data_folder, feats):
    '''
    ------------ Single-position combinations ------------ 
    '''
    seeds = ['seed100', 'seed200', 'seed0']
    for s in seeds:
 
        for i in range (len(data_folder)):
            # save path
            save_path = os.path.join(current_dir, 'AT_Great', 'experiment_configs', 'up_'+data_folder[i],  'fold_params_' + s)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # data_path
            data_path = os.path.join(current_dir, 'AT_Great', 'data', data_folder[i])

            # --------------------------------------------------------------------------
            # Based on the fold generator, save the mean and std for each fold setting, 
            # single positions, but we need to extend to n-1 positions
            for sub in range(1,9): # 8 subjects

                # # # TODO: remove this line
                # # Redo for subject 7
                # if sub != 7:
                #     continue

                print('----------------------------------------------------------------\n', 'sub:', sub)
                _data_path = os.path.join(data_path, f'participant_{str(sub)}')
                fold_gen_path = os.path.join('AT_Great','experiment_configs', 'up_'+ data_folder[i], 'fold_generator_' + s ,  f'participant_{str(sub)}.yaml') # also inside exp config
                position_dict = {}

                for pos in range(1, 11): # 9 positions
                    temp_dict={}

                    for fold in range(5): 
                        dataset = AT_GreatFolded(args, _data_path, f'participant_{str(sub)}', [str(pos)], # [str(pos)], 
                                transform=True, target_transform=None, fold=fold,
                                datafold_path=fold_gen_path, feats=feats[i], get_fold_set='train', # feats='ftd_base'
                                mean=None, std=None)
                        mu, std = dataset.data_mean, dataset.data_std 

                        
                        # testing data
                        temp = dataset.fold_data 
                        norm_temp = (temp - mu) / std
                        min_vals, _ = torch.min(norm_temp, dim=0)
                        max_vals, _ = torch.max(norm_temp, dim=0)
                        print(f"Min values: {min_vals}")
                        print(f"Max values: {max_vals}")
                        import matplotlib.pyplot as plt
                        fig, axs = plt.subplots(1, 2, figsize=(15, 6))

                         # Plot original data distribution
                        # plt.hist(temp[:, 0, 0].numpy(), bins=50, alpha=0.5, label='Original Data')
                        axs[0].hist(temp[:, 0].numpy(), bins=50, alpha=0.7, color='blue')
                        plt.xlabel('Value')
                        axs[0].set_title('Original Data Distribution')
                        plt.title('Distribution of Original and Normalized Data')
                        axs[0].set_xlabel('Value')
                        # plt.savefig(f'./data_distribution_{sub}_{pos}_{fold}.png')
                        axs[0].set_ylabel('Frequency')
                        # plt.show()


                        # Plot normalized data distribution
                        # temp_dict.update({fold :fold_gen})
                        axs[1].hist(norm_temp[:, 0, 0].numpy(), bins=50, alpha=0.7, color='green')
                         # print('position_dict:', position_dict)
                        axs[1].set_title('Normalized Data Distribution')
                        plt.savefig(f'./data_distribution_{sub}_{pos}_{fold}.png')  
                        plt.close()
                        plt.close()         
def run_great_getmean_std_multipos(current_dir, args, data_folder, feats, positions):
    '''
    ------------ Multi-position combinations ------------ 
    '''
    for i in range (2): # two sets of data: ftd_base, aug
        # ------------ Multi-position combinations ------------ 
        for k in range(len(positions)):
            temp_poss = [positions[j] for j in range(len(positions)) if j != k]
            pos_mergefname = '_'.join(temp_poss)
            print('Train Positions', pos_mergefname)
            # -------- For table separation print out ----------
            # print("$P_{", pos_mergefname, "}$ $\ to$ $P_{", positions[k],"}")


            # save path
            save_path = os.path.join(current_dir, 'AT_Great', 'experiment_configs', data_folder[i],  'fold_params_multi_pos', f'trainpos_{pos_mergefname}')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # data_path
            data_path = os.path.join(current_dir, 'AT_Great', 'data', data_folder[i])

            # --------------------------------------------------------------------------
            # Based on the fold generator, save the mean and std for each fold setting, 
            # single positions, but we need to extend to n-1 positions
            for sub in range(1,9): # 8 subjects
                print('----------------------------------------------------------------\n', 'sub:', sub)
                _data_path = os.path.join(data_path, f'participant_{str(sub)}')
                fold_gen_path = os.path.join('AT_Great','experiment_configs', data_folder[i], 'fold_generator',  f'participant_{str(sub)}.yaml') # also inside exp config
                position_dict = {}

                for pos in range(1, 11): #  positions
                    temp_dict={}

                    for fold in range(5): # 5 folds
                        dataset = AT_GreatFolded(args, _data_path, f'participant_{str(sub)}', temp_poss , # [str(pos)], 
                                transform=True, target_transform=None, fold=fold,
                                datafold_path=fold_gen_path, feats=feats[i], get_fold_set='train', # feats='ftd_base'
                                mean=None, std=None)
                        mu, std = dataset.data_mean, dataset.data_std 

                        fold_gen = {'train_mean': mu, 'train_std': std}
                        temp_dict.update({fold :fold_gen})
                    position_dict.update({pos : temp_dict})
                # print('position_dict:', position_dict)
                save_fold_gen_params(position_dict, f'participant_{str(sub)}.yaml', save_path) 



if __name__ == '__main__':
    # Load the config file
    current_dir = os.getcwd()
    config = get_configs(os.path.join(os.getcwd(), 'AT_Great', 'experiment_configs', 'exp', 'great_exp1.yaml'))
    datafile_path = os.path.abspath(config["dataset"]["config"])
    args = get_configs(datafile_path)

    data_folder = ['processed_data_16x9feats']#, 'processed_data_144x256feats_augstride10and10']
    feats = ['ftd_base']#, 'aug']
    
    

    # ------------ Single-position combinations ------------
    run_great_getmean_std_singlepos(current_dir, args, data_folder, feats)

    # ------------ Multi-position combinations ------------
    # diag_pos = args['position_scenarios']['diag']['pos'] # OR 
    # cross_pos = args['position_scenarios']['cross']['pos']
    # positions_arr = [cross_pos, diag_pos]
    # for positions in positions_arr:
    #     print('Positions:', positions)
    #     start_time = time.time()
        
    #     run_great_getmean_std_multipos(current_dir, args, data_folder, feats, positions)
        
    #     end_time = time.time()
    #     run_time = end_time - start_time
    #     print("Total run time:", run_time, "seconds")


'''

    # save path
    save_path = os.path.join(current_dir, 'AT_Great', 'experiment_configs', data_folder,  'fold_params')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
   
    # data_path
    data_path = os.path.join(current_dir, 'AT_Great', 'data', data_folder)
    # r'/home/kasia/AT_Great/AT_Great/processed_aug_data/hudgins256_100/'
    

    # --------------------------------------------------------------------------
    # Based on the fold generator, save the mean and std for each fold setting, 
    # single positions, but we need to extend to n-1 positions
    for sub in range(1,9): # 8 subjects
        print('----------------------------------------------------------------\n', 'sub:', sub)
        _data_path = os.path.join(data_path, f'participant_{str(sub)}')
        fold_gen_path = os.path.join('AT_Great','experiment_configs', data_folder, 'fold_generator',  f'participant_{str(sub)}.yaml') # also inside exp config
        position_dict = {}

        for pos in range(1, 10): #9 positions
            temp_dict={}
            
            for fold in range(5): 
                dataset = AT_GreatFolded(args, _data_path, f'participant_{str(sub)}', [str(pos)], # [str(pos)], 
                        transform=True, target_transform=None, fold=fold,
                        datafold_path=fold_gen_path, feats=feats, get_fold_set='train', # feats='ftd_base'
                        mean=None, std=None)
                mu, std = dataset.data_mean, dataset.data_std 
                

                fold_gen = {'train_mean': mu, 'train_std': std}
                temp_dict.update({fold :fold_gen})
            position_dict.update({pos : temp_dict})
        # print('position_dict:', position_dict)
        save_fold_gen_params(position_dict, f'participant_{str(sub)}.yaml', save_path) 
'''

    # #  ---------------- example run -----------------
    # sub = str(1)
    # pos = 1
    # fold = 0
    # _data_path = os.path.join(data_path, f'participant_{str(sub)}')
    # fold_gen_path = os.path.join('AT_Great','experiment_configs','fold_generator_9feats',  f'participant_{str(sub)}.yaml') # also inside exp config
     
    # dataset = AT_GreatFolded(args, _data_path, f'participant_{str(sub)}', [str(1), str(3)], # [str(pos)], 
    #                     transform=True, target_transform=True, fold=fold,
    #                     datafold_path=fold_gen_path, feats='aug', get_fold_set='train', 
    #                     mean=None, std=None)
                
    # print(dataset.__getitem__(0), 'datatset length' , dataset.__len__())
    # x, y = dataset.get_fold_transformed_data(), dataset.fold_lbl
    # mu, std = dataset.data_mean, dataset.data_std 
    # print('position data', dataset.lbl[:,2].unique(return_counts=True))
    # # x_train = dataset.
    # # simple_LDA(x_train, y_train, x_test, y_test, pos1, pos2, fold)



'''
 # dataset = AT_GreatFolded(args, _data_path, f'participant_{str(sub)}', [str(pos)], 
    #                     transform=True, target_transform=None, aug_data=True, fold=temp_fold, datafold_path=fold_gen_path)

    # # ----------------------------------------------------------------
    # s1_path =  r'/home/kasia/AT_Great/AT_Great/processed_aug_data/hudgins256_100/participant_5' 
    # # r'/home/kasia/AT_Great/AT_Great/data/timefreq_processed_data_SLIDE10/participant_5' #os.path.join(os.getcwd(), 'great', 'data', 'great_aug', args['source']['sub'])
    # dataset = AT_GreatFolded(args, s1_path, args['source']['sub'], args['source']['pos'], 
    #                    transform=True, target_transform=None, aug_data=True, fold=0, 
    #                    datafold_path=None, get_fold_set='train', norm_dim=0, mean=None, std=None)
    
    # print('Fold, ', dataset.fold, 'train idx', dataset.train_trial_ids, 'test idxs' , dataset.test_trial_ids)
    # print(dataset.get_norms(dim=0))
    # print(dataset.__getitem__(0))

    #----------------------------------------------------------------


    #----------------------------------------------------------------
  

    # print('ll', flatten([torch.where(dataset.lbl[:,0].to('cpu') == i)[0].tolist() for i in dataset.train_trial_ids]))
        #   flatten([torch.where(dataset.lbl[:,0].to('cpu') == 3)[0]).tolist() for i in fold_idxs])
    # print('train', train)
    # print('test', test)
    # ------------------------------------------------------
            
    # ------------------------------------------------------
    # train_idx = flatten([np.where(self.lbl[:,0].to('cpu').numpy() == i)[0].tolist() for i in train_trial_ids])
    # test_idx = flatten([np.where(self.lbl[:,0].to('cpu').numpy() == i)[0].tolist() for i in test_trial_ids])

    # def get_trial_grasp_keys(self):
    #     self.unique_mappings, indices = np.unique(self.lbl[:, [0, 1]].to('cpu').numpy(), axis=0, return_index=True)
    #     return self.unique_mappings
    

    # def trials_split(self):
    #     skf = StratifiedKFold(n_splits=self.k_folds, random_state=self.seed, shuffle=True) # False
    #     b = skf.split(self.unique_mappings[:,0], self.unique_mappings[:,1])
    #     fold_dict = {}
    #     for fold, (train_trial_ids, test_trial_ids) in enumerate(b):
    #         print(f'FOLD {fold}')
    #         print('--------------------------------')
    #         fold_dict[fold] = {'train_trial_ids': train_trial_ids.tolist(), 'test_trial_ids': test_trial_ids.tolist()}
    #     print('fold trials', fold_dict)
    #     return fold_dict


### ----------------------------------------------------

# class AT_Great(Base_Great):
#     def __init__(self, config, data_path, sub, pos, transform=None, target_transform=None, aug_data=False):
#         super().__init__(config, data_path, sub, pos, transform, target_transform, aug_data)
#         self.position = pos
#         self.trial = None
#         self.seed=42
#         # load data
#         self.lbl = self.__agg_lbl(data_path)
        
#         # build folds
#         self.get_trial_grasp_keys()
#         if 'fold' in config:
#             self.fold = config['fold']
#         else:
#             self.fold = 0  # default fold

#         self.k_folds = 5
#         self.fold_data = self.trials_split()

#         # get params to normalize data
#         self.get_norms()

#     def get_trial_grasp_keys(self):
#         self.unique_mappings, indices = np.unique(self.lbl[:, [0, 1]].to('cpu').numpy(), axis=0, return_index=True)
#         return self.unique_mappings
    

#     def trials_split(self):
#         skf = StratifiedKFold(n_splits=self.k_folds, random_state=self.seed, shuffle=True) # False
#         b = skf.split(self.unique_mappings[:,0], self.unique_mappings[:,1])
#         fold_dict = {}
#         for fold, (train_trial_ids, test_trial_ids) in enumerate(b):
#             print(f'FOLD {fold}')
#             print('--------------------------------')
#             fold_dict[fold] = {'train_trial_ids': train_trial_ids.tolist(), 'test_trial_ids': test_trial_ids.tolist()}
#         print('fold trials', fold_dict)
#         return fold_dict


#     def get_norms(self):
#         # make a copy of the data?
#         temp_data = self.data.detach().clone()
    
#         temp_data = temp_data.transpose(1, 2).reshape(-1, self.data.shape[1])
#         # Normalize data
#         # temp_data = (temp_data - torch.mean(temp_data, dim=0)) / torch.std(temp_data, dim=0)
#         self.data_mean = torch.mean(temp_data, dim=0) # should be for each feature meaning (144 shape of data)
#         self.data_std = torch.std(temp_data, dim=0)
#         print('mean', self.data_mean.shape, 'std', self.data_std.shape)
#         # # Normalize data
#         # scaler = StandardScaler()
#         # all_data_normalized = scaler.fit_transform(self.data.numpy())

#         # # Convert back to PyTorch tensor
#         # all_data_normalized = torch.tensor(all_data_normalized, dtype=torch.float32)
#         return self.data_mean.reshape(-1,1), self.data_std.reshape(-1,1) # reshape to (144, 1) shape of data


#     def __getitem__(self, index):
#         sample = self.data[index]
#         label = self.lbl[index]

#         #TODO: Include self.pos for the position of the sample. 
#         # Depends on the objective and the model.

#         #TODO: Implement logic to transform data
#         if self.transform:
#             # Create transform with normalization
#             # self.transform = transforms.Compose([
#             # transforms.ToTensor(),
#             # transforms.Normalize(mean=self.data_mean, std=self.data_std)
#             # ])
        
#             # sample = self.transform(sample)
            
#             sample = (sample - self.data_mean) / self.data_std

#             print('sample was tansformedddd')
#             # print('self.data_mean) / self.data_std:', self.data_mean,  self.data_std)

#         if self.target_transform:
#             label = self.target_transform(label)
#         return sample, label # TODO possibly split into sample, label, position

#     def get_lbl(self):
#         # TODO implement the logic to return the label and position
#         return self.trial, self.lbl, self.pos
        

#     def get_lbl_data(self):
#         self.trial = self.lbl[:,0]   
#         self.lbl = self.lbl[:,1]
#         self.pos = self.lbl[:,2]
#         return self.trial, self.lbl, self.pos


#     def __agg_lbl(self, data_path):
#         self.lbl = self.__update_lbl_trials(data_path)
#         return self.lbl


#     def get_fold_idxs(self, fold):
#         train_trial_ids = self.fold_data[fold]['train_trial_ids']
#         test_trial_ids = self.fold_data[fold]['test_trial_ids']
#         train_idx = flatten([np.where(self.lbl[:,0].to('cpu').numpy() == i)[0].tolist() for i in train_trial_ids])
#         test_idx = flatten([np.where(self.lbl[:,0].to('cpu').numpy() == i)[0].tolist() for i in test_trial_ids])
#         return train_idx, test_idx


#     def __update_lbl_trials(self, data_path):
#         global_index = 0
#         final_lbl = torch.empty(0, 3) # dim: trial, label, position
#         for pos in self.position:
#             for block in self.config['pos_day_block'][pos]:
#                 lbl = torch.tensor(torch.load(os.path.join(data_path, f'{block}_position_{pos}_y.pt')))
                
#                 for trial in torch.unique(lbl[:, 0]):
#                     temp_idx = list(torch.where(trial.item() == lbl[:,0]))[0].tolist()
#                     lbl[temp_idx, 0] = global_index
#                     global_index += 1 # update the key
                
#                 # update the trial values in the lbl tensor
#                 final_lbl = torch.cat([final_lbl, lbl], dim=0)
            
#             print( 'unique trial', torch.unique(final_lbl[:,0], return_counts=True))
#         return final_lbl
    

#     def get_fold(self, ):
#         pass



'''