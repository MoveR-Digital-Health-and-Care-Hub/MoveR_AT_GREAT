from sklearn.model_selection import StratifiedKFold
import torch
import numpy as np
import os
import yaml
from AT_Great.datautil.util import *
from AT_Great.experiment_configs.utils import get_configs


class FoldGenerator():
    def __init__(self, config, data_path, k_folds, pos, seed=42, diag=False):
        self.config = config
        self.data_path = data_path
        self.seed = seed
        self.k_folds = k_folds
        self.position = pos 
        self.diag=diag
        
        # Load the lbl tensor wiht updated keys, same as AT_GreatFolded Dataset
        self.lbl = update_lbl_trials(data_path, self.config, self.position)
        self.unique_mappings = self.get_trial_grasp_keys()

        # fold generator
        self.fold_generator = self.trials_split()


    def get_trial_grasp_keys(self):
        '''
        Since fold generator is applied to a specific data_path with single position, 
        this function creates unique mapping of trial, label.
        '''
        unique_mappings, indices = np.unique(self.lbl[:, [0, 1]].to('cpu').numpy(), axis=0, return_index=True)
        return unique_mappings 
    

    def trials_split(self):
        '''
        Using unique mapping from get_trial_grasp_keys(), we split the trials into k-folds.
        This split on trials and labels prevents the same trial from being in both train and test set, 
        creating a data leak.
        
        return: fold_dict: dictionary with keys as fold number and 
        values as train and test trial idxs in unique mapping.
        To get the actual trials idxs, use get_fold_trials() function.
        '''
        skf = StratifiedKFold(n_splits=self.k_folds, random_state=self.seed, shuffle=True) # False
        b = skf.split(self.unique_mappings[:,0], self.unique_mappings[:,1])
        fold_dict = {}
        for fold, (train_ids, test_ids) in enumerate(b):
            print(f'FOLD {fold}')
            print('--------------------------------')
            fold_dict[fold] = {'train_ids': train_ids.tolist(), 'test_ids': test_ids.tolist()}
        print('fold trials', fold_dict)
        return fold_dict
    

    def get_fold_trials(self, fold_dict, fold):
        '''
        Read the trials based on idxs from unique_mappings.
        '''
        temp_idx = np.array(fold_dict[fold]['train_ids'])
        temp_test_idx = np.array(fold_dict[fold]['test_ids'])
        fold_trials = self.unique_mappings[temp_idx]
        fold_test_trials = self.unique_mappings[temp_test_idx]
        return fold_trials, fold_test_trials


    def get_all_fold_trials(self):
        '''
        Get all the fold trials for the given position for train and test split i each position and fold.
        
        '''
        all_fold_trials = {}
        for fold in range(self.k_folds):
            fold_trials, fold_test_trials = self.get_fold_trials(self.fold_generator, fold)
            all_fold_trials[fold] = {'train_trials': fold_trials.tolist(), 'test_trials': fold_test_trials.tolist()}

            fold_trials = np.array(fold_trials)
            fold_test_trials = np.array(fold_test_trials)
            # print('Unique values in fold_test_trials for idx 1:\n', np.unique(fold_test_trials[:, 1], return_counts=True))
            # print('Unique values in fold_train_trials for idx 1:\n', np.unique(fold_trials[:, 1], return_counts=True))
        return all_fold_trials
    

    # def __update_lbl_trials(self, data_path):
    #     '''
    #     Update the trial values in the lbl tensor. The trial values are updated to be unique across all the blocks 
    #     (2 blocks in each position have hte same trial keys - w.r.t. data recording protocol).
    #     The counter is then reset and start from 0 for the next position. 
    #     This is important for the fold generation as the trial values are used as keys to split the data into folds.
    #     The lbl values are still unique, having [trial, label, position] as the key.
    #     '''
    #     global_index = 0
    #     final_lbl = torch.empty(0, 3) # dim: trial, label, position

    #     # Can extend to multiple subjects if needed
    #     for pos in self.position:
    #         for block in self.config['pos_day_block'][pos]:
    #             lbl = torch.load(os.path.join(data_path, f'{block}_position_{pos}_y.pt'))
                
    #             temp_arr = lbl.clone() 
    #             temp_trials = torch.zeros(temp_arr.shape[0])
    #             unique_values = torch.unique(temp_arr[:, 0]) 
    #             no_unique_trials = len(unique_values)
    #             # print('Block:', block, 'Unique trials:', no_unique_trials)

    #             for value in unique_values:
    #                 indices = torch.where(temp_arr[:,0] == value)[0]
    #                 temp_trials[indices] = global_index
    #                 global_index += 1

    #             temp_arr[:,0] = temp_trials
    #             final_lbl = torch.cat([final_lbl, temp_arr], dim=0)
    #             # print('final lbl shape', final_lbl.shape, 'final lbl trials:', len(torch.unique(final_lbl[:, 0])))
    #             # print('final unique:', torch.unique(final_lbl[:, 0], return_counts=True))
    #     return final_lbl


# ---------------------- Save fold_generator dict to a file -------------------------
def save_fold_generator(dictt, fname, fold_dir):
    with open(os.path.join(fold_dir, fname), 'w') as file:
        yaml.dump(dictt, file)
   

def generate_folds(args, data_path, save_path, seed=0):
    '''
    Run to generate folds.
    '''
   # ---------------- Fold generator for 5-fold, each position, per subject ---------------------------	
    for sub in range(1,9): # 8 subjects

        print('----------------------------------------------------------------\n', 'sub:', sub)
        _data_path = os.path.join(data_path, f'participant_{str(sub)}')
        
        position_dict = {}
        for pos in range(1, 11): # 10 positions
            args['fold_seed'] = seed
            fold_gen = FoldGenerator(args, _data_path, args["k_folds"], [str(pos)], args['fold_seed'])
            temp_folds = fold_gen.get_all_fold_trials()
            position_dict.update({pos : temp_folds})

        # print('position_dict:', position_dict)
        
        # TODO uncomment the line below to save the fold_generator dict to a file
        save_fold_generator(position_dict, f'participant_{str(sub)}.yaml', save_path)




if __name__ == "__main__":

    # Load the config file
    current_dir = os.getcwd()
    config = get_configs(os.path.join(os.getcwd(), 'AT_Great', 'experiment_configs', 'exp', 'great_exp1.yaml'))
    datafile_path = os.path.abspath(config["dataset"]["config"])
    args = get_configs(datafile_path)

    ########################################
    # Choose the data to generate folds for
    ########################################
    temp = ['processed_data_16x9feats']#, 'processed_data_144x256feats_augstride10and10'] 
    seeds = [100, 200, 0]
    for s in seeds:

        for t in temp:
            save_path = os.path.join(current_dir, 'AT_Great', 'experiment_configs', 'up_'+t , 'fold_generator'+'_seed'+str(s))
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # data dir to read te data from
            data_path = '/home/kasia/AT_Great/AT_Great/data/' + t
            # generate folds
            generate_folds(args, data_path, save_path, seed=s) # 0, 100, 200 

        


        # print('----------------------------------------------------------------\n', 'sub:', sub)
        
        # pos=10
        # _data_path = os.path.join(data_path, f'participant_{str(sub)}')
        # # Load the existing dictionary from participant_{str(sub)}_pos11.yaml
        # existing_dict = {}
        # with open(os.path.join(save_path, f'participant_{str(sub)}.yaml'), 'r') as file:
        #     existing_dict = yaml.load(file, Loader=yaml.FullLoader)

        # # Update the existing dictionary with the new key=pos and its corresponding value from position_dict
        # fold_gen = FoldGenerator(args, _data_path, args["k_folds"], [str(pos)], args['fold_seed'])
        # temp_folds = fold_gen.get_all_fold_trials()
        # existing_dict.update({pos: temp_folds})

        # # Save the updated dictionary back to participant_{str(sub)}_pos11.yaml
        # save_fold_generator(existing_dict, f'participant_{str(sub)}.yaml', save_path)

    '''
    def __update_lbl_trials(self, data_path):
        global_index = 0
        final_lbl = torch.empty(0, 3) # dim: trial, label, position
        for pos in self.position:
            for block in self.config['pos_day_block'][pos]:
                if self.diag==True and pos==10:
                    diag_center = 5 # we overwrite this pos from 5 to 10 so its easier to 
                    lbl = torch.tensor(torch.load(os.path.join(data_path, f'{block}_position_{diag_center}_y_diag.pt')))
                else:
                    lbl = torch.tensor(torch.load(os.path.join(data_path, f'{block}_position_{pos}_y.pt')))
                print('lblb device:', lbl.device)

                for trial in torch.unique(lbl[:, 0]):
                    # need to update the trial values in the lbl tensor to not repeat the trial keys for 2 blocks
                    # otherwise if you run unique fn() it will return hald of the values
                    temp_idx = list(torch.where(trial.item() == lbl[:,0]))[0].tolist()
                    lbl[temp_idx, 0] = global_index
                    global_index += 1 # update the key
                    
                
                # update the trial values in the lbl tensor
                final_lbl = final_lbl.to(lbl.device)  # Move final_lbl to the same device as lbl
                final_lbl = torch.cat([final_lbl, lbl], dim=0)
            
            print( 'unique trial', torch.unique(final_lbl[:,0], return_counts=True))
        return final_lbl
        '''
    '''def __update_lbl_trials_old(self, data_path):
       
        Update the trial values in the lbl tensor. The trial values are updated to be unique across all the blocks.
        The counter is then resetn and start from 0 for the next position. 
        This is important for the fold generation as the trial values are used as keys to split the data into folds.
        The lbl values are still unique, having [trial, label, position] as the key.
        
        final_lbl = torch.empty(0, 3) # dim: trial, label, position
        for pos in self.position:
            global_index = 0
            b=0
            for block in self.config['pos_day_block'][pos]:
                lbl = torch.load(os.path.join(data_path, f'{block}_position_{pos}_y.pt'))
                
                temp_arr = lbl.clone() # make a copy of an arr
                unique_values, idxs = torch.unique(temp_arr[:, 0], return_inverse=True)
                no_unique_trials = len(unique_values)
                # print('Block:', block, 'Unique trials:', no_unique_trials, 'get inverse:', idxs)

                if b == 1:
                    # ('Block', block)
                    global_index += no_unique_trials  # 30 
                    idxs += global_index
                    # print('idxs:', idxs)

                b+=1
                temp_arr[:, 0] = idxs
                final_lbl = torch.cat([final_lbl, temp_arr], dim=0)
                # print('final lbl shape', final_lbl.shape, 'final lbl trials:', len(torch.unique(final_lbl[:, 0])))
                # print('final unique:', torch.unique(final_lbl[:, 0], return_counts=True))
        return final_lbl
'''