import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import wandb
import random
from AT_Great.trainers import *
import os
from AT_Great.experiment_configs.utils import get_configs
import torch.nn as nn
from AT_Great.models.TCN import TCN
from AT_Great.datautil.great_dataloader import *
from AT_Great.trainers import *
from AT_Great.models.MLP import MLP
from AT_Great.models.CNN import CNN1D, CNN
import argparse
from AT_Great.trainers import *
import pandas as pd


from sklearn.metrics import accuracy_score
from AT_Great.models.LDA import lda_acc

from AT_Great.models.hmc import h_clf_acc

# that will be helpful for 
# from AT_Great.datautil.util import map_pos
#     y_train = y_train[:,2] #  pos is the 3rd column
#     y_train = map_pos(y_train)

def prep_data(dataset:AT_GreatFolded, config):
    x_train_normed = dataset.get_fold_transformed_data().numpy()
    # x_train, y_train = dataset.fold_data, dataset.fold_lbl
    if config['hudgins_feats'] ==True:
        # to get only hudgins feats
        x_train_normed = x_train_normed[:, :, :4]
    x_train_normed =  x_train_normed.reshape(x_train_normed.shape[0], -1)
    
    y_train = dataset.fold_lbl.numpy().astype(int)
    # y_train = y_train[:,1] #  pos is the 3rd column
    
    return x_train_normed, y_train[:,1], y_train[:,2] 



def train_exp(sub_id, 
              data_config,
              config, 
              device, 
              df, 
              glob_no, 
              folds=5, 
              compute_norms=False, 
              feats='ftd_base'):
    
    data_config['lda'] = True

    for fold in range(folds): 
        print('------------------------------------- fold:', fold, '------------------------------------- ')
        # ----------- get data -----------   
        data_config['fold'] = fold
        if data_config['task'] == 'source_only':
            
            train_dataset, val_dataset, mean, std= fold_dataset_load_lda(data_config['source']['data_path'], 
                                                    data_config['source']['sub'], data_config['source']['pos'], 
                                                    data_config, fold, feats=feats, mean=None, std=None,
                                                    compute_norms=compute_norms)
        
        else:
            raise NotImplementedError
            
        x_train, y_train, z_train = prep_data(train_dataset, config)
        x_test,  y_test, z_test = prep_data(val_dataset, config)

        #-------------- Compute the LDA accuracy---------------
        # acc = lda_acc(train_dataset, val_dataset, mean, std)
        predict_lvl2, z_acc = h_clf_acc(x_test, 
                                        y_test, 
                                        z_test, 
                                        x_train, 
                                        y_train, 
                                        z_train,
                                        config)


        print('fold:', fold,  'predict_lvl2:', predict_lvl2, 'z_acc:', z_acc)
        scores_dict = {f'score{i+1}': predict_lvl2[i] for i in range(len(predict_lvl2))}
        # Append the metrics to the dataframe
        new_row = pd.DataFrame({'sub': [sub_id],
                'source': [data_config['source']['pos']],
                'target': [data_config['target']['pos']],
                'fold': [fold],
                'z_acc': [z_acc],
                **scores_dict})
        df = pd.concat([df, new_row], ignore_index=True)
        glob_no += 1
    return df, glob_no


def train_subs(data_config, config, device, train_pos, test_pos, df, glob_no, seed_fold, fold_params, compute_norms=False):
    
    # ----------- get subject -----------
    for sub_id in range(1, 9):
        print('\n-------------------------------------')
        print('sub_id:', sub_id, 'position:', train_pos, 'test pos', test_pos)
        s_temp_path = r'/home/kasia/AT_Great/AT_Great/processed_aug_data/hudgins256_100/participant_'+str(sub_id)
        data_config['fold_config'] =  '/home/kasia/AT_Great/AT_Great/experiment_configs/up_processed_data_16x9feats/'+ seed_fold+'/participant_'+str(sub_id)+ '.yaml'
        data_config['fold_params'] =  '/home/kasia/AT_Great/AT_Great/experiment_configs/up_processed_data_16x9feats/'+fold_params+'/participant_'+str(sub_id)+ '.yaml'
        
        data_config['source']['data_path'] = r'/home/kasia/AT_Great/AT_Great/data/processed_data_16x9feats/participant_'+str(sub_id)
        data_config['target']['data_path'] =  r'/home/kasia/AT_Great/AT_Great/data/processed_data_16x9feats/participant_'+str(sub_id)
        data_config['source']['sub'] = [sub_id]
        data_config['target']['sub'] = [sub_id]
        config['use_wandb'] = False
        df, glob_no = train_exp(sub_id, data_config, config, device, df, glob_no, feats=data_config['feats'], compute_norms=compute_norms)
    return df, glob_no



if __name__ == "__main__":

    # ----------- get configs -----------
    config = get_configs(os.path.join(os.getcwd(), 'AT_Great', 'experiment_configs', 'exp', 'hmc.yaml'))
    datafile_path = os.path.abspath(config["dataset"]["config"])
    data_config = get_configs(datafile_path)
    config["dataset"]["config"] = data_config
    config["data_config"] = data_config


    glob_no = 0
    
    df = pd.DataFrame(columns=['sub', 
                              'source', 
                              'target', 
                              'fold', 
                              'z_acc', 
                              'score1', 
                              'score2', 
                              'score3', 
                              'score4', 
                              'score5', 
                              'score6',
                              'score7',
                              'score8',
                              'score9'])
    
    data_config['task'] = 'source_only'
    data_config['source']['pos'] = config['poss']
    data_config['target']['pos'] = config['poss']
    df, _ = train_subs(data_config, 
                        config, 
                        'cuda:0',
                        config['poss'],
                        config['poss'],
                        df, 
                        glob_no, 
                        config['seed_fold_gen_path'],
                        config['fold_params_dirs'],
                        compute_norms=True)
    s = 'seed200'
    fname = 'hmc_naive_' + s + '.csv'
    # Save the dataframe to a CSV file
    df.to_csv(fname, index=False)