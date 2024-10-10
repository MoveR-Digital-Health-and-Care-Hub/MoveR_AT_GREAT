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


def prep_data_pos_clf(dataset:AT_GreatFolded, config, grasp): 
    x_train =  dataset.fold_data.numpy()   
    y_train = dataset.fold_lbl.numpy().astype(int)
    
    # get only the grasp
    grasp_indices = (y_train[:, 1] == grasp)
    y_train = y_train[grasp_indices]
    x_train = x_train[grasp_indices]
    
    # from that grasp get only the pos lbl and map them
    y_train = y_train[:, 2]  # pos is the 3rd column
    y_train = map_pos(y_train)


    # get x; norm data only for that grasp across given positions
    dim = 0
    temp_data = x_train.reshape(-1, x_train.shape[2])
    mean = np.mean(temp_data, axis=dim) #.reshape(-1,1) # should be for each feature meaning (144 shape of data)
    std = np.std(temp_data, axis=dim) #.reshape(-1,1)
    x_train = (x_train - mean) / (std + 1e-8)
    x_train_normed = x_train

    # get hudgins feats
    if config['hudgins_feats'] ==True:
        # to get only hudgins feats
        x_train_normed = x_train_normed[:, :, :4]
    x_train_normed =  x_train_normed.reshape(x_train_normed.shape[0], -1)
    
    return x_train_normed, y_train



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
            
            train_dataset, val_dataset, _, _ = fold_dataset_load_lda(data_config['source']['data_path'], 
                                                    data_config['source']['sub'], data_config['source']['pos'], 
                                                    data_config, fold, feats=feats, mean=None, std=None,
                                                    compute_norms=compute_norms)
        
        else:
            raise NotImplementedError
        
        temp_test_dict = {}
        temp_train_dict = {}
        for grasp in config['grasps']:
            x_train, y_train = prep_data_pos_clf(train_dataset, config, grasp=grasp)
            x_test,  y_test = prep_data_pos_clf(val_dataset, config, grasp=grasp)

            #-------------- Compute the LDA accuracy---------------
            test_acc, train_acc = lda_acc(x_train, y_train, x_test, y_test)
            temp_test_dict.update({str(grasp)+'_test': test_acc})
            temp_train_dict.update({str(grasp)+'_train': train_acc})

        # Append the metrics to the dataframe
        new_row = pd.DataFrame({'sub': [sub_id],
                    'source': [data_config['source']['pos']],
                    'target': [data_config['target']['pos']],
                    'fold': [fold],
                    **temp_test_dict, 
                    **temp_train_dict})
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
    config = get_configs(os.path.join(os.getcwd(), 'AT_Great', 'experiment_configs', 'exp', 'lda_exp.yaml'))
    datafile_path = os.path.abspath(config["dataset"]["config"])
    data_config = get_configs(datafile_path)
    config["dataset"]["config"] = data_config
    config["data_config"] = data_config


    glob_no = 0
    
    
    
    data_config['task'] = 'source_only'
    s = 's200'
    fnames = ['cross_pos_clf_' + s + '.csv', 'diag_pos_clf_' + s + '.csv']

    for i,(pos, fname) in enumerate(zip(config['poss'], fnames)):
        df = pd.DataFrame(columns=['sub', 
                              'source', 
                              'target', 
                              'fold', 
                              '1_test',
                              '2_test',
                              '3_test',
                              '4_test',
                              '5_test',
                              '6_test',
                              '1_train',
                              '2_train',   
                              '3_train',
                              '4_train',
                              '5_train',
                              '6_train'])
        
        pos = [str(p) for p in pos]
        data_config['source']['pos'] = pos
        data_config['target']['pos'] = pos
        
        df, _ = train_subs(data_config, 
                            config, 
                            'cuda:0',
                            pos,
                            pos,
                            df, 
                            glob_no, 
                            config['seed_fold_gen_path'][0],
                            config['fold_params_dirs'][0],
                            compute_norms=True)
   

        df.to_csv(fname, index=False)


# tabulate.tabulate(table, headers=headers, tablefmt="latex_raw", floatfmt=".1f")