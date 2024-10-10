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
    x_train_normed = dataset.get_fold_transformed_data().numpy()
    # x_train, y_train = dataset.fold_data, dataset.fold_lbl
    if config['hudgins_feats'] ==True:
        # to get only hudgins feats
        x_train_normed = x_train_normed[:, :, :4]
    x_train_normed =  x_train_normed.reshape(x_train_normed.shape[0], -1)
    
    y_train = dataset.fold_lbl.numpy().astype(int)
    
    # get only the grasp
    grasp_indices = (y_train[:, 1] == grasp)
    y_train = y_train[grasp_indices]
    x_train_normed = x_train_normed[grasp_indices]
    
    # from that grasp get only the pos lbl and map them
    y_train = y_train[:, 2]  # pos is the 3rd column
    y_train = map_pos(y_train)
    
    return x_train_normed, y_train



def prep_data(dataset:AT_GreatFolded, config):
    x_train_normed = dataset.get_fold_transformed_data().numpy()
    # x_train, y_train = dataset.fold_data, dataset.fold_lbl
    if config['hudgins_feats'] ==True:
        # to get only hudgins feats
        x_train_normed = x_train_normed[:, :, :4]
    x_train_normed =  x_train_normed.reshape(x_train_normed.shape[0], -1)
    
    y_train = dataset.fold_lbl.numpy().astype(int)
    y_train = y_train[:,1]
    return x_train_normed,  y_train


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
            

        elif data_config['task'] == 'transfer_learning':
            train_dataset, s_val_dataset, mean, std  = fold_dataset_load_lda(data_config['source']['data_path'], 
                                                                    data_config['source']['sub'], data_config['source']['pos'], 
                                                                    data_config, fold, mean=None, std=None, feats=feats,
                                                                    compute_norms=compute_norms)
        
            val_dataset, _, _  = fold_dataset_load_lda(data_config['target']['data_path'], data_config['target']['sub'],
                                                  data_config['target']['pos'], data_config, fold, mean, std, 
                                                  feats=feats, compute_norms=compute_norms,fold_idx_set_test='all') 
        else:
            raise NotImplementedError
        
        if config['pos_clf'] == True:
            x_train, y_train = prep_data_pos_clf(train_dataset, config)
            x_test,  y_test = prep_data_pos_clf(val_dataset, config)
        else:
            x_train, y_train = prep_data(train_dataset, config)
            x_test,  y_test = prep_data(val_dataset, config)

        #-------------- Compute the LDA accuracy---------------
        # acc = lda_acc(train_dataset, val_dataset, mean, std)
        test_acc, train_acc = lda_acc(x_train, y_train, x_test, y_test)


        print('fold:', fold,  'acc:', test_acc, 'train_acc:', train_acc)

        glob_no += 1
        # Append the metrics to the dataframe
        new_row = pd.DataFrame({'sub': [sub_id],
                                'source': [data_config['source']['pos']],
                                'target': [data_config['target']['pos']],
                                'fold': [fold],
                                'test_acc': [test_acc], 
                                'train_acc': [train_acc]})
                                # 'test_acc': [test_acc],})
        df = pd.concat([df, new_row], ignore_index=True)
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



def run_lda_exp_single_pos(data_config, config, device,df, glob_no, fnames=None):
    for s, seed_fold_gen_path, fold_params in zip(config['seeds'], config['seed_fold_gen_path'], config['fold_params_dirs'] ):
        print('seed_fold_gen_path:', seed_fold_gen_path)

        if fnames is None:
            fnames = ['hudgins_lda_cross_metrics_' + s + '.csv', 'hudgins_lda_diag_metrics_' + s + '.csv']

        for i,(pos, fname) in enumerate(zip(config['poss'], fnames)):
            data_config['feats'] = 'ftd_base'
            for p1 in pos:
                for p2 in pos:
                    if p1 == p2:
                        
                        data_config['task'] = 'source_only'
                    else:
                        data_config['task'] = 'transfer_learning'

                    data_config['source']['pos'] = [str(p1)]
                    data_config['target']['pos'] = [str(p2)]
                    
                    df, glob_no = train_subs(data_config, 
                                             config, 
                                             device, 
                                             data_config['source']['pos'], 
                                             data_config['target']['pos'], 
                                             df, 
                                             glob_no, 
                                             seed_fold_gen_path,
                                             fold_params = fold_params,
                                             compute_norms=False)
                        
            df.to_csv(fname, index=False)


def run_lda_loocv_domain( data_config, config, device, df, glob_no, fnames=None):

    for s, seed_fold_gen_path, fold_params in zip(config['seeds'], config['seed_fold_gen_path'], config['fold_params_dirs'] ):
        print('seed_fold_gen_path:', seed_fold_gen_path)

        if fnames is None:
            fnames = ['lda_cross_loocv_metrics_' + s + '.csv', 'lda_diag_loocv_metrics_' + s + '.csv']

        for i,(pos, fname) in enumerate(zip(config['poss'], fnames)):
            data_config['feats'] = 'ftd_base'

            for elem in pos:
                print('--------------------------------- Test on:', elem, '---------------------------------')
                train_pos =  [str(elem)]
                test_pos = [str(i) for i in pos if i != elem]

                data_config['source']['pos'] = train_pos
                data_config['target']['pos'] = test_pos

                df, glob_no = train_subs(data_config, 
                                         config, 
                                         device, 
                                         train_pos, 
                                         test_pos, 
                                         df, 
                                         glob_no, 
                                         seed_fold_gen_path,
                                         fold_params = fold_params,
                                         compute_norms=True)

            # Save the dataframe to a CSV file
            df.to_csv(fname, index=False)


if __name__ == "__main__":

    # ----------- get configs -----------
    p = os.path.join(os.getcwd(), 'AT_Great', 'experiment_configs', 'exp', 'lda_exp.yaml')
    print('config path:', p)
    config = get_configs(p)
    datafile_path = os.path.abspath(config["dataset"]["config"])
    data_config = get_configs(datafile_path)
    config["dataset"]["config"] = data_config
    config["data_config"] = data_config

    glob_no = 0
    df = pd.DataFrame(columns=['sub', 
                              'source', 
                              'target', 
                              'fold', 
                              'test_acc', 
                              'train_acc'])

    run_lda_loocv_domain( data_config, config, device, df, glob_no)
    #run_lda_exp_single_pos(data_config, config, device, df, glob_no)