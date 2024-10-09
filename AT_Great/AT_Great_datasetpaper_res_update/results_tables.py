import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import t
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np


def read_pos_clf(fname):
    df = pd.read_csv(fname)
    avg = df.iloc[:, 4:].mean().round(2)
    std = df.iloc[:, 4:].std().round(2)
    print("Average:\n", avg)
    print("Standard Deviation:\n", std)
    return df

def read_one_to_one(fname):
    df = pd.read_csv(fname)
    grouped = df.groupby(['source', 'target']).agg(['mean', 'std']).round(2)
    print("Grouped Mean and Standard Deviation:\n", grouped)
    return df


if __name__ == "__main__":

    # -------------  pos_clf ------------- 
    # fname1 = r'/home/kasia/AT_Great/AT_Great/AT_Great_datasetpaper_res_update/pos_clf/cross_pos_clf_s200.csv'
    # fname2 = r'/home/kasia/AT_Great/AT_Great/AT_Great_datasetpaper_res_update/pos_clf/diag_pos_clf_s200.csv'
    # df1 = read_pos_clf(fname2)

    # -------------  one to one experiments ------------- 
    # fname1 = r'/home/kasia/AT_Great/AT_Great/AT_Great_datasetpaper_res_update/one-to-one/hudgins_lda_cross_metrics_s200.csv'
    # fname2 = r'/home/kasia/AT_Great/AT_Great/AT_Great_datasetpaper_res_update/one-to-one/hudgins_lda_diag_metrics_s200.csv'
    # read_one_to_one(fname2)

    # -------------  read OVR experiments -------------  
    '''
    In this setting I am training on n-1 positions and testing on the remianing position.
    '''
    # fname1 = r'/home/kasia/AT_Great/AT_Great/AT_Great_datasetpaper_res_update/OVR/lda_cross_loocv_metrics_s200.csv'
    # fname2 = r'/home/kasia/AT_Great/AT_Great/AT_Great_datasetpaper_res_update/OVR/lda_diag_loocv_metrics_s200.csv'
    # read_one_to_one(fname1)

    '''
    In this setting I am testing on n-1 positions and training on the remianing position.
    '''
    # fname1 = r'/home/kasia/AT_Great/AT_Great/AT_Great_datasetpaper_res_update/OVR_train1pos_testremaining/lda_cross_loocv_metrics_s200.csv'
    # fname2 = r'/home/kasia/AT_Great/AT_Great/AT_Great_datasetpaper_res_update/OVR_train1pos_testremaining/lda_diag_loocv_metrics_s200.csv'
    # read_one_to_one(fname2)

    # -------------  read HMC and SoftHMC experiments -------------
    # fname1 = r'/home/kasia/AT_Great/AT_Great/AT_Great_datasetpaper_res_update/SoftHMC/hmc_naive_seed200.csv'
    # fname2 = r'/home/kasia/AT_Great/AT_Great/AT_Great_datasetpaper_res_update/HMC/hmc_strict_seed200.csv'
    # read_pos_clf(fname2)
