import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# from at_source.modelling.lda import *
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def z_score(x_train, x_test):
    scaler = StandardScaler()
    scaler = scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test


def lda_acc(x_test, y_test, x_train, y_train, positions=None):
    x_train, x_test = z_score(x_train, x_test)

    # Create an LDA classifier
    lda = LDA()
    lda.fit(x_train, y_train.ravel())
    y_pred = lda.predict(x_test)
    accuracy = accuracy_score(y_test.ravel(), y_pred)
    #print(f"Grasp: {positions}, Accuracy:, {accuracy}")
    return accuracy

def lda_model(x_train, y_train):
    lda = LDA()
    lda.fit(x_train, y_train)
    return lda



def clf_2lvl_acc(x_test, y_test, z_train, x_train, y_train, z_test):
    # ---------- Norm Data ----------
    x_train, x_test = z_score(x_train, x_test)

    # ---------- TRAINING ----------
    # FIT x to the first encoder
    lda1 = lda_model(x_train, z_train) 
    x_transformed = lda1.transform(x_train)
    z_prime_train = lda1.predict(x_train)

    # FIT the second encoder
    x_transformed = np.concatenate((x_train, z_prime_train.reshape(-1, 1)), axis=1)
    lda2 = lda_model(x_transformed, y_train) 
    

    # ---------- TESTING ----------
    predict_lvl1 = lda1.predict(x_test)
    predict_lvl2 = lda2.predict(np.concatenate((x_test, predict_lvl1.reshape(-1, 1)), axis=1))

    return predict_lvl2



def h_clf_acc(x_test, y_test, z_test, x_train, y_train, z_train, config):
    
    # ---------- Norm Data ----------
    # x_train, x_test = z_score(x_train, x_test)

    # ---------- TRAINING ----------
    # FIT x to the first encoder
    lda1 = lda_model(x_train, z_train) 
    z_prime_train = lda1.predict(x_train)
    print('acc train lvl 1:', accuracy_score(z_train, z_prime_train))
    # Apply min-max normalization - cause the values of position are  muhc higher than the x_train data
    z_prime_train_norm = (z_prime_train - np.min(z_prime_train)) / (np.max(z_prime_train) - np.min(z_prime_train))

    # FIT the second encoder
    _dict = group_by(z_train)
    x_transformed = np.concatenate((x_train, z_prime_train_norm.reshape(-1, 1)), axis=1)
    lvl2_models = fit_level2(_dict, x_transformed, y_train)

    # ---------- TESTING ----------
    z_prime_test = lda1.predict(x_test)
    z_prime_test_norm = (z_prime_test - np.min(z_prime_test)) / (np.max(z_prime_test) - np.min(z_prime_test))
    xtest_transformed = np.concatenate((x_test, z_prime_test_norm.reshape(-1, 1)), axis=1)
    # predict_lvl1 = lda1.predict(x_test)  # predict the position
    test_dict = group_by(z_prime_test)  # group data based on the prediction label  
    predict_lvl2 = predict_level2(z_prime_test, lvl2_models, test_dict, xtest_transformed, y_test, z_test, config)
    z_acc = accuracy_score(z_test, z_prime_test)
    return predict_lvl2, z_acc



def fit_level2(_dict, x_transformed, y):
    for z_key, _ in _dict.items():
        idx = np.array(_dict[z_key])
        # print('IDX:', idx.shape, type(idx[0]))
        if z_key == 1:
            lda_z1 = lda_model(x_transformed[idx], y[idx])
        elif z_key == 2:
            lda_z2 = lda_model(x_transformed[idx], y[idx])
        elif z_key == 3:
            lda_z3 = lda_model(x_transformed[idx], y[idx])
        elif z_key == 4:
            lda_z4 = lda_model(x_transformed[idx], y[idx])
        elif z_key == 5:
            lda_z5 = lda_model(x_transformed[idx], y[idx])
        elif z_key == 6:
            lda_z6 = lda_model(x_transformed[idx], y[idx])
        elif z_key == 7:
            lda_z7 = lda_model(x_transformed[idx], y[idx])
        elif z_key == 8:
            lda_z8 = lda_model(x_transformed[idx], y[idx])
        elif z_key == 9:
            lda_z9 = lda_model(x_transformed[idx], y[idx])
        else:
            raise ValueError ("The given position z does not exist.")
    return {1: lda_z1,
            2: lda_z2,
            3: lda_z3,
            4: lda_z4,
            5: lda_z5,
            6: lda_z6,
            7: lda_z7,
            8: lda_z8,
            9: lda_z9}


def predict_level2(pred_pos, lvl2_models, pred_dict, xtest_transformed, y_test, z_test, config):
    for z_key, idx in pred_dict.items():

        if z_key == 1:
            final_pred = lda2_predict(lvl2_models[z_key], xtest_transformed[idx])
            score1 = multilbl_eval(pred_pos[idx], final_pred, z_test[idx], y_test[idx], config['mode'])
            print(f'Key {z_key}, acc: {score1}')

        elif z_key == 2:
            final_pred = lda2_predict(lvl2_models[z_key], xtest_transformed[idx])
            score2 = multilbl_eval(pred_pos[idx], final_pred, z_test[idx], y_test[idx], config['mode'])
            print(f'Key {z_key}, acc: {score2}')

        elif z_key == 3:
            final_pred = lda2_predict(lvl2_models[z_key], xtest_transformed[idx])
            score3 = multilbl_eval(pred_pos[idx], final_pred, z_test[idx], y_test[idx], config['mode'])
            print(f'Key {z_key}, acc: {score3}')

        elif z_key == 4:
            final_pred = lda2_predict(lvl2_models[z_key], xtest_transformed[idx])
            score4 = multilbl_eval(pred_pos[idx], final_pred, z_test[idx], y_test[idx], config['mode'])
            print(f'Key {z_key}, acc: {score4}')

        elif z_key == 5:
            final_pred = lda2_predict(lvl2_models[z_key], xtest_transformed[idx])
            score5 = multilbl_eval(pred_pos[idx], final_pred, z_test[idx], y_test[idx], config['mode'])
            print(f'Key {z_key}, acc: {score5}')

        elif z_key == 6:
            final_pred = lda2_predict(lvl2_models[z_key], xtest_transformed[idx])
            score6 = multilbl_eval(pred_pos[idx], final_pred, z_test[idx], y_test[idx], config['mode'])
            print(f'Key {z_key}, acc: {score6}')

        if z_key == 7:
            final_pred = lda2_predict(lvl2_models[z_key], xtest_transformed[idx])
            score7 = multilbl_eval(pred_pos[idx], final_pred, z_test[idx], y_test[idx], config['mode'])
            print(f'Key {z_key}, acc: {score7}')

        elif z_key == 8:
            final_pred = lda2_predict(lvl2_models[z_key], xtest_transformed[idx])
            score8 = multilbl_eval(pred_pos[idx], final_pred, z_test[idx], y_test[idx], config['mode'])
            print(f'Key {z_key}, acc: {score8}')

        elif z_key == 9:
            final_pred = lda2_predict(lvl2_models[z_key], xtest_transformed[idx])
            score9 = multilbl_eval(pred_pos[idx], final_pred, z_test[idx], y_test[idx], config['mode'])
            print(f'Key {z_key}, acc: {score9}')

    # todo return all scores1-9) and avg them across all the cases.
    res = np.array([score1, score2, score3, score4, score5, score6, score7, score8, score9])
    return res


def group_by(y):
    pos = np.unique(y)
    temp_dict = {}
    for p in pos:
        temp_dict[p] = np.where(y == p)[0]
    return temp_dict

def multilbl_eval(pred_pos, final_pred, z_test, y_test, mode='naive'): #'naive'
    temp_score = 0

    if mode == 'naive': # free to predict any position
        for i in range(len(pred_pos)):
            if final_pred[i] == y_test[i]:
                temp_score += 1
    elif mode == 'strict':
        for i in range(len(pred_pos)):
            if pred_pos[i] == z_test[i] and final_pred[i] == y_test[i]:
                temp_score += 1
        
    acc = temp_score/int(len(pred_pos))
    return acc


def lda2_predict(model, x_transformed):
    predicted = model.predict(x_transformed)
    return predicted