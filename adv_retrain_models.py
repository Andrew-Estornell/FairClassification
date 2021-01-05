#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 13:38:50 2021

@author: joeyallen
"""
import numpy as np
import pandas as pd
import pickle as pkl
import itertools as itr
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, make_scorer

from fair.model import Model as FModel
from prep.prep import prep
import optimal_decision_making.manipulation as manip
from equalibrium_exp import false_positive_rate, false_negative_rate, compute_metric_across_groups, compute_roc_across_groups


def gen_param_grid(param_grid):
    keys = list(param_grid.keys())
    combinations = [{keys[i]: combo[i] for i in range(len(combo))} for combo in list(itr.product(*(param_grid[key] for key in keys)))]
    return {'predictor_param_dict': combinations}


def gen_sample_weights(data, feat):
    g1_size = sum(data[feat])
    g2_size = len(data) - g1_size
    weights = np.array([1.0/g1_size if data[feat].iloc[i] == 1 else 1.0/g2_size for i in range(len(data))])
    return weights


if __name__ == '__main__':
    np.random.seed(101)

    info = [('data/Data_1980.csv',   'RECID',  ['TIME', 'FILE'],    {},                                                          ['WHITE'], 1),
            ('data/adult.csv',       'income', ['native-country'],  {'race': (' White', ' Black'), 'sex': (' Male', ' Female')}, ['race'],  0),
            ('data/lawschool2.csv',  'bar1',   ['cluster'],         {},                                                          ['race'],  0),
            ('data/student-mat.csv', 'G3',     [],                  {'sex': ('M', 'F'), 'school': ('GP', 'MS'), 'address': ('U', 'R'), 'famsize': ('GT3', 'LE3'), 'Pstatus': ('A', 'T'), 'schoolsup': ('yes', 'no'), 'famsup': ('yes', 'no'), 'paid': ('yes', 'no'), 'activities': ('yes', 'no'), 'nursery': ('yes', 'no'), 'higher': ('yes', 'no'), 'internet': ('yes', 'no'), 'romantic': ('yes', 'no')},  ['sex'],   0)]

    f_save_names = ['recidivism', 'adult', 'lawschool', 'student']

    scoring = make_scorer(roc_auc_score, needs_proba=True)
    n_jobs = -1
    gammas = [0.2, 0.1, 0.01]
    k = 5

    base_params = {'GB': {'max_depth':    [2, 5, 10],
                          'n_estimators': [75*i + 50 for i in range(5)]},
                   'SV': {'kernel': ['linear', 'poly', 'rbf'],
                          'C':      [10, 1, 0.1, 0.01]},
                   'LG': {'max_iter': [500, 1000, 2000],
                          'C':        [10, 1, 0.1, 0.01]}}

    reg_params = {'LR': {'fit_intercept': [True]},
                  'DT': {'max_depth':    [2*i for i in range(1, 5)],
                         'max_features': [1, 2, 5]}}
    
    num_iters = 5
    manip_cols_all = [['race', 'sex', 'workclass', 'marital-status'],
				      ['WHITE', 'ALCHY', 'JUNKY', 'MARRIED', 'MALE'],
				      ['gender', 'race', 'fulltime', 'fam_inc'],
				      ['sex', 'freetime', 'studytime', 'goout', 'Fedu']]
    #alphas = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.1, 0.01, 0]

    ctr = 0
    for file_name, target_column, cols_to_remove, variables_to_be_made_binary, sensative_features, flip_0_and_1_labes in info:
        print(file_name)
        outputs = []
        if 'adult' in file_name:
            alpha = 0.2
            manip_cols = manip_cols_all[0]
        elif 'Data_1980' in file_name:
            alpha = 0.7
            manip_cols = manip_cols_all[1]
        elif 'lawschool' in file_name:
            alpha = 0.1
            manip_cols = manip_cols_all[2]
        else:
            alpha = 0.1
            manip_cols = manip_cols_all[3]
        #################
        # PREPROCESS DATA
        #################
        X, y = prep(file_name, target_column, cols_to_remove, flip_0_and_1_labes, bin_vals=variables_to_be_made_binary)

        ###############
        # MODEL SET UP
        ###############
        base_clfs = {'GB_base':          GridSearchCV(GradientBoostingClassifier(),                                                              param_grid=base_params['GB'],                n_jobs=n_jobs, scoring=scoring),
                     'SV_base':          GridSearchCV(SVC(probability=True),                                                                     param_grid=base_params['SV'],                n_jobs=n_jobs, scoring=scoring),
                     'LG_base':          GridSearchCV(LogisticRegression(),                                                                      param_grid=base_params['LG'],                n_jobs=n_jobs, scoring=scoring)}
        equa_clfs = {'GB_equal':         GridSearchCV(GradientBoostingClassifier(),                                                              param_grid=base_params['GB'],                n_jobs=n_jobs, scoring=scoring),
                     'SV_equal':         GridSearchCV(SVC(probability=True),                                                                     param_grid=base_params['SV'],                n_jobs=n_jobs, scoring=scoring),
                     'LG_equal':         GridSearchCV(LogisticRegression(),                                                                      param_grid=base_params['LG'],                n_jobs=n_jobs, scoring=scoring)}
        fair_lrgs = {'LR_' + str(gamma): FModel(sensative_features, predictor=LinearRegression(),                   gamma=gamma, max_iters=10)                                                                              for gamma in gammas}
        fair_dtrs = {'DT_' + str(gamma): GridSearchCV(FModel(sensative_features, predictor=DecisionTreeRegressor(), gamma=gamma, max_iters=10), param_grid=gen_param_grid(reg_params['DT']), n_jobs=n_jobs, scoring=scoring) for gamma in gammas}

        #############################
        # SPLIT DATA AND TRAIN MODELS
        #############################
        results_data = []
        
        kf = KFold(n_splits=k)
        for train_index, test_index in kf.split(X, y):
            split_data = {}
            X_train, y_train, X_test, y_test = X.iloc[train_index], y.iloc[train_index], X.iloc[test_index], y.iloc[test_index]
            outputs.append({'data':   (X_test, y_test),
                            'models': {}})
            g0_index = [i for i in range(len(X_test)) if X_test[sensative_features[0]].iloc[i] == 0]
            g1_index = [i for i in range(len(X_test)) if X_test[sensative_features[0]].iloc[i] == 1]
            for clf_name, clf in list(base_clfs.items()) + list(equa_clfs.items()) + list(fair_lrgs.items()) + list(fair_dtrs.items()):
                print(clf_name, end=', ')
                split_data[clf_name] = {}
                
                if 'equal' in clf_name:
                    clf.fit(X_train, y_train, sample_weight=gen_sample_weights(X_train, sensative_features[0]))
                else:
                    clf.fit(X_train, y_train)
                
                pred_p = clf.predict_proba(X_test)[:, 1]
                pred = clf.predict(X_test)
                print(roc_auc_score(y_test, pred_p))
                # calculate metrics on test set
                g0_true_fp, g1_true_fp = compute_metric_across_groups(y_test.to_numpy(), pred,      g0_index, g1_index, false_positive_rate)
                g0_true_fn, g1_true_fn = compute_metric_across_groups(y_test.to_numpy(), pred,      g0_index, g1_index, false_negative_rate)
                g0_true_roc, g1_true_roc = compute_roc_across_groups(y_test.to_numpy(), pred_p,      g0_index, g1_index, roc_auc_score)
                
                split_data[clf_name]['base'] = {}
                split_data[clf_name]['base']['g0'] = {'FP':g0_true_fp, 'FN':g0_true_fn,'roc':g0_true_roc}
                split_data[clf_name]['base']['g1'] = {'FP':g1_true_fp, 'FN':g1_true_fn,'roc':g1_true_roc}
                
                [all_new_X], opt_pred_ps_all_clfs, opt_preds_all_clfs = manip.optimal_agent_strats_for_cata_features(X_train, y_train, manip_cols, np.array([clf]), alpha=alpha, decision_type='preds')
                new_X=[]
                for row in all_new_X:
                    new_X.append(row[0])
                new_X = pd.DataFrame(new_X, columns=X.columns)
                
                for i in range(num_iters):
                    if 'equal' in clf_name:
                        clf.fit(X_train, y_train, sample_weight=gen_sample_weights(X_train, sensative_features[0]))
                    else:
                        clf.fit(X_train, y_train)
                    [all_new_X], opt_pred_ps_all_clfs, opt_preds_all_clfs = manip.optimal_agent_strats_for_cata_features(X, y, manip_cols, np.array([clf]), alpha=alpha, decision_type='preds')
                    new_X=[]
                    for row in all_new_X:
                        new_X.append(row[0])
                    new_X = pd.DataFrame(new_X, columns=X.columns)
                    clf.fit(new_X, y)
    
                    pred_p = clf.predict_proba(X_test)[:, 1]
                    pred = clf.predict(X_test)
                    print(roc_auc_score(y_test, pred_p))
                    # calculate metrics on test set
                    g0_true_fp, g1_true_fp = compute_metric_across_groups(y_test.to_numpy(), pred,      g0_index, g1_index, false_positive_rate)
                    g0_true_fn, g1_true_fn = compute_metric_across_groups(y_test.to_numpy(), pred,      g0_index, g1_index, false_negative_rate)
                    g0_true_roc, g1_true_roc = compute_roc_across_groups(y_test.to_numpy(), pred_p,      g0_index, g1_index, roc_auc_score)
                    
                    split_data[clf_name]['iter_'+str(i)] = {}
                    split_data[clf_name]['iter_'+str(i)]['g0'] = {'FP':g0_true_fp, 'FN':g0_true_fn,'roc':g0_true_roc}
                    split_data[clf_name]['iter_'+str(i)]['g1'] = {'FP':g1_true_fp, 'FN':g1_true_fn,'roc':g1_true_roc}
                
                    #outputs[-1]['models'][clf_name] = clf
            results_data.append(split_data)
        with open('Outputs/' + f_save_names[ctr] + '_advRetrained.pickle', 'wb') as handle:
            pkl.dump(results_data, handle, pkl.HIGHEST_PROTOCOL)
        ctr += 1











