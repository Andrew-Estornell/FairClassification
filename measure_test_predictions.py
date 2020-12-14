#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 22:20:55 2020

@author: joeyallen
"""
from sklearn.metrics import roc_auc_score
import pickle
import pandas as pd
import numpy as np


def getFPR(dataframe, label='income'):
    fp = (dataframe.pred.astype(bool) & (~dataframe[label].astype(bool))).sum()
    n = (~dataframe[label].astype(bool)).sum()
    return round(float(fp)/n,3)

def getFNR(dataframe,label='income'):
    fn = (~dataframe.pred.astype(bool) & (dataframe[label].astype(bool))).sum()
    p = dataframe[label].astype(bool).sum()
    return round(float(fn)/p,3)

def get_eval_stats(dataframe, sense_feats=['is_White','gender'], label='income'):
    dataframe = dataframe.copy()
    array_data = []
    if len(sense_feats) == 1:
        col_names = [sense_feats[0], 'Population Size', 'auc_roc','FPR','FNR']
        array_data.append(['Both', dataframe.shape[0], roc_auc_score(dataframe[label], dataframe.p_prob), getFPR(dataframe, label), getFNR(dataframe,label)])
        grouped = dataframe.groupby(sense_feats)
        for name, group in dataframe.groupby(sense_feats):
            array_data.append([name, group.shape[0], roc_auc_score(group[label], group.p_prob), getFPR(group,label), getFNR(group,label)])
    elif len(sense_feats) == 2:
        col_names = [sense_feats[0], sense_feats[1], 'Population Size', 'auc_roc','FPR','FNR']
        array_data.append(['Both', 'Both',dataframe.shape[0], roc_auc_score(dataframe[label], dataframe.p_prob), getFPR(dataframe, label), getFNR(dataframe,label)])
        grouped = dataframe.groupby(sense_feats[0])
        for name, group, in grouped:
            array_data.append([name, 'Both',group.shape[0], roc_auc_score(group[label], group.p_prob), getFPR(group, label), getFNR(group, label)])
        
        grouped = dataframe.groupby(sense_feats[1])
        for name, group, in grouped:
            array_data.append(['Both', name, group.shape[0], roc_auc_score(group[label], group.p_prob), getFPR(group, label), getFNR(group,label)])
        
        grouped = dataframe.groupby(sense_feats)
        for name, group in dataframe.groupby(sense_feats):
            array_data.append([name[0], name[1], group.shape[0], roc_auc_score(group[label], group.p_prob), getFPR(group,label), getFNR(group,label)])
        
    return pd.DataFrame(array_data, columns=col_names)

def arr_to_string(arr):
    return 'AND'.join(arr)

def return_combo_arrs(arr):
    if len(arr) == 1:
        return [arr]
    elif len(arr) == 2:
        return [[arr[0]],[arr[1]],arr]

model_names = ['GB','SVM','LR']

fair_gamma_opts = [0.2,0.1,0.01]
fairer_models = []
fairer_model_names = []

datasets = ['adult','Data_1980','lawschool2','student-mat']
labels = ['income','RECID','bar1','G3']
cols_to_drop_sets = [ [], ['TIME', 'FILE'], ['cluster'],[]]

sens_feats_sets = [['race_is_ White', 'sex_is_ Male'],\
                   ['WHITE'],\
                   ['gender','race'],\
                   ['sex_is_M']\
                   ]

models_dir = 'Models/'

for dataset, label, cols_to_drop, sens_feats in zip(datasets, labels, cols_to_drop_sets, sens_feats_sets):
    scalers = {}
    test_data = {}
    all_model_data = pickle.load(open('Models/'+dataset+'_trainedModelInfo.pickle', 'rb'))
    for i in range(1,6):
        scalers[i] = all_model_data['split'+str(i)]['scaler']
        test_data[i] = all_model_data['split'+str(i)]['test_data']
    for modelname in model_names:
        '''evaluate baseline models'''
        dfs = []
        for i in range(1,6):
            model = all_model_data['split'+str(i)]['models']['optimal_'+modelname]
            testX = scalers[i].transform(test_data[i].drop([label],axis=1).values)
            testY = test_data[i][label].values
            
            pred = model.predict(testX)
            eval_df = test_data[i].copy()
            eval_df['pred'] = pred
            eval_df['p_prob'] = model.predict_proba(testX)[:,1]
            dfs.append(eval_df.copy())
        
        print(dataset, modelname, get_eval_stats(pd.concat(dfs), sens_feats, label))
        
        
    for feat_combo in return_combo_arrs(sens_feats):
        for modelname in model_names:
            for i in range(1,6):
                naivefair_model = all_model_data['split1']['models'][arr_to_string(feat_combo)+'_'+modelname]
        
        for modelname in model_names:
            for i in range(1,6):
                for gamma in fair_gamma_opts:
                    betterfair_model = all_model_data['split1']['models'][arr_to_string(feat_combo)+'_betterFairness_gamma'+str(gamma)+modelname]
        #fair_testX = pd.DataFrame(testX, columns=test_data[i].drop([label],axis=1).columns)
        #fair_testY = test_data[i][label]
            
            
    
    
