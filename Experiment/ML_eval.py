#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import pickle

GBC_params = {'min_samples_split':[2, 3, 4, 5,6], 'min_samples_leaf':[1,2,3,4,5,7], \
                  'max_depth':[1,2,3,4,5,6,7,8,9], 'max_features':['auto','sqrt','log2'], 'n_estimators':[250]}

def minMax1D(arr):
    return ((arr-arr.min())/(arr.max()-arr.min()))

def getFPR(dataframe):
    fp = (dataframe.pred.astype(bool) & (~dataframe.income.astype(bool))).sum()
    n = (~dataframe.income.astype(bool)).sum()
    return round(float(fp)/n,3)

def getFNR(dataframe):
    fn = (~dataframe.pred.astype(bool) & (dataframe.income.astype(bool))).sum()
    p = dataframe.income.astype(bool).sum()
    return round(float(fn)/p,3)

def get_eval_stats(dataframe, fairness='none'):
    dataframe = dataframe.copy()
    dataframe.at[dataframe.sex==1,'sex'] = 'Male'
    dataframe.at[dataframe.sex==0,'sex'] = 'Female'

    array_data = []
    col_names = ['Race','Sex','Population Size','auc_roc','FPR','FNR','fairness']
    acc = round(roc_auc_score(dataframe.income,dataframe.pred),3)
    array_data.append(['All','All', dataframe.shape[0], round(roc_auc_score(dataframe.income,dataframe.pred),3), getFPR(dataframe), getFNR(dataframe), fairness])
    
    for name, group in dataframe.groupby('race'):
        array_data.append([name, 'All', group.shape[0],round(roc_auc_score(group.income,group.pred),3), getFPR(group), getFNR(group), fairness])
       
    for name, group in dataframe.groupby('sex'):
        array_data.append(['All', name, group.shape[0], round(roc_auc_score(group.income,group.pred),3), getFPR(group), getFNR(group), fairness])
    
    for racename, racegroup in dataframe.groupby('race'):
        for sexname, sexgroup in racegroup.groupby('sex'):
            array_data.append([racename,sexname,sexgroup.shape[0], round(roc_auc_score(group.income,group.pred),3), getFPR(sexgroup), getFNR(sexgroup), fairness])
    return pd.DataFrame(array_data, columns=col_names)

def get_sample_weight(df, key='race'):
    scaler = StandardScaler()
    if key == 'race':
        return minMax1D(compute_sample_weight('balanced',df[key])**1)
    elif key == 'sex':
        return minMax1D(compute_sample_weight('balanced',df[key])**1)
    elif key == 'both':
        return minMax1D(compute_sample_weight('balanced',df['race'] + df['sex'].astype(str))**1)

isValidation = np.random.randint(10,size=df.shape[0])>6
dfs = []
valid_dfs = []
GBC_dict = {}
for fairKey in ['none', 'race', 'sex', 'both']:
    isFair = True
    
    if fairKey == 'none':
        isFair = False
    
    df = pd.read_csv('processed_adults.csv')
    df = df[df.race.str.contains('White') | df.race.str.contains('Black')]
    discrete_cols = ['workclass','education','marital-status','occupation','relationship','race','native-country']
    
     #70% train data
    train_df = df[~isValidation].copy()
    
    #train_df = train_df[train_df.sex==0] # trains on only women
    
    X = train_df.drop(discrete_cols+['income'],axis=1).values
    Y = train_df.income.values
    
    kf = KFold(n_splits=20)
    
    results_arr = np.zeros(X.shape[0])
    
    
    '''for train_ind, test_ind in kf.split(X,Y):
        scaler = StandardScaler()
        
        trainX = X[train_ind]
        trainX = scaler.fit_transform(trainX)
        
        testX = X[test_ind]
        testX = scaler.transform(testX)
        
        trainY = Y[train_ind]
        testY = Y[test_ind]
            
        model = GridSearchCV(GBC(), param_grid=GBC_params)
        if isFair:
            weights = get_sample_weight(train_df.iloc[train_ind], key=fairKey)
            model.fit(trainX, trainY, sample_weight=weights)
        else:
            model.fit(trainX, trainY)
        pred = model.predict(testX)
        print(accuracy_score(testY, pred))
        results_arr[test_ind] = pred
    
    train_df['pred'] = results_arr
    
    print('total',accuracy_score(train_df.income,train_df.pred))
    
    dfs.append(get_eval_stats(train_df, fairness=fairKey))'''


    val_df = df[isValidation].copy()
    valX = val_df.drop(discrete_cols+['income'],axis=1).values
    valY = val_df.income.values
    
    scaler = StandardScaler()
    
    model = GridSearchCV(GBC(), param_grid=GBC_params)
    if isFair:
        weights = get_sample_weight(train_df,key=fairKey)
        model.fit(scaler.fit_transform(X), Y, sample_weight=weights)
    else:
        model.fit(scaler.fit_transform(X),Y)
        
    pred = model.predict(scaler.transform(valX))
    print('validation acc',accuracy_score(valY, pred))
    
    val_df['pred'] = pred
    
    valid_dfs.append(get_eval_stats(val_df, fairness=fairKey))
    
    GBC_dict[fairKey] = model


#all_train_results = pd.concat(dfs)
all_val_results = pd.concat(valid_dfs)

#all_train_results.to_csv('all_train_results2.csv',index=False)
all_val_results.to_csv('all_val_results_GBC_gridsearch.csv',index=False)




















