#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 13:57:49 2020

@author: joeyallen
"""
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier as GBC, GradientBoostingRegressor as GBR
import itertools
from sklearn.utils.class_weight import compute_sample_weight
import pickle
from fair import model as fmodel
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

def fair_clf(sense_feats=None, reg=GridSearchCV(GBR(),param_grid=GBC_params, n_jobs=-1), fairness='FP', C=10, gamma=0.01, max_iters=50, verbose=False):
    if sense_feats == None:
        print("Fair model requires sensitive features.")
        exit(-1)
    return fmodel.Model(sense_feats, C=C, printflag=verbose, max_iters=max_iters, gamma=0.01, fairness_def=fairness, predictor=reg)

def process_categorical(df, discrete_cols):
    for col in discrete_cols:
        for val in df[col].unique():
            df[col+'_is_'+val] = (df[col]==val).astype(int)
    return df.drop(discrete_cols, axis=1)

def return_combo_arrs(arr):
    if len(arr) == 1:
        return [arr]
    elif len(arr) == 2:
        return [[arr[0]],[arr[1]],arr]
       
def minMax1D(arr):
    return ((arr-arr.min())/(arr.max()-arr.min()))

def get_sample_weight(df, key=['race']):
    if len(key) == 1:
        return minMax1D(compute_sample_weight('balanced',df[key[0]]))
    elif len(key)==2:
        return minMax1D(compute_sample_weight('balanced',df[key[0]].astype(str) + df[[key[1]]].astype(str)))

def arr_to_string(arr):
    return 'AND'.join(arr)


GBC_params = {'min_samples_split':[2, 3, 5], 'min_samples_leaf':[1,2,7], \
                  'max_depth':[1,2,3,4], 'max_features':['auto','sqrt','log2'], 'n_estimators':[250,500]}

DT_params = {'min_samples_split':[2, 3, 4, 5,6], 'min_samples_leaf':[1,2,3,4,5,7], \
                  'max_depth':[2,4,8,16,32], 'max_features':['auto','sqrt','log2']}

SVM_params = {'kernel':['linear', 'poly', 'rbf'], 'gamma':['scale','auto'],'probability':[True],'cache_size':[1024]}

SVR_params = {'kernel':['linear', 'poly', 'rbf'], 'gamma':['scale','auto'],'cache_size':[1024]}

model_names = ['GB','Tree','SVM','LR']
models = [GridSearchCV(GBC(), param_grid=GBC_params, n_jobs=-1),\
          GridSearchCV(DecisionTreeClassifier(), param_grid=DT_params),\
          GridSearchCV(SVC(), param_grid=SVM_params),\
          LogisticRegression()
          ]

base_fairer_regressors = [GridSearchCV(GBR(),param_grid=GBC_params, n_jobs=-1),\
                          GridSearchCV(DecisionTreeRegressor(),param_grid=DT_params),\
                          GridSearchCV(SVR(),param_grid=SVR_params),\
                          LinearRegression() ]
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
    hasIndex = dataset in ['communities','lawschool2']
    if hasIndex:
        index_col = 0
    else:
        index_col = None
    
    df = pd.read_csv('data/'+dataset+'.csv', index_col=index_col)
    df.drop(cols_to_drop, axis=1, inplace=True)
    discrete_cols = [x for x in df.columns if df[x].dtype==np.dtype('O')]
    df = process_categorical(df, discrete_cols)
    X = df.drop([label],axis=1).values
    Y = df[label].values
    kf = KFold(n_splits=5)
    i = 0
    for train_ind, test_ind in kf.split(X,Y):
        i += 1
        scaler = StandardScaler()
        
        trainX = X[train_ind]
        trainX = scaler.fit_transform(trainX)
        
        testX = X[test_ind]
        testX = scaler.transform(testX)
        
        trainY = Y[train_ind]
        testY = Y[test_ind]
        with open(models_dir+'scaler_'+dataset+'split'+str(i)+'.pickle','wb') as handle:
                pickle.dump(scaler, handle, pickle.HIGHEST_PROTOCOL)
        df.iloc[test_ind].to_csv(models_dir+'test_dataframe_'+dataset+'split'+str(i)+'.csv')
        
        for model, modelname in zip(models, model_names):
            model.fit(trainX, trainY)
            with open(models_dir+'optimal_'+modelname+dataset+'split'+str(i)+'.pickle','wb') as handle:
                pickle.dump(model, handle, pickle.HIGHEST_PROTOCOL)
            print(dataset, modelname, roc_auc_score(testY, model.predict(testX)))
        
        for feat_combo in return_combo_arrs(sens_feats):
            '''naive fairness'''
            sample_weight = get_sample_weight(df.iloc[train_ind], feat_combo)
            for model, modelname in zip(models,model_names):
                model.fit(trainX,trainY, sample_weight=sample_weight)
                with open(models_dir+arr_to_string(feat_combo)+'_'+modelname+dataset+'split'+str(i)+'.pickle','wb') as handle:
                    pickle.dump(model, handle, pickle.HIGHEST_PROTOCOL)
                print(dataset, modelname, arr_to_string(feat_combo), roc_auc_score(testY, model.predict(testX)))
            '''better fairness'''
            for gamma in fair_gamma_opts:
                for regressor, modelname in zip(base_fairer_regressors, model_names):
                    model = fair_clf(sense_feats=feat_combo, reg=regressor, gamma=gamma)
                    fair_trainX = pd.DataFrame(scaler.transform(df.iloc[train_ind].drop([label],axis=1).values), columns=df.iloc[train_ind].drop([label],axis=1).columns)
                    fair_testX = pd.DataFrame(scaler.transform(df.iloc[test_ind].drop([label],axis=1).values), columns=df.iloc[test_ind].drop([label],axis=1).columns)
                    model.fit(fair_trainX,df.iloc[train_ind][label])
                    with open(models_dir+arr_to_string(feat_combo)+'_betterFairness_gamma'+str(gamma)+modelname+dataset+'split'+str(i)+'.pickle','wb') as handle:
                        pickle.dump(model, handle, pickle.HIGHEST_PROTOCOL)
                    print(dataset, modelname, 'better', arr_to_string(feat_combo), roc_auc_score(testY, model.predict(df.iloc[test_ind].drop([label],axis=1))))
        
                
            
            
            
        
        
    