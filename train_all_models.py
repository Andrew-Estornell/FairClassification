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
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import itertools as itr
from sklearn.metrics import balanced_accuracy_score, make_scorer
import copy as copy


np.random.seed(42)

def gen_param_grid(param_grid):
    keys = list(param_grid.keys())
    combinations = [{keys[i]: combo[i] for i in range(len(combo))} for combo in list(itr.product(*(param_grid[key] for key in keys)))]
    return {'predictor_param_dict': combinations}

def fair_clf(sense_feats=None, reg=LinearRegression(), fairness='FP', C=10, gamma=0.01, max_iters=50, verbose=False):
    if sense_feats == None:
        print("Fair model requires sensitive features.")
        exit(-1)
    return fmodel.Model(sense_feats, C=C, printflag=verbose, max_iters=max_iters, gamma=gamma, fairness_def=fairness, predictor=reg)

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
    if np.unique(arr).shape[0]==1:
        return np.ones(arr.shape[0])
    else:
        return ((arr-arr.min())/(arr.max()-arr.min()))

def get_sample_weight(df, key=['race']):
    if len(key) == 1:
        return minMax1D(compute_sample_weight('balanced',df[key[0]]))
    elif len(key)==2:
        return minMax1D(compute_sample_weight('balanced',df[key[0]].astype(str) + df[key[1]].astype(str)))

def arr_to_string(arr):
    return 'AND'.join(arr)


GBC_params = {'max_depth':[3, 6, 10], 'n_estimators':[250,500]}

#DT_params = {'min_samples_split':[2, 3, 4, 5,6], 'min_samples_leaf':[1,2,3,4,5,7], \
#                  'max_depth':[2,4,8,16,32], 'max_features':['auto','sqrt','log2']}

SVM_params = {'kernel':['linear', 'poly', 'rbf'], 'gamma':['scale','auto'],'cache_size':[1024]}

log_params = {'max_iters'}
#SVR_params = {'kernel':['linear', 'poly', 'rbf'], 'gamma':['scale','auto'],'cache_size':[1024]}

mld_params = {'GB': GBC_params, 'SVM': SVM_params, 'LR': {'fit_intercept':[True]}}

model_names = ['GB','SVM','LR']

reg_names = ['GB', 'DT', 'LR']
DT_params = {'max_depth': [2, 5, 10], 'max_features': [2, 4]}
reg_params_ = {'GB': GBC_params, 'DT': DT_params, 'LR': {'fit_intercept':[True]}}

n_jobs=-1
models = [GridSearchCV(GBC(), param_grid=GBC_params, n_jobs=n_jobs, cv=5),\
          #GridSearchCV(DecisionTreeClassifier(), param_grid=DT_params, cv=5),\
          GridSearchCV(SVC(probability=True), param_grid=SVM_params, n_jobs=n_jobs, cv=5),\
          LogisticRegression()
          ]

#base_fairer_regressors = [GridSearchCV(GBR(), param_grid=GBC_params, n_jobs=-1, cv=5),\
#                          GridSearchCV(DecisionTreeRegressor(),param_grid=DT_params, cv=5),\
#                          GridSearchCV(SVR(),param_grid=SVR_params, cv=5),\
#                          LinearRegression() ]
base_fairer_regressors = [GBR(), DecisionTreeRegressor(), LinearRegression()]
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
    #if dataset in ['adult','Data_1980','lawschool2']:
        #continue
    print(dataset)
    
    hasIndex = dataset in ['communities','lawschool2']
    if hasIndex:
        index_col = 0
    else:
        index_col = None
    
    dataset_saveData = {}
    
    df = pd.read_csv('data/'+dataset+'.csv', index_col=index_col)
    df.drop(cols_to_drop, axis=1, inplace=True)
    discrete_cols = [x for x in df.columns if df[x].dtype==np.dtype('O')]
    df = process_categorical(df, discrete_cols)
    X = df.drop([label],axis=1).values
    Y = df[label].values
    print(X.shape)
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
        dataset_saveData['split'+str(i)] = {}
        #with open(models_dir+'scaler_split'+str(i)+'.pickle','wb') as handle:
                #pickle.dump(scaler, handle, pickle.HIGHEST_PROTOCOL)
        #dataset_saveData['split'+str(i)]['scaler'] = scaler
        dataset_saveData['split'+str(i)]['test_data'] = (copy.deepcopy(testX), copy.deepcopy(testY), df.columns)#df.iloc[test_ind].copy()
        dataset_saveData['split'+str(i)]['models'] = {}
        
        df.iloc[test_ind].to_csv(models_dir+'test_dataframe_'+dataset+'split'+str(i)+'.csv')
        
        for model, modelname in zip(models, model_names):
            model.fit(trainX, trainY)
            #with open(models_dir+'optimal_'+modelname+dataset+'split'+str(i)+'.pickle','wb') as handle:
                #pickle.dump(model, handle, pickle.HIGHEST_PROTOCOL)
            dataset_saveData['split'+str(i)]['models']['optimal_'+modelname] = model
            print(dataset, modelname, roc_auc_score(testY, model.predict(testX)))

        for feat_combo in return_combo_arrs(sens_feats):
            '''naive fairness'''
            sample_weight = get_sample_weight(df.iloc[train_ind], feat_combo)
            for model, modelname in zip(models,model_names):
                model.fit(trainX,trainY, sample_weight=sample_weight)
                #with open(models_dir+arr_to_string(feat_combo)+'_'+modelname+dataset+'split'+str(i)+'.pickle','wb') as handle:
                    #pickle.dump(model, handle, pickle.HIGHEST_PROTOCOL)
                dataset_saveData['split'+str(i)]['models'][arr_to_string(feat_combo)+'_'+modelname] = model
                print(dataset, modelname, arr_to_string(feat_combo), roc_auc_score(testY, model.predict(testX)))

            print('better fairness')
            for gamma in fair_gamma_opts:
                for regressor, modelname in zip(base_fairer_regressors, reg_names):
                    model = fair_clf(sense_feats=feat_combo, reg=copy.deepcopy(regressor), gamma=gamma)#, verbose=True)
                    reg_params = gen_param_grid(reg_params_[modelname])
                    model = GridSearchCV(model, reg_params, n_jobs=n_jobs, scoring=make_scorer(balanced_accuracy_score), cv=5)



                    fair_trainX = pd.DataFrame(scaler.transform(df.iloc[train_ind].drop([label],axis=1).values), columns=df.iloc[train_ind].drop([label],axis=1).columns)
                    fair_testX = pd.DataFrame(scaler.transform(df.iloc[test_ind].drop([label],axis=1).values), columns=df.iloc[test_ind].drop([label],axis=1).columns)
                    model.fit(fair_trainX,df.iloc[train_ind][label])
                    #with open(models_dir+arr_to_string(feat_combo)+'_betterFairness_gamma'+str(gamma)+modelname+dataset+'split'+str(i)+'.pickle','wb') as handle:
                        #pickle.dump(model, handle, pickle.HIGHEST_PROTOCOL)
                    dataset_saveData['split'+str(i)]['models'][arr_to_string(feat_combo)+'_betterFairness_gamma'+str(gamma)+modelname] = model
                    arr_to_string(feat_combo)+'_'+modelname
                    print('test ', dataset, modelname, 'better', arr_to_string(feat_combo), roc_auc_score(testY, model.predict_proba(fair_testX)[:,1]))
                    print('train', dataset, modelname, 'better', arr_to_string(feat_combo), roc_auc_score(trainY, model.predict_proba(fair_trainX)[:,1]))
    with open(models_dir+dataset+'_trainedModelInfo.pickle', 'wb') as handle:
        pickle.dump(dataset_saveData, handle, pickle.HIGHEST_PROTOCOL)
                
            
            
            
        
        
    