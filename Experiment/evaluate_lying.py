# -*- coding: utf-8 -*-
'''
    To Joe, something like this, thanks: 
    # Learn fair classifier
    fclf = fair_clf(sense_feats=sense_feats, verbose=True)
    fclf.fit(X, y)

    # Check training performance
    pred_p = fclf.predict_proba(X)[:,1]
        '''
        
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import pickle

models_dict = pickle.load(open('FairClassification/Experiment/GBC_models.pickle','rb'))

df = pd.read_csv('FairClassification/Experiment/processed_data_with_validation_key.csv')
valid_df = df[df.isValidation].copy()
discrete_cols = ['workclass','education','marital-status','occupation','relationship','race','native-country']
valid_df.drop(discrete_cols+['income','isValidation'],axis=1, inplace=True)

fair_by_race_model = models_dict['race'].best_estimator_

for index, row in valid_df.iterrows():
    #each row is a person in validation
    print(fair_by_race_model.predict_proba(row.values.reshape(1,-1)))

    
    ''' #make changes ie
    row['occupation_is_ Adm-clerical'] = 0
    row['occupation_is_ Exec-managerial'] = 1
    '''
    print(fair_by_race_model.predict_proba(row.values.reshape(1,-1)))
    #compare value after lie