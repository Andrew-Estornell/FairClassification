# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle as pkl
import optimal_decision_making.manipulation as manip
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score

def false_positive_rate(pred, y_true):
	n = len(y_true)
	f_poses = sum(1 for i in range(n) if pred[i]==1 and y_true[i]==0)
	tot_negs = sum(1 for i in range(n) if y_true[i]==0)
	return f_poses/float(tot_negs)

def compute_metric_across_groups(y_true, pred, g0_index, g1_index, metric):
	return metric(pred[g0_index], y_true[g0_index]), metric(pred[g1_index], y_true[g1_index])

def adv_retraining(X, y, clf, max_iters, manip_cols, alpha):
    clf.fit(X, y)
    for j in range(max_iters):
        [all_new_X], opt_pred_ps_all_clfs, opt_preds_all_clfs = manip.optimal_agent_strats_for_cata_features(X, y, manip_cols, np.array([clf]), alpha=alpha, decision_type='preds')
        new_X=[]
        for row in all_new_X:
            new_X.append(row[0])
        new_X = pd.DataFrame(new_X, columns=X.columns)
        clf.fit(new_X, y)
    return clf

if __name__=='__main__':
    # file name for pre-trained models
    f_names        = ['../Outputs/adult.pickle', '../Outputs/recidivism.pickle', '../Outputs/lawschool.pickle', '../Outputs/student.pickle']
    
   	# protected attributes for each of the files
    sense_feats    = ['race', 'WHITE', 'race', 'sex']
   	# Columns which agents are able to lie abour
   	#     - each row corresponds to a file in f_names
    manip_cols_all = [['race', 'sex', 'workclass'], #'marital-status', 'occupation', 'relationship'],
   				      ['WHITE', 'ALCHY', 'JUNKY'], #'MARRIED', 'MALE'],
   				      ['gender', 'race'],
   				      ['sex', 'freetime', 'studytime', 'famrel']]#, 'goout', 'health']]
    #scalar for cost of lying
    alphas = [1, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
    alphas = [1,0.1,0.01,0.001]

   	#########################################
    for f_name, sense_feat, manip_cols in zip(f_names, sense_feats, manip_cols_all):
        d = pkl.load(open(f_name, 'rb'))
        d_retrained = []
        for i in range(len(d)):
            split = d[i]
            retrained_split = {'data':split['data'], 'alpha_clfs':{}}
            X, y = split['data']
            #y = y.to_numpy()
            clfs = split['models']
            for alpha in alphas:
                if alpha not in retrained_split['alpha_clfs'].keys():
                    retrained_split['alpha_clfs'][alpha] = {}
                for clfname in split['models'].keys():
                    clf = split['models'][clfname]
                    alpha_optimized_clf = adv_retraining(X,y, clf, 5, manip_cols, alpha)
                    retrained_split['alpha_clfs'][alpha][clfname]=alpha_optimized_clf
            d_retrained.append(retrained_split)
        
        with open(f_name[:-7]+'_manip.pickle', 'wb') as handle:
           pkl.dump(d_retrained, handle, pkl.HIGHEST_PROTOCOL)

