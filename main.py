from fair import model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.tree import DecisionTreeRegressor as DTR
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import itertools as itr
from sklearn.metrics import balanced_accuracy_score, make_scorer
import numpy as np

#C=10
#max_iter=100 (min 50)
#Gamma \in [0.2, 0.1, 0.01]
#5 gammas
'''
C=10
max_iter=100
Gamma \in [0.2, 0.01]
5 gammas
0.2, 0.1, 0.01
Min for max_iter = 50
'''


def fair_clf(sense_feats=None, reg=LinearRegression(), fairness='FN', C=10, gamma=0.01, max_iters=50, verbose=False):
    if sense_feats == None:
        print("Fair model requires sensitive features.")
        exit(-1)
    return model.Model(sense_feats, C=C, printflag=verbose, max_iters=max_iters, gamma=gamma, fairness_def=fairness, predictor=reg)

def gen_param_grid(param_grid):
    keys = list(param_grid.keys())
    combinations = [{keys[i]: combo[i] for i in range(len(combo))} for combo in list(itr.product(*(param_grid[key] for key in keys)))]
    return {'predictor_param_dict': combinations}

if __name__ == "__main__":


    ################ EXAMPLE CODE ####################

    # Process data
    df = pd.read_csv('data/Data_1980.csv')
    sense_feats = ['WHITE', 'MALE', 'MARRIED']
    df.drop(['TIME', 'FILE'], axis=1, inplace=True)
    y = df['RECID']
    df.drop('RECID', axis=1, inplace=True)
    X = df

    # Learn fair classifier
    #reg = LinearRegression()#DTR(max_depth=5, max_features=2)
    reg = GBR()
    #GBC_params = {'max_depth':[3, 6, 10], 'n_estimators':[250,500]}
    
    fclf = fair_clf(sense_feats=sense_feats, reg=reg, verbose=True, max_iters=5000)

    #reg_params = gen_param_grid(GBC_params)
    #fclf = GridSearchCV(fclf, reg_params, n_jobs=-1, scoring=make_scorer(balanced_accuracy_score), verbose=1)
    #fclf = fair_clf(sense_feats=sense_feats, reg=LinearRegression(), verbose=True)
    fclf.fit(X, y)

    # Check training performance
    pred_p = fclf.predict_proba(X)[:,1]

    print(np.unique(pred_p, return_counts=True))
    print("train AUC:", roc_auc_score(y, pred_p))



