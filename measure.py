import sys
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import roc_auc_score
from fair import model
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
import pickle
from numpy import array
from numpy.linalg import norm


def fair_clf(sense_feats=None, reg=GBR(), fairness='FP', C=10, gamma=0.01, max_iters=20, verbose=False):
    if sense_feats == None:
        print("Fair model requires sensitive features.")
        exit(-1)
    return model.Model(sense_feats, C=C, printflag=verbose, max_iters=max_iters, gamma=gamma, fairness_def=fairness, predictor=reg)


def prep(file_name, target_col, remove_cols, rev, bin_vals=None):
    df = pd.read_csv(file_name)
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)

    #print(df)
    if len(remove_cols) > 0:
        df.drop(remove_cols, axis=1, inplace=True)
    for col in bin_vals.keys():
        #print(bin_vals[col])
        #print(df[col].unique())
        df = df[(df[col] == bin_vals[col][0]) | (df[col] == bin_vals[col][1])]
        df[col] = pd.Series(np.array([1 if df[col][i]== bin_vals[col][0] else 0 for i in df.index]), index=df.index)



    target = df[target_col]
    if rev:
        target = 1 + (-1*df[target_col])
    df.drop(target_col, inplace=True, axis=1)

    cata_cols, cont_cols = [], []
    cata_thresh = 7
    for col in df.columns:
        if str in [type(elem) for elem in df[col].unique()]:
            cata_cols.append(col)
        elif 2 < df[col].nunique() < cata_thresh:
            cata_cols.append(col)
        elif df[col].nunique() >= cata_thresh:
            cont_cols.append(col)
    df = pd.get_dummies(df, columns=cata_cols)
    for col in cont_cols:
        df[col] = (df[col] - df[col].min()) / float(df[col].max() - df[col].min())


    return df, target

info    = [('data/Data_1980.csv',            'RECID',               ['TIME','FILE'], {},                                       ['WHITE'],       1),
		 ('data/adult.csv',                            'income',              [],              {'race': (' White', ' Black')},           ['race'],        0),
		 ('data/communities.csv',                      'ViolentCrimesPerPop', [],              {},                                       ['race'],        1),
		 ('data/lawschool.csv',                        'bar1',                [],              {},                                       ['race'],        0),
		 ('data/student-mat.csv',                      'G3',                  [],              {},                                       ['sex'],         0)]

f_save_names = ['recidivism',
				'adult',
				'communities',
				'lawschool',
				'student']


for file_name, target_column, cols_to_remove, variables_to_be_made_binary, sensative_features, flip_0_and_1_labes in info:
    X, y = prep(file_name,  target_column, cols_to_remove, flip_0_and_1_labes, bin_vals=variables_to_be_made_binary)
    # print(X,y)

    if file_name == 'data/adult.csv':
        # Learn fair classifier
        sense_feats = ['race']
        fclf = fair_clf(sense_feats=sense_feats, verbose=True, gamma=0.0000001, max_iters=20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
        fclf.fit(X_train, y_train)


        # Check training performance
        pred_p = fclf.predict_proba(X_test)[:,1]

        # how to get the probability of say, row 37 get assigned label 1 ? #
        print("pred_p :", fclf.predict_proba(X_test)[:,1])
        print(len(fclf.predict_proba(X_test)[:,1]))
        print("train AUC:", roc_auc_score(y_test, pred_p))


        # measure the utility gain of lying for 1 agent. So, each of the rows/agents will get a new probability
        # after lying. What I do is to get an avg-ed probability gain or loss across the column
        # 4 different lies (sex, race ....) generate different combinations of dataframes to be fed into it
        features_to_lie_about = ['race']
        for feature in features_to_lie_about:
            X[feature].replace({1: 0, 0: 1}, inplace=True)
            fclf = fair_clf(sense_feats=sense_feats, verbose=True, gamma=0.0000001, max_iters=20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            fclf.fit(X_train, y_train)

            pred_p_lied = fclf.predict_proba(X_test)[:, 1]

            print("pred_p lied about : " + feature, fclf.predict_proba(X_test)[:, 1])
            print("train AUC lied about : " + feature, roc_auc_score(y_test, pred_p))
            vector_diff = pred_p - pred_p_lied
            print("avg utility (defined as probability to get 1) difference ", vector_diff.mean())
            l1_original = norm(pred_p, 1)
            l1_lied = norm(pred_p_lied,1)
            print("l1 norm difference:", l1_original-l1_lied)









