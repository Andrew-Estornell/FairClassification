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











    ctr = 0
    for file_name, target_column, cols_to_remove, variables_to_be_made_binary, sensative_features, flip_0_and_1_labes in info:
        print(file_name)
        outputs = []

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
        fair_lrgs = {'LR_' + str(gamma): FModel(sensative_features, predictor=LinearRegression(),                   gamma=gamma, max_iters=200)                                                                              for gamma in gammas}
        fair_dtrs = {'DT_' + str(gamma): GridSearchCV(FModel(sensative_features, predictor=DecisionTreeRegressor(), gamma=gamma, max_iters=200), param_grid=gen_param_grid(reg_params['DT']), n_jobs=n_jobs, scoring=scoring) for gamma in gammas}

        #############################
        # SPLIT DATA AND TRAIN MODELS
        #############################
        kf = KFold(n_splits=k)
        for train_index, test_index in kf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train_index], y.iloc[train_index], X.iloc[test_index], y.iloc[test_index]

            # SAVE DATA
            outputs.append({'data':   (X_test, y_test),
                            'models': {}})

            for clf_name, clf in list(base_clfs.items()) + list(equa_clfs.items()) + list(fair_lrgs.items()) + list(fair_dtrs.items()):
                print(clf_name, end=', ')
                if 'equal' in clf_name:
                    clf.fit(X_train, y_train, sample_weight=gen_sample_weights(X_train, sensative_features[0]))
                else:
                    clf.fit(X_train, y_train)

                pred_p = clf.predict_proba(X_test)[:, 1]
                print(roc_auc_score(y_test, pred_p))
                # SAVE EACH MODEL AFTER TRAINING
                outputs[-1]['models'][clf_name] = clf

        with open('Outputs/' + f_save_names[ctr] + '.pickle', 'wb') as handle:
            pkl.dump(outputs, handle, pkl.HIGHEST_PROTOCOL)
        ctr += 1











