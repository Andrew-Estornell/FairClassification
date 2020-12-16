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
    weights = np.array([1.0/g1_size if data[feat][i] == 1 else 1.0/g2_size for i in range(len(data))])
    return weights


if __name__ == '__main__':
    np.random.seed(101)

    info = [('data/Data_1980.csv',   'RECID',  ['TIME','FILE'], {},                             ['WHITE'], 1),
            ('data/adult.csv',       'income', [],              {'race': (' White', ' Black')}, ['race'],  0),
            ('data/lawschool2.csv',  'bar1',   ['cluster'],     {},                             ['race'],  0),
            ('data/student-mat.csv', 'G3',     [],              {},                             ['sex'],   0)]

    f_save_names = ['recidivism', 'adult', 'lawschool', 'student']

    scoring = make_scorer(roc_auc_score, needs_proba=True)
    n_jobs = -1
    gammas = [0.2, 0.1, 0.01]
    k = 5

    base_params = {'GB': {'max_depth':    [2, 5, 10],
                          'n_estimators': [50*i + 50 for i in range(10)]},
                   'SV': {'kernel': ['linear', 'poly', 'rbf'],
                          'C':      [10, 1, 0.1, 0.01]},
                   'LG': {'max_iters': [200, 500, 100],
                          'C':         [10, 1, 0.1, 0.01]}}

    reg_params = {'LR': {'fit_intercept': [True]},
                  'DT': {'max_depth': [2*i for i in range(1, 7)],
                         'max_features': [1, 2, 5, 10]}}
    ctr = 0
    for file_name, target_column, cols_to_remove, variables_to_be_made_binary, sensative_features, flip_0_and_1_labes in info:
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
        fair_lrgs = {'LR_' + str(gamma): GridSearchCV(FModel(sensative_features, predictor=LinearRegression(),      gamma=gamma, max_iters=200), param_grid=gen_param_grid(reg_params['LR']), n_jobs=n_jobs, scoring=scoring) for gamma in gammas}
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
                
                if 'equal' in clf_name:
                    clf.fit(X_train, y_train, sample_weights=gen_sample_weights(X, sensative_features[0]))
                else:
                    clf.fit(X_train, y_train)

                pred_p = clf.predict_proba(X_test)
                print(roc_auc_score(y_test, pred_p))
                # SAVE EACH MODEL AFTER TRAINING
                outputs[-1]['models'][clf_name] = clf

        with open('Models/' + f_save_names[ctr] + '.pickle', 'wb') as handle:
            pkl.dump(outputs, handle, pkl.HIGHEST_PROTOCOL)











