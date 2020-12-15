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

option_values = {
  "race":['White','Black'],
  "sex": ['Male','Female'],
  "both": ['White','Black','Male','Female'],
  "none": ['White','Black','Male','Female'],
}

def not_lie_and_lie_test_joey(options):
    valid_df = df[df.isValidation].copy()
    discrete_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'option',
                     'native-country']
    valid_df.drop(discrete_cols + ['income', 'isValidation'], axis=1, inplace=True)

    joey_fair_option_proba_list = []
    joey_lied_fair_option_proba_list = []
    for index, row in valid_df.iterrows():
        for option in options:
            this_row_proba = models_dict[option].best_estimator_.predict_proba(row.values.reshape(1, -1))[:, 1]
            joey_fair_option_proba_list.append(this_row_proba)
            option_cols = [col for col in valid_df.columns if option in col]
            for col in option_cols:
                if row[col] == 1:
                    row[col] = 0
                    col_other = ''
                    for col1 in option_cols:
                        if col != col1 and ('White' in col1 or 'Black' in col1):
                            col_other = col1
                            row[col_other] = 1
                    this_row_lied_proba = models_dict['option'].best_estimator_.predict_proba(
                        row.values.reshape(1, -1))[:, 1]
                    joey_lied_fair_option_proba_list.append(this_row_lied_proba)
                    row[col] = 1
                    row[col_other] = 0
        vector_diff = arr = np.array(joey_fair_option_proba_list) - np.array(joey_lied_fair_option_proba_list)
        print("avg utility (defined as probability to get 1) difference ", vector_diff.mean())
        l1_original = norm(np.array(joey_fair_option_proba_list), 1)
        l1_lied = norm(np.array(joey_lied_fair_option_proba_list), 1)
        print("l1 norm difference:", l1_original - l1_lied)

def not_lie_and_lie_test_andrew(features):
    pass


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

models_dict = pickle.load(open('Experiment/GBC_models.pickle','rb'))

df = pd.read_csv('Experiment/processed_data_with_validation_key.csv')

print(models_dict.keys())


for file_name, target_column, cols_to_remove, variables_to_be_made_binary, sensative_features, flip_0_and_1_labes in info:
    if file_name == 'data/adult.csv':
        features = ['race','sex','both','none']
        for feature in features:
            if feature == 'race':
                # Joey fair race model
                valid_df = df[df.isValidation].copy()
                discrete_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                                 'native-country']
                valid_df.drop(discrete_cols + ['income', 'isValidation'], axis=1, inplace=True)
                
                joey_fair_race_proba_list = []
                joey_lied_fair_race_proba_list = []
                for index, row in valid_df.iterrows():
                    this_row_proba = models_dict['race'].best_estimator_.predict_proba(row.values.reshape(1, -1))[:, 1]
                    joey_fair_race_proba_list.append(this_row_proba)
                    race_cols = [col for col in valid_df.columns if 'race' in col]
                    for col in race_cols:
                       if row[col] == 1:
                           row[col] = 0
                           col_other = ''
                           for col1 in race_cols:
                               if col != col1 and ('White' in col1 or 'Black' in col1):
                                   col_other = col1
                                   row[col_other] = 1
                           this_row_lied_proba = models_dict['race'].best_estimator_.predict_proba(row.values.reshape(1, -1))[:, 1]
                           joey_lied_fair_race_proba_list.append(this_row_lied_proba)
                           row[col] = 1
                           row[col_other] = 0
                           break
                vector_diff = arr = np.array(joey_fair_race_proba_list) - np.array(joey_lied_fair_race_proba_list)
                print("avg utility (defined as probability to get 1) difference ", vector_diff.mean())
                l1_original = norm(np.array(joey_fair_race_proba_list), 1)
                l1_lied = norm(np.array(joey_lied_fair_race_proba_list), 1)
                print("l1 norm difference:", l1_original - l1_lied)


                # Andrew fair race model
                X, y = prep(file_name, target_column, cols_to_remove, flip_0_and_1_labes,
                            bin_vals=variables_to_be_made_binary)
                sense_feats = ['race']
                fclf = fair_clf(sense_feats=sense_feats, verbose=True, gamma=0.0000001, max_iters=20)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                fclf.fit(X_train, y_train)

                pred_p = fclf.predict_proba(X_test)[:, 1]

                X[feature].replace({1: 0, 0: 1}, inplace=True)
                fclf = fair_clf(sense_feats=sense_feats, verbose=True, gamma=0.0000001, max_iters=20)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                fclf.fit(X_train, y_train)

                pred_p_lied = fclf.predict_proba(X_test)[:, 1]

                # print("train AUC lied about : " + feature, roc_auc_score(y_test, pred_p))
                vector_diff = pred_p - pred_p_lied
                print("avg utility (defined as probability to get 1) difference ", vector_diff.mean())
                l1_original = norm(pred_p, 1)
                l1_lied = norm(pred_p_lied, 1)
                print("l1 norm difference:", l1_original - l1_lied)

                pass
            elif feature == 'sex':
                # Joey fair sex model

                # Andrew fair sex model
                pass
            elif feature == 'both':
                # Joey fair both sex and race model

                # Andrew fair both sex and race model
                pass
            elif feature == 'none':
                # Joey base model

                pass













