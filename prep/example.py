import pandas as pd
import numpy as np

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

info    = [('data/recidivism/Data_1980.csv',            'RECID',               ['TIME','FILE'], {},                                       ['WHITE'],       1),
		 ('data/adult.csv',                            'income',              [],              {'race': (' White', ' Black')},           ['race'],        0),
		 ('data/communities2.csv',                      'ViolentCrimesPerPop', [],              {},                                       ['race'],        1),
		 ('data/lawschool2.csv',                        'bar1',                [],              {},                                       ['race'],        0),
		 ('data/student-mat.csv',                      'G3',                  [],              {},                                       ['sex'],         0)]

f_save_names = ['recidivism',
				'adult',
				'communities',
				'lawschool',
				'student']


for file_name, target_column, cols_to_remove, variables_to_be_made_binary, sensative_features, flip_0_and_1_labes in info:
    X, y = prep(file_name,  target_column, cols_to_remove, flip_0_and_1_labes, bin_vals=variables_to_be_made_binary)