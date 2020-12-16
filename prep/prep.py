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
        df[col] = pd.Series(np.array([1 if df[col][i] == bin_vals[col][0] else 0 for i in df.index]), index=df.index)



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

