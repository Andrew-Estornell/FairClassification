# -*- coding: utf-8 -*-
import pandas as pd
df = pd.read_csv('adult.csv')

discrete_cols = ['workclass','education','marital-status','occupation','relationship','race','native-country']
for col in discrete_cols:
    for val in df[col].unique():
        df[col+'_is_'+val] = (df[col]==val).astype(int)

#df.drop(discrete_cols, axis=1, inplace=True)

df.at[df.sex==' Male','sex'] = 1
df.at[df.sex==' Female','sex'] = 0



df.to_csv('processed_adults.csv',index=False)