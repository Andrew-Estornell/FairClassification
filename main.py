from fair import model
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import roc_auc_score

def fair_clf(sense_feats=None, reg=LinearRegression(), fairness='FP', C=10, gamma=0.01, max_iters=20, verbose=False):
    if sense_feats == None:
        print("Fair model requires sensitive features.")
        exit(-1)
    return model.Model(sense_feats, C=C, printflag=verbose, max_iters=max_iters, gamma=0.01, fairness_def=fairness, predictor=reg)



if __name__ == "__main__":


    ################ EXAMPLE CODE ####################

    # Process data
    df = pd.read_csv('data/Data_1980.csv')
    sense_feats = ['WHITE']
    df.drop(['TIME', 'FILE'], axis=1, inplace=True)
    y = df['RECID']
    df.drop('RECID', axis=1, inplace=True)
    X = df

    # Learn fair classifier
    fclf = fair_clf(sense_feats=sense_feats, verbose=True)
    fclf.fit(X, y)

    # Check training performance
    pred_p = fclf.predict_proba(X)[:,1]
    print("train AUC:", roc_auc_score(y, pred_p))



