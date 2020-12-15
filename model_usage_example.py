import pickle as pkl
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score
import numpy as np

if __name__=='__main__':
    d = pkl.load(open('Models/adult_trainedModelInfo(1).pickle', 'rb'))

    # iterate over each of the 5 data splits
    for split in d.keys():
        print(split)

        # pull data and models
        # columns represent the original columns of the data frame (after one-hot-encoding)
        X, y, columns = d[split]['test_data']
        clfs = d[split]['models']

        # test model function
        for clf_name, clf in clfs.items():
            pred, pred_p = clf.predict(X), clf.predict_proba(X)[:,1]

            bla = balanced_accuracy_score(y, pred)
            acc = accuracy_score(y, pred)
            auc = roc_auc_score(y, pred_p)

            print('   ', clf_name, 'bla: ', np.round(bla, 2), 'acc: ', np.round(acc, 2), 'auc: ', np.round(auc, 2))







