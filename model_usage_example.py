import pickle5 as pkl
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score
import numpy as np
import pandas
if __name__=='__main__':
    f_names = ['Models/adult.pickle', 'Models/recidivism.pickle', 'Models/lawschool.pickle', 'Models/student.pickle']
    for f_name in f_names:
        d = pkl.load(open(f_name, 'rb'))
        avg_scores = {name: [0, 0, 0] for name in d[0]['models'].keys()}


        ###################################
        # ITERATE OVER EACH OF THE 5 SPLITS
        ###################################
        for split in d:

            #####################################
            # PULL THE SAVED MODES AND TEST DATA
            # X is a data frame, y is a series
            #####################################
            X, y = split['data']
            clfs = split['models']


            ####################
            # TESTING EACH MODEL
            ####################
            for clf_name, clf in clfs.items():
                pred, pred_p = clf.predict(X), clf.predict_proba(X)[:,1]

                auc = roc_auc_score(y, pred_p)
                bla = balanced_accuracy_score(y, pred)
                acc = accuracy_score(y, pred)

                avg_scores[clf_name][0] += auc
                avg_scores[clf_name][1] += bla
                avg_scores[clf_name][2] += acc

        print(f_name)
        for clf_name, score in avg_scores.items():
            print('   ', clf_name, 'auc: ', np.round(score[0]/float(len(d)), 2), 'bla: ', np.round(score[1]/float(len(d)), 2), 'acc: ', np.round(score[2]/float(len(d)), 2))
        print()







