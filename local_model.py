import pickle as pkl
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score
import numpy as np
import pandas
from numpy.linalg import norm
if __name__=='__main__':
    f_names = ['Models/adult.pickle', 'Models/recidivism.pickle', 'Models/lawschool.pickle', 'Models/student.pickle']
    f_names_small = ['Models/adult.pickle']
    for f_name in f_names_small:
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

                # lie on rows that have negative prediction labels
                pos_indices = [i for i, x in enumerate(pred) if x == 1]
                neg_indices = [i for i, x in enumerate(pred) if x != 1]

                # for categorical_col in X.select_dtypes(['uint8','int32']).columns:
                #     res = [i for i in test_list if subs in i]

                lie_columns = ['race','sex']
                for lied_col in lie_columns:
                    print('Lying about column: ' + lied_col)
                    for neg_index in neg_indices:
                        lied_row = X.iloc[neg_index]
                        original_value = lied_row[lied_col]
                        print(lied_row)
                        lied_row[lied_col] = abs(1-original_value) # perform a value flip
                        print(lied_row)
                        print('\n')

                    pred_lied, pred_p_lied = clf.predict(X), clf.predict_proba(X)[:, 1]

                    vector_diff = pred_lied - pred
                    print("avg utility (defined as probability to get 1) difference ", vector_diff.mean())
                    l1_original = norm(pred_p, 1)
                    l1_lied = norm(pred_p_lied, 1)
                    print("l1 norm difference:", l1_original - l1_lied)


                    # flip back
                    print('Flip back to original column: ' + lied_col)
                    for neg_index in neg_indices:
                        lied_row = X.iloc[neg_index]
                        old_value = lied_row[lied_col]
                        lied_row[lied_col] = abs(1-old_value) # perform a value flip

        print(f_name)
        for clf_name, score in avg_scores.items():
            print('   ', clf_name, 'auc: ', np.round(score[0]/float(len(d)), 2), 'bla: ', np.round(score[1]/float(len(d)), 2), 'acc: ', np.round(score[2]/float(len(d)), 2))
        print()

