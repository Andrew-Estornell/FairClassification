import pickle as pkl
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score
import numpy as np
import pandas as pd
from numpy.linalg import norm
from pandas.testing import assert_frame_equal
import statistics,itertools

def measure_fairness(clf,X,original_pred,original_pred_p,alpha,features_to_lie):
    float_types = ['age','fnlwgt','eduction-num','capital-gain','capital-loss','hours-per-week']
    basic_categorical_types = ['workclass','education','marital-status','occupation','relationship']
    flip_types = ['race','sex']

    pos_indices = [i for i, x in enumerate(original_pred) if x == 1]
    neg_indices = [i for i, x in enumerate(original_pred) if x != 1]

    mean_improvement_dict = {}

    combination_list = []

    for feature in features_to_lie:
        matching = [s for s in X.columns if feature in s]
        combination_list.append(matching)

    i = 0
    for tuple in itertools.product(*combination_list):
        print(list(itertools.product(*combination_list)))
        print(float(i)/len(list(itertools.product(*combination_list))))
        i += 1
        l1_norm_diff_dict = {}
        improvement_list = []
        no_changed_on_list = []
        valid_indices = []
        print('Lying about column: ' + "_".join(tuple))
        X_copy = X.copy(deep=True)
        for neg_index in neg_indices:
            old_row = X_copy.iloc[neg_index].copy(deep=True)
            the_same_flag = False
            for ele in tuple:
                if ele in flip_types:
                    original_value = X_copy.iloc[neg_index].at[ele]
                    X_copy.iloc[neg_index, X_copy.columns.get_loc(ele)] = int(abs(1 - original_value))
                else:
                    for basic_type in basic_categorical_types:
                        if basic_type in ele:
                            matching = [s for s in X_copy.columns if basic_type in s]
                            X_copy.iloc[neg_index, X_copy.columns.get_loc(ele)] = 1
                            for match in matching:
                                if match != ele:
                                    X_copy.iloc[neg_index, X_copy.columns.get_loc(match)] = 0
                    if X_copy.iloc[neg_index].at[ele] == X.iloc[neg_index].at[ele]:
                        the_same_flag = True
            if not the_same_flag:
                valid_indices.append(neg_index)
            the_same_flag = False

            lied_row = X_copy.iloc[neg_index].copy(deep=True)

            l1_original = norm(old_row, 1)
            l1_lied = norm(lied_row, 1)
            l1_norm_diff = l1_original - l1_lied

            l1_norm_diff_dict[neg_index] = l1_norm_diff

            pred_lied, pred_p_lied = clf.predict(X_copy), clf.predict_proba(X_copy)[:, 1]
            vector_diff = pred_lied - original_pred
            mean_vector_diff = vector_diff.mean()
        #print(valid_indices)
        for neg_index in valid_indices:
            # if original_pred[neg_index] > pred_lied[neg_index]:
            #     print(neg_index)
            improvement = (pred_lied[neg_index] - original_pred[neg_index]) - alpha * l1_norm_diff_dict[neg_index]
            improvement_list.append(improvement)
        #print(improvement_list)
        np_improvement_list = np.array(improvement_list)
        mean_improvement_dict['_'.join(tuple)] = np_improvement_list.mean()
        #print(mean_improvement_dict)



if __name__=='__main__':
    f_names = ['Outputs/adult.pickle', 'Models/recidivism.pickle', 'Models/lawschool.pickle', 'Models/student.pickle']
    f_names_small = ['Outputs/adult.pickle']
    for f_name in f_names_small:
        d = pkl.load(open(f_name, 'rb'))
        avg_scores = {name: [0, 0, 0] for name in d[0]['models'].keys()}
        count_for_printing = 0

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
            print(X.dtypes)
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

                measure_fairness(clf,X,pred,pred_p,0,['race','education','relationship'])




        print(f_name)
        for clf_name, score in avg_scores.items():
            print('   ', clf_name, 'auc: ', np.round(score[0]/float(len(d)), 2), 'bla: ', np.round(score[1]/float(len(d)), 2), 'acc: ', np.round(score[2]/float(len(d)), 2))
        print()