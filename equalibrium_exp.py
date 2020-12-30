import numpy as np
import pandas as pd
import pickle as pkl
import optimal_decision_making.manipulation as manip
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score

def false_positive_rate(pred, y_true):
	n = len(y_true)
	f_poses = sum(1 for i in range(n) if pred[i]==1 and y_true[i]==0)
	tot_negs = sum(1 for i in range(n) if y_true[i]==0)
	return f_poses/float(tot_negs)

def compute_metric_across_groups(y_true, pred, g0_index, g1_index, metric):
	return metric(pred[g0_index], y_true[g0_index]), metric(pred[g1_index], y_true[g1_index])


if __name__=='__main__':

	# file name for pre-trained models
	f_names        = ['Outputs/adult.pickle', 'Outputs/recidivism.pickle', 'Outputs/lawschool.pickle', 'Outputs/student.pickle']
	# protected attributes for each of the files
	sense_feats    = ['race', 'WHITE', 'race', 'sex']
	# Columns which agents are able to lie abour
	#     - each row corresponds to a file in f_names
	manip_cols_all = [['race', 'sex', 'workclass'], #'marital-status', 'occupation', 'relationship'],
				      ['WHITE', 'ALCHY', 'JUNKY'], #'MARRIED', 'MALE'],
				      ['gender', 'race'],
				      ['sex', 'freetime', 'studytime', 'famrel']]#, 'goout', 'health']]
	# Scalar for cost of lying
	alphas = [1, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
	# Saves the false positive rates to a pickle file in Outputs
	#    - inner dict will be indexed by classifier names
	results_saved = {(f_name, alpha): {} for f_name in f_names for alpha in alphas}
	#########################################
	# Main experiment loop
	# Iterates over each file
	# Computes optimal lies for each agent, for each classifier from the files
	#      - cost of lying  is f(x_old, x_new), for now f(x_old, x_new) = alpha ||x_old - x_new||
	#      - optimal lies are computed for several different values of alpha
	#      - our results should describe how unfair equilibrium are as a function of alpha
	#      - large alpha means no one lies, small alpha means everyone can lie, we care about "medium" values of alpha
	for f_name, sense_feat, manip_cols in zip(f_names, sense_feats, manip_cols_all):
		d = pkl.load(open(f_name, 'rb'))

		# dict to hold the average scores of
		#avg_scores = {name: [0, 0, 0] for name in d[0]['models'].keys()}

		# Each data set has two sensitive groups since each protected feature is binary
		# dict to hold the false positive rates for each group, on each clf
		#    - true => true featuers, opti => optimal lie given clf and alpha
		avg_fp_true_g0, avg_fp_true_g1 = {name: 0 for name in d[0]['models'].keys()}, {name: 0 for name in d[0]['models'].keys()}
		avg_fp_opti_g0, avg_fp_opti_g1 = {name: 0 for name in d[0]['models'].keys()}, {name: 0 for name in d[0]['models'].keys()}

		# Perform experiment for each alpha
		for alpha in alphas:
			# Experiments are averages over each of the 5 splits from our files
			for split in d:
				X, y = split['data']
				y = y.to_numpy()
				clfs = split['models']

				# Indexes of each member of groups 0 and 1 respectively
				g0_index = [i for i in range(len(X)) if X[sense_feat].iloc[i] == 0]
				g1_index = [i for i in range(len(X)) if X[sense_feat].iloc[i] == 1]

				###############################################
				# Primary function call
				#    - computes the optimal lie for each agent in X for each clf, given alpha
				opt_lies_all_clfs, opt_pred_ps_all_clfs, opt_preds_all_clfs =\
					manip.optimal_agent_strats_for_cata_features(X, y, manip_cols, clfs.values(), alpha=alpha,
																 decision_type='preds')

				# Iterate over each of the clfs we just computed the optimal lies for
				for i, itm in enumerate(clfs.items()):
					clf_name, clf = itm

					# 0-1 predictions and probabilities (strategic and true) for agents, from current clf
					opt_preds, opt_pred_ps = opt_preds_all_clfs[i], opt_pred_ps_all_clfs[i]
					pred,      pred_p      = clf.predict(X), clf.predict_proba(X)[:,1]

					# False positive rates for each group both at equilibrium and under truthful submission
					g0_true_fp, g1_true_fp = compute_metric_across_groups(y, pred,      g0_index, g1_index, false_positive_rate)
					g0_opti_fp, g1_opti_fp = compute_metric_across_groups(y, opt_preds, g0_index, g1_index, false_positive_rate)

					# Adding FPs for current split to the average
					avg_fp_true_g0[clf_name] += g0_true_fp/float(len(d))
					avg_fp_opti_g0[clf_name] += g0_opti_fp/float(len(d))
					avg_fp_true_g1[clf_name] += g1_true_fp/float(len(d))
					avg_fp_opti_g1[clf_name] += g1_opti_fp/float(len(d))


			# For each file, displaying the false positive rates between groups from each of the clfs
			print('data', f_name)
			print("ALPHA", alpha)
			# Grabbing the false positive rate from each clf, of each group.
			for g0_true, g1_true, g0_opti, g1_opti in zip(avg_fp_true_g0.items(), avg_fp_true_g1.items(), avg_fp_opti_g0.items(), avg_fp_opti_g1.items()):
				clf_name = g0_true[0]
				g0_fp_true = g0_true[1]
				g1_fp_true = g1_true[1]
				g0_fp_opti = g0_opti[1]
				g1_fp_opti = g1_opti[1]

				# Save false positive as (true0, true1, opti0, opti1) tuple
				results_saved[(f_name, alpha)][clf_name] = (g0_fp_true, g1_fp_true, g0_fp_true, g1_fp_opti)

				print(clf_name, end=':: ')
				print("False positive diff g0 vs g1 || true:", round(abs(g0_fp_true - g1_fp_true), 2),
					                               '|| opti:', round(abs(g0_fp_opti - g1_fp_opti), 2))

			print()
	# Save final results
	with open('Outputs/false_postive_results.pickle', 'wb') as handle:
		pkl.dump(results_saved, handle, pkl.HIGHEST_PROTOCOL)

