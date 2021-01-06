import numpy as np
import pandas as pd
import pickle as pkl
import optimal_decision_making.manipulation as manip
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
import time as time
from prep.prep import prep

def false_positive_rate(pred, y_true):
	n = len(y_true)
	f_poses  = sum(1 for i in range(n) if pred[i]==1 and y_true[i]==0)
	tot_negs = sum(1 for i in range(n) if y_true[i]==0)
	return f_poses/float(tot_negs)

def false_negative_rate(pred, y_true):
	n = len(y_true)
	f_neg   = sum(1 for i in range(n) if pred[i] == 0 and y_true[i] == 1)
	tot_pos = sum(1 for i in range(n) if y_true[i] == 1)
	return f_neg/float(tot_pos)

def compute_metric_across_groups(y_true, pred, g0_index, g1_index, metric):
	return metric(pred[g0_index], y_true[g0_index]), metric(pred[g1_index], y_true[g1_index])

def compute_roc_across_groups(y_true, pred, g0_index, g1_index, metric):
	return metric(y_true[g0_index], pred[g0_index]), metric(y_true[g1_index],pred[g1_index])


if __name__=='__main__':

	info = [
			(
			'data/adult.csv', 'income', ['native-country'], {'race': (' White', ' Black'), 'sex': (' Male', ' Female')},
			['race'], 0),
			('data/Data_1980.csv', 'RECID', ['TIME', 'FILE'], {}, ['WHITE'], 1),
			('data/lawschool2.csv', 'bar1', ['cluster'], {}, ['race'], 0),
			('data/student-mat.csv', 'G3', [],
			 {'sex': ('M', 'F'), 'school': ('GP', 'MS'), 'address': ('U', 'R'), 'famsize': ('GT3', 'LE3'),
			  'Pstatus': ('A', 'T'), 'schoolsup': ('yes', 'no'), 'famsup': ('yes', 'no'), 'paid': ('yes', 'no'),
			  'activities': ('yes', 'no'), 'nursery': ('yes', 'no'), 'higher': ('yes', 'no'), 'internet': ('yes', 'no'),
			  'romantic': ('yes', 'no')}, ['sex'], 0)]

	f_save_names = ['adult', 'recidivism', 'lawschool', 'student']
	Xsandys= [prep(file_name, target_column, cols_to_remove, flip_0_and_1_labes, bin_vals=variables_to_be_made_binary) for file_name, target_column, cols_to_remove, variables_to_be_made_binary, sensative_features, flip_0_and_1_labes in info]


	# file name for pre-trained models
	f_names        = ['Outputs/adult.pickle',
					  'Outputs/recidivism.pickle',
					  'Outputs/lawschool.pickle',
					  'Outputs/student.pickle']
	# protected attributes for each of the files
	sense_feats    = ['race',
					  'WHITE',
					  'race',
					  'sex']
	# Columns which agents are able to lie abour
	#     - each row corresponds to a file in f_names
	manip_cols_all = [['race', 'sex', 'workclass', 'marital-status'],
					  ['WHITE', 'ALCHY', 'JUNKY', 'MARRIED', 'MALE'],
					  ['gender', 'race', 'fulltime', 'fam_inc'],
					  ['sex', 'freetime', 'studytime', 'goout', 'Fedu']]

	#manip_cols_all = [[feat] for feat in sense_feats]
	# Scalar for cost of lying
	alphas = [0.1, 0.2, 0.1, 0.1]
	max_iters = 200
	betas = [1, 2, 3, 4, 5]
	# Saves the false positive rates to a pickle file in Outputs
	#    - inner dict will be indexed by classifier names
	results_saved = {f_name: {} for f_name in f_names}
	#########################################
	# Main experiment loop
	# Iterates over each file
	# Computes optimal lies for each agent, for each classifier from the files
	#      - cost of lying  is f(x_old, x_new), for now f(x_old, x_new) = alpha ||x_old - x_new||
	#      - optimal lies are computed for several different values of alpha
	#      - our results should describe how unfair equilibrium are as a function of alpha
	#      - large alpha means no one lies, small alpha means everyone can lie, we care about "medium" values of alpha
	j = "agreed"
	outputs = []
	for f_index, f_name, sense_feat, manip_cols, alpha in zip(range(len(f_names)), f_names, sense_feats, manip_cols_all, alphas):
		d = pkl.load(open(f_name, 'rb'))

		# dict to hold the average scores of
		#avg_scores = {name: [0, 0, 0] for name in d[0]['models'].keys()}

		# Perform experiment for each alpha

		# Each data set has two sensitive groups since each protected feature is binary
		# dict to hold the false positive rates for each group, on each clf
		#    - true => true featuers, opti => optimal lie given clf and alpha
		avg_epsilon = {(name, beta): 0 for name in d[0]['models'].keys() for beta in betas}


		# Experiments are averages over each of the 5 splits from our files

		for ss, split in enumerate(d):
			X_test, y_test = split['data']
			y_test = y_test.to_numpy()

			X_train = pd.concat([Xsandys[f_index][0], X_test, X_test]).drop_duplicates(keep=False)



			y_train = Xsandys[f_index][1].loc[X_train.index].to_numpy()
			clfs = split['models']
			clfs = {'DT_0.01':clfs['DT_0.01'], 'LR_0.01':clfs['LR_0.01']}

			for beta in betas:
				ps = [[0 for _ in range(len(manip_cols))] for clf in clfs]
				costs = None
				for itr in range(max_iters):
					t = time.time()
					opt_lies_all_clfs, lie_count_by_index, best_costs_per_clf = \
						manip.optimal_agent_strats_for_cata_features_with_audit(X_test, y_test, manip_cols, clfs.values(), ps, alpha=alpha, decision_type='preds')
					print('output', ss, itr, beta, ':', round(time.time() - t, 2), f_name, alpha)

					print("#################################################################################")

					for l, lie_count in enumerate(lie_count_by_index):
						tot = float(sum(lie_count))
						for i in range(len(lie_count)):
							ps[l][i] = min(1, beta* lie_count[i]/tot)
						left = beta - sum(ps[l])
						while left > 0.000001:
							non1_index = [j for j in range(len(ps[l])) if ps[l][j] < 1]
							for j in non1_index:
								ps[l][j] = min(1.0, ps[l][j] + 1/float(len(non1_index)))
							left = beta - sum(ps[l])
							if sum(ps[l]) == len(ps[1]):
								break
					#for cost in best_costs_per_clf:
					#	print(cost)
					#	print('max_cost', max(cost))#, 'min_cost', min([cc for cc in cost if cc > 0]))

						#print(beta, ps[l])
						# if sum(ps[l]) < beta and any (pp != 1 for pp in ps[l]):
						# 	left = beta - sum(ps[l])
						# 	non1_index = [j for j in range(len(ps[l])) if ps[l][j] != 1]
						# 	for index in non1_index:
						# 		ps_old = ps[l][index]
						# 		ps[l][index] = min(1, ps[l][index] + left)
						# 		left -= ps[l][index] - ps_old
						# 		if left <= 0:
						# 			break
					# for pp in ps:
					# 	print('clf1', pp)
					# 	print('2')

					costs = best_costs_per_clf
				for clf_index, clf in enumerate(clfs):
					non_zero_costs = [cc for cc in costs[clf_index] if cc > 0]
					#print(non_zero_costs)
					if len(non_zero_costs) == 0:
						continue
					else:
						avg_epsilon[(clf, beta)] += (1 - min(non_zero_costs))/float(len(d))
		outputs.append(avg_epsilon)





		# Save final results
		with open('Outputs/false_postive_results_audting' + str(j) + '.pickle', 'wb') as handle:
			pkl.dump(outputs, handle, pkl.HIGHEST_PROTOCOL)
