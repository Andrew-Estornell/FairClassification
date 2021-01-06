import numpy as np
import itertools as itr
import copy as copy


def optimal_agent_strats_for_cata_features(X, y, manip_cols, clfs, alpha=0.1, f=lambda a1, a2: np.linalg.norm((a1 - a2), ord=1),
                                           decision_type='preds'):
    """
    Finds the optimal lie for each agent in X, for each of the clfs,

    :param manip_cols:     columns able to be lied about (these are given as columns prior to one-hot-encoding)
    :param clfs:           list of classifiers
    :param alpha:          scalar for cost function
    :param f:              cost function and agent must pay for submitting a2 when they originally had type a1
    :param decision_type:  {'perds', 'proba'} determines if agents care about labels or probabilities
    """

    d = len(X.columns)
    # Nested list where first dimension is each manipulable columns
    #                 - second dimension is each column which is associated with the manipulable column after encoding
    manip_col_indexes  = [[i for i, col in enumerate(X.columns) if prefix in col]
                          for prefix in manip_cols]
    # Index of all columns which cannot be lied about (after one-hot-encoding)
    static_col_indexes = [i for i in range(d) if not any(i in indexes for indexes in manip_col_indexes)]

    # Computes all possible lies that agents are able to tell
    # lies is a template, since each agent all have the same manipulable columns, we need only compute all possible lies
    # once. Agents can then choose from this set without the need to recompute.
    lies = compute_all_lies(manip_col_indexes, d)

    # Holds all lies for agents
    # Holds costs for each such lie
    all_strats = []
    costs      = []
    # Weave lies into each agents features
    for i, x in enumerate(X.to_numpy()):
        # Copy lies
        new_lies = 1*lies
        # Set each row in new_lies to be x, when the submit the lie in that row of new_lies
        new_lies[:, static_col_indexes] = x[static_col_indexes]
        # Save lies and costs
        all_strats.append(new_lies)
        costs.append([alpha*f(x, new_lie) for new_lie in new_lies])

    s = len(all_strats[0])

    # reshape costs and all_strats to be flat
    #    - previously each was grouped by the agent telling the lies
    all_strats = np.array([strat for strat_group in all_strats for strat in strat_group ])
    costs      = np.array([cost for cost_group in costs for cost in cost_group])

    # Holds best lies, and corresponding probability and labels of each lie for each clf
    best_strats = [[] for _ in clfs]
    best_probas = [[] for _ in clfs]
    best_labels = [[] for _ in clfs]

    # Compute optimal lie for each classifier
    for i, clf in enumerate(clfs):
        # True values
        true_pred_p = clf.predict_proba(X)[:,1]
        # optimal values computed for every agent and every possible lie (all strats is size len(lies)*len(*X))
        pred_ps_expanded = clf.predict_proba(all_strats)[:,1]
        pred_expanded = clf.predict(all_strats)

        # Compute the utility that each lie has, utility depends on decision_type, alpha, and f
        utility_expanded = None
        if   decision_type == 'probas':
            utility_expanded = pred_ps_expanded - np.repeat(true_pred_p, repeats=s, axis=0) - costs
        elif decision_type == 'preds':
            utility_expanded = pred_expanded - np.repeat(true_pred_p, repeats=s, axis=0) - costs

        # Each agent has len(lies) number of lies in all_strats
        #    - iterate over each chunk of size len(lies) and find the best utility lie in that chuck,
        #    - This is the best lie for that agent.
        for cut in range(len(utility_expanded)//s):
            # Index of best lie for that given agent
            best_index = np.argmax(utility_expanded[s*cut: s*(cut + 1)])

            # Uses index of best lie to grab the optimal probability, label, and lie
            best_probas[i].append(pred_ps_expanded[cut*s + best_index])
            best_labels[i].append(pred_expanded[cut*s + best_index])
            best_strats[i].append([all_strats[cut*s + best_index], utility_expanded[cut*s + best_index], costs[cut*s + best_index]])


    # Change to arrays for experiment code purposes
    best_probas = np.array(best_probas)
    best_labels = np.array(best_labels)
    return best_strats, best_probas, best_labels



def optimal_agent_strats_for_cata_features_2(X, y, manip_cols, clfs, alpha=0.1, f=lambda a1, a2: np.linalg.norm((a1 - a2), ord=1),
                                           decision_type='preds'):
    """
    Finds the optimal lie for each agent in X, for each of the clfs,

    :param manip_cols:     columns able to be lied about (these are given as columns prior to one-hot-encoding)
    :param clfs:           list of classifiers
    :param alpha:          scalar for cost function
    :param f:              cost function and agent must pay for submitting a2 when they originally had type a1
    :param decision_type:  {'perds', 'proba'} determines if agents care about labels or probabilities
    """

    d = len(X.columns)
    # Nested list where first dimension is each manipulable columns
    #                 - second dimension is each column which is associated with the manipulable column after encoding
    manip_col_indexes  = [[i for i, col in enumerate(X.columns) if prefix in col]
                          for prefix in manip_cols]
    # Index of all columns which cannot be lied about (after one-hot-encoding)
    static_col_indexes = [i for i in range(d) if not any(i in indexes for indexes in manip_col_indexes)]
    manip_col_indexes_falt = [i for i in range(d) if i not in static_col_indexes]
    X = X.to_numpy()

    # Computes all possible lies that agents are able to tell
    # lies is a template, since each agent all have the same manipulable columns, we need only compute all possible lies
    # once. Agents can then choose from this set without the need to recompute.
    lies = compute_all_lies(manip_col_indexes, d)

    lie_indexes_by_lowest_cost = []
    for x in X:
        sorted_costs_and_index = sorted([(l, alpha*f(x[manip_col_indexes_falt], lie[manip_col_indexes_falt]))
                                        for l, lie in enumerate(lies)], key=lambda xx: xx[1])
        sorted_indexes = [l for l, _ in sorted_costs_and_index]
        lie_indexes_by_lowest_cost.append(sorted_indexes)




    best_strats_for_each_clf = []
    for clf in clfs:
        optimal_strats = {i: None for i in range(len(X))}
        pred = clf.predict(X)
        for i, score in enumerate(pred):
            if score == 1:
                optimal_strats[i] = X[i]
        for l in range(len(lies)):
            indexes_left_to_search = [i for i in optimal_strats if optimal_strats[i] is None]
            if len(indexes_left_to_search) == 0:
                break
            X_search = copy.deepcopy(X[indexes_left_to_search])
            for col_index in manip_col_indexes_falt:
                X_search[:,col_index] = lies[l][col_index]
            costs = np.array([alpha*f(X_search[i], X[indexes_left_to_search[i]]) for i in range(len(indexes_left_to_search))])


            if decision_type == 'preds':
                pred = clf.predict(X_search)

                for i in range(len(pred)):
                    if costs[i] >= 1:
                        optimal_strats[indexes_left_to_search[i]] = X[indexes_left_to_search[i]]
                    elif pred[i] == 1:
                        optimal_strats[indexes_left_to_search[i]] = X_search[i]
        for i in optimal_strats:
            if optimal_strats[i] is None:
                optimal_strats[i] = X[i]
        best_strats_for_each_clf.append(list(optimal_strats.values()))
        # for ii, strat in enumerate(list(optimal_strats.values())):
        #     for xx_true, xx_opt in zip(X[ii], strat):
        #         if xx_true != xx_opt:
        #             print(X[ii], 'not equal', strat)
    return np.array(best_strats_for_each_clf), [], []#[clf.predict(np.array(best_strats_for_each_clf[i])) for i, clf in enumerate(clfs)], [clf.predict_proba(np.array(best_strats_for_each_clf[i])) for i, clf in enumerate(clfs)]



def optimal_agent_strats_for_cata_features_with_audit(X, y, manip_cols, clfs, ps, alpha=0.1, f=lambda a1, a2: np.linalg.norm((a1 - a2), ord=1),
                                           decision_type='preds'):
    """
    Finds the optimal lie for each agent in X, for each of the clfs,

    :param manip_cols:     columns able to be lied about (these are given as columns prior to one-hot-encoding)
    :param clfs:           list of classifiers
    :param alpha:          scalar for cost function
    :param f:              cost function and agent must pay for submitting a2 when they originally had type a1
    :param decision_type:  {'perds', 'proba'} determines if agents care about labels or probabilities
    """

    d = len(X.columns)
    # Nested list where first dimension is each manipulable columns
    #                 - second dimension is each column which is associated with the manipulable column after encoding
    manip_col_indexes  = [[i for i, col in enumerate(X.columns) if prefix in col]
                          for prefix in manip_cols]
    # Index of all columns which cannot be lied about (after one-hot-encoding)
    static_col_indexes = [i for i in range(d) if not any(i in indexes for indexes in manip_col_indexes)]
    manip_col_indexes_falt = [i for i in range(d) if i not in static_col_indexes]
    X = X.to_numpy()

    # Computes all possible lies that agents are able to tell
    # lies is a template, since each agent all have the same manipulable columns, we need only compute all possible lies
    # once. Agents can then choose from this set without the need to recompute.
    lies = compute_all_lies(manip_col_indexes, d)

    lie_indexes_by_lowest_cost = []
    for x in X:
        sorted_costs_and_index = sorted([(l, alpha*f(x[manip_col_indexes_falt], lie[manip_col_indexes_falt]))
                                        for l, lie in enumerate(lies)], key=lambda xx: xx[1])
        sorted_indexes = [l for l, _ in sorted_costs_and_index]
        lie_indexes_by_lowest_cost.append(sorted_indexes)




    best_strats_for_each_clf = []
    best_costs_for_each_clf  = []
    for clf_index, clf in enumerate(clfs):
        p = ps[clf_index]
        optimal_strats = {i: None for i in range(len(X))}
        optimal_costs  = {i: 0 for i in range(len(X))}
        pred = clf.predict(X)
        for i, score in enumerate(pred):
            if score == 1:
                optimal_strats[i] = X[i]
        for l in range(len(lies)):
            indexes_left_to_search = [i for i in optimal_strats if optimal_strats[i] is None]
            if len(indexes_left_to_search) == 0:
                break
            X_search = copy.deepcopy(X[indexes_left_to_search])
            for col_index in manip_col_indexes_falt:
                X_search[:,col_index] = lies[l][col_index]
            original_costs = np.array([alpha*f(X_search[i], X[indexes_left_to_search[i]]) for i in range(len(indexes_left_to_search))])
            costs = original_costs +  np.array([sum(p[l]*int(np.array_equiv(X_search[i][manip_col_indexes[l]],X[indexes_left_to_search[i]][manip_col_indexes[l]])) for l in range(len(manip_col_indexes)))
                               for i in range(len(indexes_left_to_search))])
            #print(costs)
            #print(costs)



            if decision_type == 'preds':
                pred = clf.predict(X_search)

                for i in range(len(pred)):
                    if original_costs[i] >= 1:
                        optimal_strats[indexes_left_to_search[i]] = X[indexes_left_to_search[i]]
                    elif pred[i] == 1 and costs[i] < 1:
                        optimal_strats[indexes_left_to_search[i]] = X_search[i]
                        optimal_costs[indexes_left_to_search[i]]  = costs[i]
        for i in optimal_strats:
            if optimal_strats[i] is None:
                optimal_strats[i] = X[i]
        best_strats_for_each_clf.append(list(optimal_strats.values()))
        best_costs_for_each_clf.append(list(optimal_costs.values()))
        # for ii, strat in enumerate(list(optimal_strats.values())):
        #     for xx_true, xx_opt in zip(X[ii], strat):
        #         if xx_true != xx_opt:
        #             print(X[ii], 'not equal', strat)
    best_strats_for_each_clf = np.array(best_strats_for_each_clf)
    best_costs_for_each_clf  = np.array(best_costs_for_each_clf)
    lie_count_by_index = [[0 for _ in range(len(manip_col_indexes))] for _ in range(len(clfs))]
    for clf_index, best_strat in enumerate(best_strats_for_each_clf):
        for i, col_index in enumerate(manip_col_indexes):
            lie_count_by_index[clf_index][i] = sum(1 for strat, x in zip(best_strat, X) if np.array_equiv(strat[col_index], x[col_index]))




    return best_strats_for_each_clf, lie_count_by_index, best_costs_for_each_clf#[clf.predict(np.array(best_strats_for_each_clf[i])) for i, clf in enumerate(clfs)], [clf.predict_proba(np.array(best_strats_for_each_clf[i])) for i, clf in enumerate(clfs)]










def compute_all_lies(col_indexes, dim):
    # Computes every possible lie given a set of manipulable columns
    lies = []
    cata_lies = list(itr.product(*[[i for i in range(len(indexes))] if len(indexes) > 1 else [0, 1] for indexes in col_indexes]))

    for cata_lie in cata_lies:
        lie = np.zeros([dim])
        for i, itm in enumerate(cata_lie):
            if len(col_indexes[i]) == 1:
                lie[col_indexes[i][0]] = itm
            else:
                lie[col_indexes[i][itm]] = 1
        lies.append(lie)

    #return lies
    return np.array(lies)

