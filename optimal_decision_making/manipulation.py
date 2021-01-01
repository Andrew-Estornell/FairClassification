import numpy as np
import itertools as itr



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
    best_probas  = [[] for _ in clfs]
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








def compute_all_lies(col_indexes, dim):
    # Computes every possible lie given a set of manipulable columns
    lies = []
    cata_lies = list(itr.product(*[[i for i in range(len(indexes))] if len(indexes) > 1 else [0, 1] for indexes in col_indexes]))

    #print(list(cata_lies))
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

