#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 18:45:48 2021

@author: joeyallen
"""
if __name__ == '__main__':
    import pickle
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    indir = 'Outputs/'
    files = [x for x in os.listdir(indir) if 'advRetrained30combo' in x]
    
    num_iters = 30
    levels = ['base']+['iter_'+str(i) for i in range(num_iters)]#, 'iter_0', 'iter_1', 'iter_2', 'iter_3', 'iter_4']
    #levels_dict = {'base':0, 'iter_0:':1, 'iter_1':2, 'iter_2':3, 'iter_3':4, 'iter_4':5}
    metrics = ['FP','FN','roc']
    clfs = ['LR_0.1','DT_0.1']#['GB_base', 'LG_base', 'GB_equal', 'LG_equal', 'LR_0.2', 'LR_0.1', 'LR_0.01', 'DT_0.2', 'DT_0.1', 'DT_0.01']
    splits = [0,1]#,2,3,4]
    groups = ['g0','g1']
    
    
    for file in files:
        #file = files[0]
        
        d = pickle.load(open(indir+file, 'rb'))
        
        columns = ['clf','metric','group'] + levels
        all_data = []
        for clf in clfs:
            data = {}
            for level in levels:
                data[level] = {}
                for metric in metrics:
                    g0_metric_values = []
                    g1_metric_values = []
                    for split in splits:
                        g0_metric_values.append(d[split][clf][level]['g0'][metric])
                        g1_metric_values.append(d[split][clf][level]['g1'][metric])
                    data[level][metric] = {}
                    data[level][metric]['g0'] = np.mean(g0_metric_values)
                    data[level][metric]['g1'] = np.mean(g1_metric_values)
            for metric in metrics:
                for group in groups:
                    all_data.append([clf,metric,group] + [data[level][metric][group] for level in levels])
        df = pd.DataFrame(all_data,columns=columns)
        df.to_csv('Outputs/metric_csvs/'+file[:-7]+'.csv',index=False)
    
    