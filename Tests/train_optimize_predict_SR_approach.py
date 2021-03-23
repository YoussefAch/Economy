import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import argparse
import json
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from TestUtils import predictSR, scoreSR
import multiprocessing
try:
    import cPickle as pickle
except ImportError:
    import pickle

filename = "experiments/experiment1/df_metrics.pkl"
with open(filename, 'rb') as input:
    df_metrics_opt = pickle.load(input)


methods = np.unique(df_metrics_opt['Method'])
datasets = list(np.unique(df_metrics_opt['Dataset']))

timeParams = list(np.unique(df_metrics_opt['timeParam']))

timeParams = list(map(float, timeParams))
timeParams.sort()
timeParams = list(map(str, timeParams))
timeParams.pop()
timeParams.append('1')


folderRealData = 'RealData'
nb_core = multiprocessing.cpu_count()
sampling_ratio = 0.05



# SR params to determine
SRParamsOptimized = {timeparam:{dataset:[]} for timeparam in timeParams for dataset in datasets}
candidates_parameters = np.linspace(-1,1,41)
possibles_gammas = []
for gamma1 in candidates_parameters:
    for gamma2 in candidates_parameters:
        for gamma3 in candidates_parameters:
            possibles_gammas.append([gamma1, gamma2, gamma3])


for dataset in datasets:

    with open(folderRealData+'/'+dataset+'/ep_probas.pkl' ,'rb') as inp:
        ep_probas = pickle.load(inp)
    with open(folderRealData+'/'+dataset+'/ep_preds.pkl' ,'rb') as inp:
        ep_preds = pickle.load(inp)

    with open(folderRealData+'/'+dataset+'/val_probas.pkl' ,'rb') as inp:
        val_probas = pickle.load(inp)
    with open(folderRealData+'/'+dataset+'/val_preds.pkl' ,'rb') as inp:
        val_preds = pickle.load(inp)


    filepathep = folderRealData+'/'+dataset+'/'+dataset+'_ESTIMATE_PROBAS.tsv'
    ep = pd.read_csv(filepathep, sep='\t', header=None, index_col=None, engine='python')
    nb_observations_ep, col = ep.shape
    max_t = col-1
    step = int(max_t*sampling_ratio) if int(max_t*sampling_ratio)>0 else 1
    timestamps = [t for t in range(step, max_t + 1, step)]


    # get X and y
    y_test_ep = ep.iloc[:, 0]

    filepathval = folderRealData+'/'+dataset+'/'+dataset+'_VAL_SCORE.tsv'
    val = pd.read_csv(filepathval, sep='\t', header=None, index_col=None, engine='python')
    nb_observations_val, col = val.shape
    y_test_val = val.iloc[:, 0]

    del ep, val

    for timecostParam in timeParams:
        timecost = float(timecostParam) * np.arange(max_t+1)
        args_parallel = []
        for elm in possibles_gammas:
            args_parallel.append([elm, timecostParam, ep_probas, ep_preds, val_probas, val_preds, y_test_ep, y_test_val, nb_observations_val, nb_observations_ep, timestamps, timecost, max_t])
        predictions = Parallel(n_jobs=nb_core)(delayed(scoreSR)(func_arg) for func_arg in args_parallel)

        index = np.argmin(np.array(predictions))
        print('time : ', timecostParam, 'Dataset', dataset, ' ', possibles_gammas[index])
        SRParamsOptimized[timecostParam][dataset] = possibles_gammas[index]



with open('experiments/experiment_SR_approach/SROptimalParams.pkl', 'wb') as inp:
    pickle.dump(SRParamsOptimized, inp)
