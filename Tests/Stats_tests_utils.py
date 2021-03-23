import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import argparse
import json
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import multiprocessing
try:
    import cPickle as pickle
except ImportError:
    import pickle
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import math
import scipy as sp
import scipy.stats as st
import itertools as it
import Orange
import math



def matrix_ranking_vsSR(methods, metric, r, datasets, df_metrics_opt, df_metrics_SR, dataset_tempcost_SR, nameFig):
    methodds = [r'Eco-$\gamma$', 'SR']
    scores_methods = []
    for method in methods:
        meth_scores = []
        SR = []
        for dataset in datasets:
            meth_scores.append(df_metrics_opt[(df_metrics_opt['Dataset']==dataset) & (df_metrics_opt['timeParam']==dataset_tempcost_SR[dataset]) & (df_metrics_opt['Method']==method)][metric].values[0])
            
            
            SR.append(list(df_metrics_SR[(df_metrics_SR['Dataset'] == dataset) & (df_metrics_SR['timeParam']==float(dataset_tempcost_SR[dataset]))][metric].values)[0])
        
        scores_methods.append(meth_scores)
    scores_methods.append(SR)
    iman_davenport, p_value, rankings_avg, rankings_cmp = friedman_test(r, *scores_methods)

    if p_value < 0.05:

        # Returns critical difference for Nemenyi
        cd = Orange.evaluation.scoring.compute_CD(avranks=rankings_avg, n=len(datasets), alpha="0.05", test="nemenyi")

        rank_viz_fpath = "experiments/PlotsPaper/figure_" + nameFig[0] +  ".png"
        lowv = math.floor(min(rankings_avg))
        highv = math.ceil(max(rankings_avg))
        width = (highv - lowv)*1.2+2   +9

        

        Orange.evaluation.scoring.graph_ranks(
            filename=rank_viz_fpath,
            avranks=rankings_avg,
            names=methodds,
            cd=cd,
            lowv=lowv,
            highv=highv,
            width=width,
            fontsize=3,
            textspace=5,
            reverse=False)
        
        plt.close()


    # get the ordered list of clssifiers
    prov_df = pd.DataFrame({'classifiers':methodds, 'rankings_avg':rankings_avg })
    prov_df.sort_values(by='rankings_avg', inplace=True)
    ordered_classifiers = methodds

    # build the matrix of pairewide comparison
    pairewise_comparison_df = pd.DataFrame(
            np.zeros(shape=(len(ordered_classifiers),len(ordered_classifiers))),
            columns=ordered_classifiers,
            index=ordered_classifiers
            )

    # pairewide comparison
    for classifier_col in ordered_classifiers:
        for classifier_row in ordered_classifiers:
            if classifier_col != classifier_row:
                z, null_hypothesis_rejected = wilcoxon_test(scores_methods[methodds.index(classifier_col)], scores_methods[methodds.index(classifier_row)])
                if null_hypothesis_rejected:
                    pairewise_comparison_df[classifier_col][classifier_row] = 1

    plt.imshow(pairewise_comparison_df, cmap=plt.cm.gray)
    plt.xticks(np.arange(0,len(pairewise_comparison_df)), pairewise_comparison_df.columns, rotation='vertical', fontsize=15)
    plt.yticks(np.arange(0,len(pairewise_comparison_df)), pairewise_comparison_df.columns, fontsize=15)
    plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
    plt.savefig("experiments/PlotsPaper/figure_" + nameFig[1] +  ".png", bbox_inches='tight')
    plt.close()



def matrix_ranking(methods, metric, r, datasets, df_metrics_opt, dataset_tempcost, nameFig):
    print('METS : ', methods)
    methodds = [r'Eco-$\gamma$', r'Eco-$\gamma$-lite', 'Eco-K', 'Eco-multi-K' ]
    scores_methods = []
    for method in methods:
        meth_scores = []
        for dataset in datasets:
            meth_scores.append(df_metrics_opt[(df_metrics_opt['Dataset']==dataset) & (df_metrics_opt['timeParam']==dataset_tempcost[dataset]) & (df_metrics_opt['Method']==method)][metric].values[0])
        scores_methods.append(meth_scores)

    iman_davenport, p_value, rankings_avg, rankings_cmp = friedman_test(r, *scores_methods)

    if p_value < 0.05:

        # Returns critical difference for Nemenyi
        cd = Orange.evaluation.scoring.compute_CD(avranks=rankings_avg, n=len(datasets), alpha="0.05", test="nemenyi")

        rank_viz_fpath = "experiments/PlotsPaper/figure_" + nameFig[0] +  ".png"
        lowv = math.floor(min(rankings_avg))
        highv = math.ceil(max(rankings_avg))
        width = (highv - lowv)*1.2+2   +7

        

        Orange.evaluation.scoring.graph_ranks(
            filename=rank_viz_fpath,
            avranks=rankings_avg,
            names=methodds,
            cd=cd,
            lowv=lowv,
            highv=highv,
            width=width,
            fontsize=3,
            textspace=5,
            reverse=False)
        
        plt.close()


    # get the ordered list of clssifiers
    prov_df = pd.DataFrame({'classifiers':methodds, 'rankings_avg':rankings_avg })
    prov_df.sort_values(by='rankings_avg', inplace=True)
    ordered_classifiers = methodds

    # build the matrix of pairewide comparison
    pairewise_comparison_df = pd.DataFrame(
            np.zeros(shape=(len(ordered_classifiers),len(ordered_classifiers))),
            columns=ordered_classifiers,
            index=ordered_classifiers
            )

    # pairewide comparison
    for classifier_col in ordered_classifiers:
        for classifier_row in ordered_classifiers:
            if classifier_col != classifier_row:
                z, null_hypothesis_rejected = wilcoxon_test(scores_methods[methodds.index(classifier_col)], scores_methods[methodds.index(classifier_row)])
                if null_hypothesis_rejected:
                    pairewise_comparison_df[classifier_col][classifier_row] = 1

    plt.imshow(pairewise_comparison_df, cmap=plt.cm.gray)
    plt.xticks(np.arange(0,len(pairewise_comparison_df)), pairewise_comparison_df.columns, rotation='vertical', fontsize=15)
    plt.yticks(np.arange(0,len(pairewise_comparison_df)), pairewise_comparison_df.columns, fontsize=15)
    plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
    plt.savefig("experiments/PlotsPaper/figure_" + nameFig[1] +  ".png", bbox_inches='tight')
    plt.close()


def friedman_test(r, *args):
    """
    source: http://tec.citius.usc.es/stac/doc/_modules/stac/nonparametric_tests.html#friedman_test

        Performs a Friedman ranking test.
        Tests the hypothesis that in a set of k dependent samples groups (where k >= 2) at least two of the groups represent populations with different median values.

        Parameters
        ----------
        sample1, sample2, ... : array_like
            The sample measurements for each group.

        Returns
        -------
        F-value : float
            The computed F-value of the test.
        p-value : float
            The associated p-value from the F-distribution.
        rankings : array_like
            The ranking for each group.
        pivots : array_like
            The pivotal quantities for each group.

        References
        ----------
        M. Friedman, The use of ranks to avoid the assumption of normality implicit in the analysis of variance, Journal of the American Statistical Association 32 (1937) 674–701.
        D.J. Sheskin, Handbook of parametric and nonparametric statistical procedures. crc Press, 2003, Test 25: The Friedman Two-Way Analysis of Variance by Ranks
    """


    k = len(args)
    if k < 2: raise ValueError('Less than 2 levels')
    n = len(args[0])
    if len(set([len(v) for v in args])) != 1: raise ValueError('Unequal number of samples')

    if r=='petit':
        rev = False
    else:
        rev = True

    rankings = []
    for i in range(n):
        row = [col[i] for col in args]
        row_sort = sorted(row, reverse=rev)
        rankings.append([row_sort.index(v) + 1 + (row_sort.count(v)-1)/2. for v in row])

    rankings_avg = [sp.mean([case[j] for case in rankings]) for j in range(k)]
    rankings_cmp = [r/sp.sqrt(k*(k+1)/(6.*n)) for r in rankings_avg]

    chi2 = ((12*n)/float((k*(k+1))))*((sp.sum(r**2 for r in rankings_avg))-((k*(k+1)**2)/float(4)))
    iman_davenport = ((n-1)*chi2)/float((n*(k-1)-chi2))

    p_value = 1 - st.f.cdf(iman_davenport, k-1, (k-1)*(n-1))

    return iman_davenport, p_value, rankings_avg, rankings_cmp




def wilco_approche_adapt_vs_nonadapt(df_metrics_adapt, df_metrics_nonadapt, metric, timeParams, datasets, methods):
    cols = ['method', 'timeParam', 'null_hypothesis_rejected', 'z']
    print('METS : ', methods)
    dataframe = pd.DataFrame(columns=cols)
    for timeparam in timeParams:
        for method in methods:
            line = [method, timeparam]
            scores_opt =  df_metrics_adapt[(df_metrics_adapt['timeParam']==timeparam) & (df_metrics_adapt['Method']==method)]['Score'].values
            scores_one =  df_metrics_nonadapt[(df_metrics_nonadapt['timeParam']==timeparam) & (df_metrics_nonadapt['Method']==method)]['Score'].values

            z, null_hypothesis_rejected = wilcoxon_test(scores_opt, scores_one)
            line.append(null_hypothesis_rejected)
            line.append(z)
            dataframe = dataframe.append({key:value for key, value in zip(cols, line)}, ignore_index=True)
    return dataframe







def wilcoxon_test(score_A, score_B):



    # compute abs delta and sign
    delta_score = [score_B[i] - score_A[i] for i in range(len(score_A))]
    sign_delta_score = list(np.sign(delta_score))
    abs_delta_score = list(map(abs, delta_score))

    N_r = float(len(delta_score))

    # hadling scores
    score_df = pd.DataFrame({'abs_delta_score':abs_delta_score, 'sign_delta_score':sign_delta_score })

    # sort
    score_df.sort_values(by='abs_delta_score', inplace=True)
    score_df.index = range(1,len(score_df)+1)

    # adding ranks
    score_df['Ranks'] = score_df.index
    score_df['Ranks'] = score_df['Ranks'].astype('float64')

    score_df.dropna(inplace=True)

    # z : pouput value
    W = sum(score_df['sign_delta_score'] * score_df['Ranks'])
    z = W/(math.sqrt(N_r*(N_r+1)*(2*N_r+1)/6.0))

    # rejecte or not the null hypothesis
    null_hypothesis_rejected = False
    if z < -1.96 or z > 1.96:
        null_hypothesis_rejected = True

    return z, null_hypothesis_rejected



def wilco_approches_vs_SR(df_metrics_approches, df_metrics_SR, metric, timeParams, datasets, methods):
    cols = ['method', 'timeParam', 'null_hypothesis_rejected', 'z']
    dataframe = pd.DataFrame(columns=cols)
    for timeparam in timeParams:
        for method in methods:
            line = [method, timeparam]
            
            scores_approche =  df_metrics_approches[(df_metrics_approches['timeParam']==timeparam) & (df_metrics_approches['Method']==method)]['Score'].values
            scores_SR =  df_metrics_SR[df_metrics_SR['timeParam']==float(timeparam)]['Score'].values
            z, null_hypothesis_rejected = wilcoxon_test(scores_approche, scores_SR)
            line.append(null_hypothesis_rejected)
            line.append(z)
            dataframe = dataframe.append({key:value for key, value in zip(cols, line)}, ignore_index=True)
    return dataframe