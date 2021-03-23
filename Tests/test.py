
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import argparse
import json
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from TestUtils import saveRealClassifiers, saveRealModelsECONOMY, score, computeScores, computeBestGroup, evaluate, computePredictionsEconomy, computeMetricsNeeded
import multiprocessing
try:
    import cPickle as pickle
except ImportError:
    import pickle
if __name__ == '__main__':

    #########################################################################################
    print('############################### TODOS ##################################')
    #########################################################################################
    # todo :
    #        - meme clustering pour les deux méthodes
    #        -  pourcentage max_t
    #        - AUC n'a pas de sens (nous n'avons pas le même classifieur pour tous les individus calibration modèle) remplacer l'auc par kappa
    #        - evaluer les classifieurs à chaque pas de temps, (choix de min_t selon la perf des classifieurs) evolution de l'auc moyenne par rapport au temps
    #        - score à rajouter sur les métriques
    #        - il faut apprendre economy models pour tous les timestamps et après pour chaque sampling pouvoir les utiliser


    #########################################################################################
    print('############################### PARAMS CONFIG ##################################')
    #########################################################################################
    # command line
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--pathToInputParams', help='path to json file with input params', required=True)
    parser.add_argument('--pathToOutputs', help='path outputs', required=True)

    args = parser.parse_args()
    pathToParams = args.pathToInputParams


    # load input params
    with open(pathToParams) as configfile:
        configParams = json.load(configfile)
    #configParams = json.load(open(pathToParams))

    # set variables
    nbDatasets = configParams['nbDatasets']
    folderRealData = configParams['folderRealData']
    sampling_ratio = configParams['sampling_ratio']
    # et summary and sort by train set size

    datasets = configParams['Datasets']
    print(datasets)
    classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    timeParams = configParams['timeParams']
    nbGroups = np.arange(1,configParams['nbGroups'])
    methods =  configParams['methods']
    C_m = configParams['misClassificationCost']
    misClassificationCost = np.array([[0,C_m],
                                      [C_m,0]])
    min_t = configParams['min_t']
    pathToClassifiers = configParams['pathToClassifiers']
    allECOmodelsAvailable = configParams['allECOmodelsAvailable']
    nb_core = multiprocessing.cpu_count()
    orderGamma = configParams['orderGamma']
    ratioVal = configParams['ratioVal']
    pathToIntermediateResults = configParams['pathToIntermediateResults']
    pathToResults = configParams['pathToResults']
    saveClassifiers = configParams['saveClassifiers']
    pathToRealModelsECONOMY = configParams['pathToRealModelsECONOMY']
    pathToSaveScores = configParams['pathToSaveScores']
    pathToSavePredictions = configParams['pathToSavePredictions']
    normalizeTime = configParams['normalizeTime']
    use_complete_ECO_model = configParams['use_complete_ECO_model']
    pathECOmodel = configParams['pathECOmodel']
    fears = configParams['fears']
    score_chosen = configParams['score_chosen']
    feat = configParams['feat']
    INF = 10000000





    ########################################################################################
    print('############################## SAVE CLASSIFIERS ################################')
    ########################################################################################
    if saveClassifiers:
        func_args_classifs = [(pathToClassifiers, folderRealData, dataset, classifier) for dataset in datasets]
        Parallel(n_jobs=nb_core)(delayed(saveRealClassifiers)(func_arg) for func_arg in func_args_classifs)




    ########################################################################################
    print('################################ SAVE ECONOMY ##################################')
    ########################################################################################
    if not allECOmodelsAvailable:
        func_args_eco = [(use_complete_ECO_model, pathECOmodel, sampling_ratio, orderGamma, ratioVal, pathToRealModelsECONOMY, pathToClassifiers, folderRealData, method, dataset, group, misClassificationCost, min_t, classifier, fears, feat) for dataset in datasets for group in nbGroups for method in methods]
        Parallel(n_jobs=nb_core)(delayed(saveRealModelsECONOMY)(func_arg) for func_arg in func_args_eco)
    



    #########################################################################################
    print('############################### Compute scores #################################')
    #########################################################################################

    func_args_score = [(score_chosen, normalizeTime, C_m, orderGamma, pathToSaveScores, pathToRealModelsECONOMY, folderRealData, method, dataset, group, misClassificationCost, min_t, paramTime) for dataset in datasets for group in nbGroups for paramTime in timeParams for method in methods]
    modelName_score = Parallel(n_jobs=nb_core)(delayed(computeScores)(func_arg) for func_arg in func_args_score)
    with open(pathToIntermediateResults+'/modelName_score.pkl', 'wb') as outfile:
        pickle.dump(modelName_score, outfile)
    #with open(pathToIntermediateResults+'/modelName_score.pkl', 'rb') as outfile:
    #    modelName_score = pickle.load(outfile)



    #########################################################################################
    print('##################### Compute best hyperparam : nbGroups #######################')
    #########################################################################################
    bestGroup = computeBestGroup(datasets, timeParams, modelName_score, INF, methods)
    with open(pathToIntermediateResults+'/bestGroup.pkl', 'wb') as outfile:
        pickle.dump(bestGroup, outfile)
    #with open(pathToIntermediateResults+'/bestGroup.pkl', 'rb') as outfile:
    #    bestGroup = pickle.load(outfile)



    #########################################################################################
    print('############################ Compute best scores ###############################')
    #########################################################################################
    # use the best group for evaluating on data test
    # compute scores
    func_args_best_score = []
    for dataset in datasets:
        for paramTime in timeParams:
            for method in methods:
                func_args_best_score.append((score_chosen, normalizeTime, C_m, pathToRealModelsECONOMY, folderRealData, dataset, method, bestGroup[method + ',' + dataset + ',' + str(paramTime)][0], paramTime, pathToSaveScores))
    best_score = Parallel(n_jobs=nb_core)(delayed(evaluate)(func_arg) for func_arg in func_args_best_score)
    with open(pathToIntermediateResults+'/best_score.pkl', 'wb') as outfile:
        pickle.dump(best_score, outfile)
    #with open(pathToIntermediateResults+'/best_score.pkl', 'rb') as outfile:
    #    best_score = pickle.load(outfile)


    #########################################################################################
    print('############################ Compute best scores post opt ###############################')
    #########################################################################################
    # use the best group for evaluating on data test
    # compute scores
    func_args_best_score_post = []
    for dataset in datasets:
        for paramTime in timeParams:
            for method in methods:
                func_args_best_score_post.append(('post', normalizeTime, C_m, pathToRealModelsECONOMY, folderRealData, dataset, method, bestGroup[method + ',' + dataset + ',' + str(paramTime)][0], paramTime, pathToSaveScores))
    best_score_post = Parallel(n_jobs=nb_core)(delayed(evaluate)(func_arg) for func_arg in func_args_best_score_post)
    with open(pathToIntermediateResults+'/best_score_post.pkl', 'wb') as outfile:
        pickle.dump(best_score_post, outfile)
    #with open(pathToIntermediateResults+'/best_score_post.pkl', 'rb') as outfile:
    #    best_score_post = pickle.load(outfile)


    #########################################################################################
    print('############################ Compute best scores ###############################')
    #########################################################################################
    results = {}
    for e in best_score:
        (modelName, score_model) = e
        results[modelName] = score_model

    with open(pathToResults+'/results.pkl', 'wb') as outfile:
        pickle.dump(results, outfile)

    results_post = {}
    for e in best_score_post:
        (modelName, score_model) = e
        results_post[modelName] = score_model

    with open(pathToResults+'/results_post.pkl', 'wb') as outfile:
        pickle.dump(results_post, outfile)

    """with open(pathToResults+'/results.pkl', 'rb') as outfile:
        results = pickle.load(outfile)
    with open(pathToResults+'/results_post.pkl', 'rb') as outfile:
        results_post = pickle.load(outfile)"""
    #########################################################################################
    print('############################ Compute predictions ###############################')
    #########################################################################################
    func_args_preds = []
    for dataset in datasets:
        for method in methods:
            for paramTime in timeParams:
                func_args_preds.append((normalizeTime, pathToSavePredictions, pathToIntermediateResults, folderRealData, pathToRealModelsECONOMY, dataset, method, bestGroup[method + ',' + dataset + ',' + str(paramTime)][0], paramTime))
    predictions = Parallel(n_jobs=nb_core)(delayed(computePredictionsEconomy)(func_arg) for func_arg in func_args_preds)
    print('FINIIIIIIIIIIIIIISH ############################################################################################"')
    #with open(pathToResults+'/predictions.pkl', 'wb') as outfile:
    #    pickle.dump(predictions, outfile)

    #with open(pathToResults+'/predictions.pkl', 'rb') as outfile:
    #    predictions = pickle.load(outfile)


    #########################################################################################
    print('############################## Compute metrics #################################')
    #########################################################################################
    
    metrics = Parallel(n_jobs=nb_core)(delayed(computeMetricsNeeded)([ sampling_ratio, func_arg, pathToIntermediateResults,folderRealData]) for func_arg in predictions)
    cols = ['Dataset','Method', 'Score', 'timeParam', 'meanTauStar','stdTauStar','meanTauPost','stdTauPost','meanTauOPT','stdTauOPT', 'mean_f_Star','std_f_Star', 'mean_f_Post','std_f_Post', 'mean_f_Opt','std_f_Opt', 'mean_diff_tauStar_tauPost','std_diff_tauStar_tauPost', 'mean_diff_fStar_fPost','std_diff_fStar_fPost', 'mean_diff_tauStar_tauOpt','std_diff_tauStar_tauOpt', 'mean_diff_fStar_fOpt','std_diff_fStar_fOpt', 'Group', 'pourcentage_min_t', 'pourcentage_max_t', 'kappa_star', 'kappa_post', 'kappa_opt', 'pourcentage_taustar_inf_taupost', 'pourcentage_taustar_inf_tauopt', 'pourcentage_taustar_supstrict_taupost', 'pourcentage_taustar_sustrict_tauopt', 'median_tau_et', 'median_tau_post', 'median_tau_opt', 'median_f_et', 'median_f_post', 'median_f_opt', 'median_diff_tau_et_post', 'median_diff_f_et_post', 'Score_post']
    df_metrics = pd.DataFrame(columns=cols)
    for e in metrics:
        for k,v in e.items():
            dataset, method = k.split(',')
            a = str(v[21])
            b = str(v[0])
            v.insert(0, results[method+','+dataset+','+a+','+b])
            v.insert(0, method)
            v.insert(0, dataset)
            v.append(results_post[method+','+dataset+','+a+','+b])
            df_metrics = df_metrics.append({key:value for key, value in zip(cols, v)}, ignore_index=True)
            print(df_metrics)
    print(df_metrics['Group'])
    with open(pathToResults+'/df_metrics.pkl', 'wb') as outfile:
        pickle.dump(df_metrics, outfile)


    