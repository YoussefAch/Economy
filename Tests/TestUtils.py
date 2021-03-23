import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import pandas as pd
import numpy as np
from Economy_K import Economy_K
from Economy_Gamma import Economy_Gamma
from Economy_Gamma_Lite import Economy_Gamma_Lite
from Economy_K_multiClustering import Economy_K_multiClustering
try:
    import cPickle as pickle
except ImportError:
    import pickle
import json
from sklearn.metrics import cohen_kappa_score
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit
import time

def saveRealClassifiers(arguments):
    pathToClassifiers, folderRealData, dataset, classifier = arguments

    # path to data
    filepathTrain = folderRealData + '/' + dataset + '/' + dataset + '_TRAIN.tsv'

    # read data
    train = pd.DataFrame.from_csv(filepathTrain, sep='\t', header=None, index_col=None)
    mn = np.min(np.unique(train.iloc[:,0].values))

    train.iloc[:,0] = train.iloc[:,0].apply(lambda e: 0 if e==mn else 1)

    # get X and y
    Y_train = train.iloc[:, 0]
    X_train = train.loc[:, train.columns != train.columns[0]]


    max_t = X_train.shape[1]
    classifiers = {}

    ## Train classifiers for each time step
    for t in range(1, max_t+1):

        # use the same type classifier for each time step
        classifier_t = clone(classifier)
        # fit the classifier
        classifier_t.fit(X_train.iloc[:, :t], Y_train)
        # save it in memory
        classifiers[t] = classifier_t

    # save the model
    with open(pathToClassifiers + '/classifier'+dataset+'.pkl', 'wb') as output:
        pickle.dump(classifiers, output)






def saveRealModelsECONOMY(arguments):


    use_complete_ECO_model, pathECOmodel, sampling_ratio, orderGamma, ratioVal, pathToRealModelsECONOMY, pathToClassifiers, folderRealData, method, dataset, group, misClassificationCost, min_t, classifier, fears, feat = arguments


    # model name
    modelName = method + ',' + dataset + ',' + str(group)
    pathECOmodel = pathECOmodel+modelName+'.pkl'
    if not (os.path.exists(pathToRealModelsECONOMY + '/' + modelName + '.pkl')):
        # path to data
        # read data
        train_classifs = pd.read_csv(folderRealData + '/' + dataset + '/' + dataset + '_TRAIN_CLASSIFIERS.tsv', sep='\t', header=None, index_col=None, engine='python')
        estimate_probas = pd.read_csv(folderRealData + '/' + dataset + '/' + dataset + '_ESTIMATE_PROBAS.tsv', sep='\t', header=None, index_col=None, engine='python')

        # read data

        mn = np.min(np.unique(train_classifs.iloc[:,0].values))
        train_classifs.iloc[:,0] = train_classifs.iloc[:,0].apply(lambda e: 0 if e==mn else 1)
        estimate_probas.iloc[:,0] = estimate_probas.iloc[:,0].apply(lambda e: 0 if e==mn else 1)


        mx_t = train_classifs.shape[1] - 1

        # time cost
        timeCost = 0.01 * np.arange(mx_t+1) # arbitrary value

        # choose the method


        if (method == 'Gamma'):
            model = Economy_Gamma(misClassificationCost, timeCost, min_t, classifier, group, orderGamma, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset, feat)
        elif (method == 'K') :
            model = Economy_K(misClassificationCost, timeCost, min_t, classifier, group, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset, feat)
        elif (method == 'Gamma_lite'):
            model = Economy_Gamma_Lite(misClassificationCost, timeCost, min_t, classifier, group, orderGamma, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset, feat)
        else:
            model = Economy_K_multiClustering(misClassificationCost, timeCost, min_t, classifier, group, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset, feat)
        # fit the model
        pathToClassifiers = pathToClassifiers + 'classifier' + dataset
        model.fit(train_classifs, estimate_probas, ratioVal, pathToClassifiers)


        # save the model
        with open(pathToRealModelsECONOMY + '/' + modelName + '.pkl', 'wb') as output:
            pickle.dump(model, output)


def transform_to_format_fears(X):
    nbObs, length = X.shape
    for i in range(nbObs):
        ts = X.iloc[i,:]
        data = {'id':[i for _ in range(length)], 'timestamp':[k for k in range(1,length+1)], 'dim_X':list(ts.values)}
        if i==0:
            df = pd.DataFrame(data)
        else:
            df = df.append(pd.DataFrame(data))
    df = df.reset_index(drop=True)
    return df



def score(model, X_test, y_test, C_m, sampling_ratio, val=None, min_t=4, max_t=50):
    nb_observations, _ = X_test.shape
    score_computed = 0
    step = int(max_t*sampling_ratio) if int(max_t*sampling_ratio)>0 else 1
    start = step
    # We predict for every time series [label, tau*]

    if val:
        with open('RealData/'+model.dataset+'/val_preds'+'.pkl' ,'rb') as inp:
            donnes_pred = pickle.load(inp)
        with open('RealData/'+model.dataset+'/val_probas'+'.pkl' ,'rb') as inp:
            donnes_proba = pickle.load(inp)
    if not val:
        with open('RealData/'+model.dataset+'/test_preds'+'.pkl' ,'rb') as inp:
            donnes_pred = pickle.load(inp)
        with open('RealData/'+model.dataset+'/test_probas'+'.pkl' ,'rb') as inp:
            donnes_proba = pickle.load(inp)


    for i in range(nb_observations):
        # The approach is non-moyopic, for each time step we predict the optimal
        # time to make the prediction in the future.

        for t in range(start, max_t+1, step):
            # first t values of x

            x = np.array(list(X_test.iloc[i, :t]))

            pb = donnes_proba[t][i]
            # compute cost of future timesteps (max_t - t)
            send_alert, cost = model.forecastExpectedCost(x,pb)
            #compute tau*
            # predict the label of our time series when tau* = 0 or when we
            # reach max_t
            if (send_alert):
                if model.fears:
                    prediction = model.classifiers[t].predict(transform_to_format_fears(x.reshape(1, -1)))[0]
                elif model.feat:
                    prediction = donnes_pred[t][i]
                else:
                    prediction = model.classifiers[t].predict(x.reshape(1, -1))[0]


                if (prediction != y_test.iloc[i]):
                    score_computed += model.timeCost[t] + C_m
                else:
                    score_computed += model.timeCost[t]
                break
    return (score_computed/nb_observations)



def score_post_optimal(model, X_test, y_test, C_m, sampling_ratio, val=None, min_t=4, max_t=50):

    nb_observations, _ = X_test.shape
    score_computed = 0
    step = int(max_t*sampling_ratio) if int(max_t*sampling_ratio)>0 else 1
    start = step

    if val:
        with open('RealData/'+model.dataset+'/val_preds'+'.pkl' ,'rb') as inp:
            donnes_pred = pickle.load(inp)
        with open('RealData/'+model.dataset+'/val_probas'+'.pkl' ,'rb') as inp:
            donnes_proba = pickle.load(inp)
    if not val:
        with open('RealData/'+model.dataset+'/test_preds'+'.pkl' ,'rb') as inp:
            donnes_pred = pickle.load(inp)
        with open('RealData/'+model.dataset+'/test_probas'+'.pkl' ,'rb') as inp:
            donnes_proba = pickle.load(inp)

    # We predict for every time series [label, tau*]
    for i in range(nb_observations):
        post_costs = []
        timestamps_pred = []
        
        for t in range(start, max_t+1, step):


            x = np.array(list(X_test.iloc[i, :t]))

            pb = donnes_proba[t][i]
            # compute cost of future timesteps (max_t - t)
            _, cost = model.forecastExpectedCost(x,pb)
            post_costs.append(cost)
            timestamps_pred.append(t)
            #compute tau*
            # predict the label of our time series when tau* = 0 or when we
            # reach max_t
        tau_post_star = timestamps_pred[np.argmin(post_costs)]

        if model.fears:
            prediction = model.classifiers[t].predict(transform_to_format_fears(x.reshape(1, -1)))[0]
        elif model.feat:
            prediction = donnes_pred[t][i]
        else:
            prediction = model.classifiers[t].predict(x.reshape(1, -1))[0]

        if (prediction != y_test.iloc[i]):
            score_computed += model.timeCost[tau_post_star] + C_m
        else:
            score_computed += model.timeCost[tau_post_star]
 

    return (score_computed/nb_observations)



def computeScores(arguments):

    score_chosen, normalizeTime, C_m, orderGamma, pathToSaveScores, pathToRealModelsECONOMY, folderRealData, method, dataset, group, misClassificationCost, min_t, paramTime = arguments

    # model name
    modelName = method + ',' + dataset + ',' + str(group)
    # path to data
    filepathTest = folderRealData+'/'+dataset+'/'+dataset+'_VAL_SCORE.tsv'

    if not (os.path.exists(pathToSaveScores+'/score'+modelName+ ',' + str(paramTime)+'.json')):
        # read data
        print(modelName)

        val = pd.read_csv(filepathTest, sep='\t', header=None, index_col=None, engine='python')

        mn = np.min(np.unique(val.iloc[:,0].values))
        val.iloc[:,0] = val.iloc[:,0].apply(lambda e: 0 if e==mn else 1)

        # get X and y
        y_val = val.iloc[:, 0]
        X_val = val.loc[:, val.columns != val.columns[0]]
        mx_t = X_val.shape[1]
        # choose the method
        try:
            with open(pathToRealModelsECONOMY + '/' + modelName + '.pkl', 'rb') as input:
                
                model = pickle.load(input)
                if normalizeTime:
                    timeCostt = (1/mx_t) * paramTime * np.arange(model.timestamps[-1] + 1)
                else:
                    timeCostt = paramTime * np.arange(model.timestamps[-1]+1)
                setattr(model, 'timeCost', timeCostt)
        except pickle.UnpicklingError:
            print('PROOOOBLEM', modelName)
        if score_chosen == 'star':
            score_model = score(model, X_val, y_val, C_m, model.sampling_ratio, val=True, max_t=mx_t)
        if score_chosen == 'post':
            score_model = score_post_optimal(model, X_val, y_val, C_m, model.sampling_ratio, max_t=mx_t)
        modelName = method + ',' + dataset + ',' + str(group) + ',' + str(paramTime)

        with open(pathToSaveScores+'/score'+modelName+'.json', 'w') as outfile:
            json.dump({modelName:score_model}, outfile)
    else:
        with open(pathToSaveScores+'/score'+modelName+ ',' + str(paramTime)+'.json') as f:
            try:
                loadedjson = json.loads(f.read())
            except:
                print('BUUUUUUUUUUUUUUUUUUUUUG',modelName)
        modelName = list(loadedjson.keys())[0]
        score_model = list(loadedjson.values())[0]
    return (modelName,score_model)





def computeBestGroup(datasets, timeParams, modelName_score, INF, methods):
    bestGroup = {method + ',' + dataset + ',' + str(timeparam):[1,INF] for method in methods for dataset in datasets  for timeparam in timeParams}
    for e in modelName_score:
        (modelName, score_model) = e
        method, dataset, group, paramTime = modelName.split(',')
        group = int(group)
        paramTime = float(paramTime)
        if paramTime == 1.0:
            paramTime = int(paramTime)
        if (bestGroup[method + ',' + dataset + ',' + str(paramTime)][1] > score_model):
            bestGroup[method + ',' + dataset + ',' + str(paramTime)][0] = group
            bestGroup[method + ',' + dataset + ',' + str(paramTime)][1] = score_model
    return bestGroup



def evaluate(arguments):

    score_chosen, normalizeTime, C_m, pathToRealModels, folderRealData, dataset, method, group, paramTime, pathToSaveScores = arguments

    # model name
    modelName = method + ',' + dataset + ',' + str(group)
    # path to data
    filepathTest = folderRealData+'/'+dataset+'/'+dataset+'_TEST_SCORE.tsv'

    if not (os.path.exists(pathToSaveScores+'/EVALscore'+modelName+ ',' + str(paramTime)+'.json')):
        # read data
        test = pd.read_csv(filepathTest, sep='\t', header=None, index_col=None, engine='python')

        mn = np.min(np.unique(test.iloc[:,0].values))

        test.iloc[:,0] = test.iloc[:,0].apply(lambda e: 0 if e==mn else 1)
        # get X and y
        y_test = test.iloc[:, 0]
        X_test = test.loc[:, test.columns != test.columns[0]]
        mx_t = X_test.shape[1]
        modelName = method + ',' + dataset + ',' + str(group)

        with open(pathToRealModels + '/'+modelName + '.pkl', 'rb') as input:
            model = pickle.load(input)
            if normalizeTime:
                timeCostt = (1/mx_t) * paramTime * np.arange(model.timestamps[-1] + 1)
            else:
                timeCostt = paramTime * np.arange(model.timestamps[-1]+1)
            setattr(model, 'timeCost', timeCostt)

        if score_chosen == 'star':
            score_model = score(model, X_test, y_test, C_m, model.sampling_ratio, val=False, max_t=mx_t)
            with open(pathToSaveScores+'/EVALscore'+modelName+'.json', 'w') as outfile:
                json.dump({modelName:score_model}, outfile)
        if score_chosen == 'post':
            score_model = score_post_optimal(model, X_test, y_test, C_m, model.sampling_ratio, val=False, max_t=mx_t)
            with open(pathToSaveScores+'/EVALscorePOST'+modelName+'.json', 'w') as outfile:
                json.dump({modelName:score_model}, outfile)
        modelName = method + ',' + dataset + ',' + str(group) + ',' + str(paramTime)

        
    else:
        if score_chosen == 'star':
            with open(pathToSaveScores+'/EVALscore'+modelName+ ',' + str(paramTime)+'.json') as f:
                try:
                    loadedjson = json.loads(f.read())
                except:
                    print('BUUUUUUUUUUUUUUUUUUUUUG',modelName)
        else:
            with open(pathToSaveScores+'/EVALscorePOST'+modelName+ ',' + str(paramTime)+'.json') as f:
                try:
                    loadedjson = json.loads(f.read())
                except:
                    print('BUUUUUUUUUUUUUUUUUUUUUG',modelName)
        modelName = list(loadedjson.keys())[0]
        score_model = list(loadedjson.values())[0]
    return (modelName, score_model)


def scoreSR(arguments):

    SRParams, timecostParam, ep_probas, ep_preds, val_probas, val_preds, y_test_ep, y_test_val, nb_observations_val, nb_observations_ep, timestamps, timecost, max_t = arguments
    
    



    score = 0
    for i in range(nb_observations_ep):

        for t in timestamps:

            proba1 = ep_probas[t][i]
            proba2 = 1.0 - proba1
            if proba1 > proba2:
                maxiProba = proba1
                scndProba = proba2
            else:
                maxiProba = proba2
                scndProba = proba1

            # Stopping rule
            sr = SRParams[0] * maxiProba + SRParams[1] * (maxiProba-scndProba) + SRParams[2] * (t / max_t)

            if sr > 0 or t==timestamps[-1]:
                
                if y_test_ep.iloc[i] == ep_preds[t][i]:
                    score += timecost[t]
                else:
                    score += timecost[t] + 1 #C_m = 1
                break


    

    


    for i in range(nb_observations_val):

        for t in timestamps:

            proba1 = val_probas[t][i]
            proba2 = 1.0 - proba1
            if proba1 > proba2:
                maxiProba = proba1
                scndProba = proba2
            else:
                maxiProba = proba2
                scndProba = proba1

            # Stopping rule
            sr = SRParams[0] * maxiProba + SRParams[1] * (maxiProba-scndProba) + SRParams[2] * (t / max_t)

            if sr > 0 or t==timestamps[-1]:
                
                if y_test_val.iloc[i] == val_preds[t][i]:
                    score += timecost[t]
                else:
                    score += timecost[t] + 1 #C_m = 1
                break
    print('------------FINISH---------- : ', timecostParam)
    return score / (nb_observations_ep + nb_observations_val)

def predictSR(arguments):
    print('DEBUT !! ')

    SRParams, sampling_ratio, folderRealData, dataset, timecostParam = arguments

    with open(folderRealData+'/'+dataset+'/test_probas.pkl' ,'rb') as inp:
        test_probas = pickle.load(inp)
    with open(folderRealData+'/'+dataset+'/test_preds.pkl' ,'rb') as inp:
        test_preds = pickle.load(inp)

    # compute AvgCost
    filepathTest = folderRealData+'/'+dataset+'/'+dataset+'_TEST_SCORE.tsv'
    test = pd.read_csv(filepathTest, sep='\t', header=None, index_col=None, engine='python')
    li, col = test.shape
    max_t = col-1
    step = int(max_t*sampling_ratio) if int(max_t*sampling_ratio)>0 else 1
    timestamps = [t for t in range(step, max_t + 1, step)]
    mn = np.min(np.unique(test.iloc[:,0].values))
    timecost = timecostParam * np.arange(max_t+1)
    test.iloc[:,0] = test.iloc[:,0].apply(lambda e: 0 if e==mn else 1)
    # get X and y
    y_test = test.iloc[:, 0]
    del test
    nb_observations=li
    
    # compute trigger times
    trigger_times = []
    score = 0
    preds = []
    for i in range(nb_observations):

        for t in timestamps:

            proba1 = test_probas[t][i]
            proba2 = 1.0 - proba1
            if proba1 > proba2:
                maxiProba = proba1
                scndProba = proba2
            else:
                maxiProba = proba2
                scndProba = proba1

            # Stopping rule
            sr = SRParams[0] * maxiProba + SRParams[1] * (maxiProba-scndProba) + SRParams[2] * (t / max_t)

            if sr > 0 or t==timestamps[-1]:
                preds.append(test_preds[t][i])
                trigger_times.append(t)
                if y_test.iloc[i] == test_preds[t][i]:
                    score += timecost[t]
                else:
                    score += timecost[t] + 1 #C_m = 1
                break
    score = score/nb_observations
    print('score : ', score)
    
    kapa = round(cohen_kappa_score(y_test.values.tolist(), preds),2) 
    print('kapa : ', kapa)
    colms = ['Dataset', 'timeParam', 'Score', 'Med_tau', 'Kappa']
    v = [dataset, timecostParam, score, np.median(np.array(trigger_times)), kapa]
    
    return {key:value for key, value in zip(colms, v)}
    

    
    
            
    




def computePredictionsEconomy(arguments):
    normalizeTime, pathToSavePredictions, pathToIntermediateResults, folderRealData, pathToRealModels, dataset, method, group, paramTime = arguments


    # path to data
    filepathTest = folderRealData+'/'+dataset+'/'+dataset+'_TEST_SCORE.tsv'
    
    modelName = method + ',' + dataset + ',' + str(group)+',' + str(paramTime)
    if not (os.path.exists(pathToSavePredictions+'/PREDECO'+modelName+'.pkl')):

        modelName = method + ',' + dataset + ',' + str(group)

        # read data
        test = pd.read_csv(filepathTest, sep='\t', header=None, index_col=None, engine='python')
        mn = np.min(np.unique(test.iloc[:,0].values))
        test.iloc[:,0] = test.iloc[:,0].apply(lambda e: 0 if e==mn else 1)
        # get X and y
        y_test = test.iloc[:, 0]
        X_test = test.loc[:, test.columns != test.columns[0]]
        mx_t = X_test.shape[1]

        with open(pathToRealModels + '/' + modelName + '.pkl', 'rb') as input:
            model = pickle.load(input)
            if normalizeTime:
                timeCostt = (1/mx_t) * paramTime * np.arange(model.timestamps[-1] + 1)
            else:
                timeCostt = paramTime * np.arange(model.timestamps[-1]+1)
            setattr(model, 'timeCost', timeCostt)

        with open(folderRealData+'/'+dataset+'/test_probas.pkl' ,'rb') as inp:
            test_probas = pickle.load(inp)
        with open(folderRealData+'/'+dataset+'/test_preds.pkl' ,'rb') as inp:
            test_preds = pickle.load(inp)


        preds_tau = model.predict(X_test, [test_probas, test_preds])

        preds_post = model.predict_post_tau_stars(X_test, [test_probas, test_preds])

        preds_optimal = model.predict_optimal_algo(X_test, y_test, test_preds)
        metric = preds_tau, preds_post, preds_optimal
        modelName = method + ',' + dataset + ',' + str(group)+',' + str(paramTime)


        with open(pathToSavePredictions+'/PREDECO'+modelName+'.pkl', 'wb') as outfile:
            pickle.dump({modelName: [preds_tau, preds_post, preds_optimal]}, outfile)

        return preds_tau, preds_post, preds_optimal, modelName
    else:
        print('hello')
        with open(pathToSavePredictions+'/PREDECO'+modelName+'.pkl', 'rb') as outfile:
            loadedjson = pickle.load(outfile)
        
    
        return list(loadedjson.values())[0][0], list(loadedjson.values())[0][1], list(loadedjson.values())[0][2], list(loadedjson.keys())[0]




def computeMetricsNeeded(arguments):

    sampling_ratio = arguments[0]
    preds_tau, preds_post, preds_optimal, modelName = arguments[1]
    pathToIntermediateResults = arguments[2]
    folderRealData = arguments[3]

    method, dataset, group, timeparam = modelName.split(',')

    organized_metrics = {dataset+','+method:[]}
    tau_et = np.array(preds_tau)[:,0]
    tau_post = np.array(preds_post)[:,0]

    #print('taille tau_et : ', tau_et.shape)

    tau_opt = np.array(preds_optimal)[:,0]
    #print(modelName)
    #print('taille tau_post : ', tau_post.shape)

    f_et = np.array(preds_tau)[:,1]

    f_post = np.array(preds_post)[:,1]

    f_opt = np.array(preds_optimal)[:,1]

    objectives = [tau_et, tau_post, tau_opt, f_et, f_post, f_opt]
    organized_metrics[dataset+','+method].append(timeparam)
    for e in objectives:
        organized_metrics[dataset+','+method].append(np.mean(e))
        organized_metrics[dataset+','+method].append(np.std(e))

    for e1, e2 in zip(objectives[1:3], objectives[4:]):
        organized_metrics[dataset+','+method].append(np.mean(abs(tau_et-e1)))
        organized_metrics[dataset+','+method].append(np.std(abs(tau_et-e1)))
        organized_metrics[dataset+','+method].append(np.mean(abs(f_et-e2)))
        organized_metrics[dataset+','+method].append(np.std(abs(f_et-e2)))

    organized_metrics[dataset+','+method].append(group) #group



    # AUC import packages ???
    filepathTest = folderRealData+'/'+dataset+'/'+dataset+'_TEST_SCORE.tsv'
    test = pd.read_csv(filepathTest, sep='\t', header=None, index_col=None, engine='python')
    mn = np.min(np.unique(test.iloc[:,0].values))
    test.iloc[:,0] = test.iloc[:,0].apply(lambda e: 0 if e==mn else 1)
    y_test = test.iloc[:, 0]
    X_test = test.loc[:, test.columns != test.columns[0]]
    max_t = X_test.shape[1]
    # pourcentage des prediction à min_t
    min_t = int(max_t*sampling_ratio) if int(max_t*sampling_ratio)>0 else 1

    for i in range(min_t, max_t+1, min_t):
        mx_t = i
    count_min_t = 0
    count_max_t = 0
    for elem in tau_et:
        if (elem == min_t):
            count_min_t += 1
        if (elem == mx_t):
            count_max_t += 1

    count_min_t /= len(tau_et)
    count_max_t /= len(tau_et)
    organized_metrics[dataset+','+method].append(count_min_t)
    organized_metrics[dataset+','+method].append(count_max_t)

    print(np.array(preds_tau)[:,2])
    kappa_tau = round(cohen_kappa_score(y_test.values.tolist(), list(map(int, np.array(preds_tau)[:,2]))),2)
    kappa_opt = round(cohen_kappa_score(y_test.values.tolist(), list(map(int, np.array(preds_post)[:,2]))),2)
    kappa_post = round(cohen_kappa_score(y_test.values.tolist(), list(map(int, np.array(preds_optimal)[:,2]))),2)

    organized_metrics[dataset+','+method].append(kappa_tau)
    organized_metrics[dataset+','+method].append(kappa_post)
    organized_metrics[dataset+','+method].append(kappa_opt)


    # pourcentage des cas ou on est précoce au post optimal
    counters = [0,0,0,0]
    for tauu, tau_ppost, tau_opti in zip(tau_et, tau_post, tau_opt):
        if tauu < tau_ppost:
            counters[0] = counters[0] + 1
        if tauu < tau_opti:
            counters[1] = counters[1] + 1
        if tauu >= tau_ppost:
            counters[2] = counters[2] + 1
        if tauu >= tau_opti:
            counters[3] = counters[3] + 1
    for i in range(4):
        counters[i] = round(counters[i] / len(tau_et),2)

    organized_metrics[dataset+','+method].append(counters[0])
    organized_metrics[dataset+','+method].append(counters[1])
    organized_metrics[dataset+','+method].append(counters[2])
    organized_metrics[dataset+','+method].append(counters[3])

    # compute medians tau_et, tau_post, tau_opt, f_et, f_post, f_opt
    organized_metrics[dataset+','+method].append(np.median(tau_et))
    organized_metrics[dataset+','+method].append(np.median(tau_post))
    organized_metrics[dataset+','+method].append(np.median(tau_opt))
    organized_metrics[dataset+','+method].append(np.median(f_et))
    organized_metrics[dataset+','+method].append(np.median(f_post))
    organized_metrics[dataset+','+method].append(np.median(f_opt))

    # compute medians differences
    organized_metrics[dataset+','+method].append(np.median(abs(tau_et-tau_post)))
    organized_metrics[dataset+','+method].append(np.median(abs(f_et-f_post)))




    with open(pathToIntermediateResults+'/MetricsNeeded'+modelName+'.pkl', 'wb') as outfile:
        pickle.dump(organized_metrics, outfile)

    return organized_metrics
