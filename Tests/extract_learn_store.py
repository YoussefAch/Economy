import pandas as pd
import tsfel
import multiprocessing
from xgboost import XGBClassifier, DMatrix
import os
import numpy as np
from joblib import Parallel, delayed
try:
    import cPickle as pickle
except ImportError:
    import pickle

"""
    This file supposes that you have already downloaded the data from : www.aaaa.bbb
    and data is stored in a folder named RealData (you can change this name using the variable folderRealData) 

    This file reproduces the results in multiple steps :
        - step 1 : extracts features 
        - step 2 : learn xgb classifiers
        - step 3 : store predictions in data files



"""

# create a folder named classifiersRealxgb
try:
    os.mkdir('/classifiersRealxgb')
except OSError:
    print('Failed creating directory')

folderRealData = 'RealData'
datasets = ['ItalyPowerDemand']
sampling_ratio = 0.05
googleSheet_name = "Features_dev"
with open('google.pkl', 'rb') as out:
    cfg_file = pickle.load(out)
nb_core = multiprocessing.cpu_count()




################################################################################################################## 
##################################################### STEP 1 #####################################################
################################################################################################################## 

for dataset in datasets:

    print(dataset)

    path1 = folderRealData+'/'+dataset+'/'+ dataset + '_TEST_SCORE.tsv'
    path2 = folderRealData+'/'+dataset+'/'+ dataset + '_VAL_SCORE.tsv'
    path3 = folderRealData+'/'+dataset+'/'+ dataset + '_TRAIN_CLASSIFIERS.tsv'
    path4 = folderRealData+'/'+dataset+'/'+ dataset + '_ESTIMATE_PROBAS.tsv'

    d1 = pd.read_csv(path1, sep='\t', header=None, index_col=None)
    d2 = pd.read_csv(path2, sep='\t', header=None, index_col=None)
    d3 = pd.read_csv(path3, sep='\t', header=None, index_col=None)
    d4 = pd.read_csv(path4, sep='\t', header=None, index_col=None)

    d1_X =  d1.loc[:, d1.columns != d1.columns[0]].to_numpy()
    d2_X =  d2.loc[:, d2.columns != d2.columns[0]].to_numpy()
    d3_X =  d3.loc[:, d3.columns != d3.columns[0]].to_numpy()
    d4_X =  d4.loc[:, d4.columns != d4.columns[0]].to_numpy()

    max_t = d1.shape[1]-1
    step = int(max_t*sampling_ratio) if int(max_t*sampling_ratio)>0 else 1
    timestamps= [t for t in range(step, max_t+1, step)]


    for t in timestamps:

        print(t)


        path1_s = folderRealData+'/'+dataset+'/'+ dataset + 'test_features_'+str(t)+'.tsv'
        path2_s = folderRealData+'/'+dataset+'/'+ dataset + 'val_features_'+str(t)+'.tsv'
        path3_s = folderRealData+'/'+dataset+'/'+ dataset + 'train_features_'+str(t)+'.tsv'
        path4_s = folderRealData+'/'+dataset+'/'+ dataset + 'ep_features_'+str(t)+'.tsv'



        try:


            if not (os.path.exists(path1_s)):
                d1_X_feat = tsfel.time_series_features_extractor(cfg_file, d1_X[:,:t], fs=100)
                d1_X_feat.columns = ['f'+str(ti) for ti in range(d1_X_feat.shape[1])]
                d1_X_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
                d1_X_feat = d1_X_feat.fillna(d1_X_feat.median())
                d1_X_feat.to_csv(path1_s,sep='\t', header=None, index=False)

            if not (os.path.exists(path2_s)):
                d2_X_feat = tsfel.time_series_features_extractor(cfg_file, d2_X[:,:t], fs=100)
                d2_X_feat.columns = ['f'+str(ti) for ti in range(d2_X_feat.shape[1])]
                d2_X_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
                d2_X_feat = d2_X_feat.fillna(d2_X_feat.median())
                d2_X_feat.to_csv(path2_s,sep='\t', header=None, index=False)

            if not (os.path.exists(path3_s)):
                d3_X_feat = tsfel.time_series_features_extractor(cfg_file, d3_X[:,:t], fs=100)
                d3_X_feat.columns = ['f'+str(ti) for ti in range(d3_X_feat.shape[1])]
                d3_X_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
                d3_X_feat = d3_X_feat.fillna(d3_X_feat.median())
                d3_X_feat.to_csv(path3_s,sep='\t', header=None, index=False)

            if not (os.path.exists(path4_s)):
                d4_X_feat = tsfel.time_series_features_extractor(cfg_file, d4_X[:,:t], fs=100)
                d4_X_feat.columns = ['f'+str(ti) for ti in range(d4_X_feat.shape[1])]
                d4_X_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
                d4_X_feat = d4_X_feat.fillna(d4_X_feat.median())
                d4_X_feat.to_csv(path4_s,sep='\t', header=None, index=False)


        except:

            print('PROBLEM , ',t,dataset)
            d1_X_pb = d1.loc[:, d1.columns != d1.columns[0]]
            d1_X_pb = d1_X_pb.iloc[:,:t]
            d1_X_pb.columns = ['f'+str(ti) for ti in range(d1_X_pb.shape[1])]
            d1_X_pb.to_csv(path1_s,sep='\t', header=None, index=False)


            d2_X_pb = d2.loc[:, d2.columns != d2.columns[0]]
            d2_X_pb = d2_X_pb.iloc[:,:t]
            d2_X_pb.columns = ['f'+str(ti) for ti in range(d2_X_pb.shape[1])]
            d2_X_pb.to_csv(path2_s,sep='\t', header=None, index=False)



            d3_X_pb = d3.loc[:, d3.columns != d3.columns[0]]
            d3_X_pb = d3_X_pb.iloc[:,:t]
            d3_X_pb.columns = ['f'+str(ti) for ti in range(d3_X_pb.shape[1])]
            d3_X_pb.to_csv(path3_s,sep='\t', header=None, index=False)



            d4_X_pb = d4.loc[:, d4.columns != d4.columns[0]]
            d4_X_pb = d4_X_pb.iloc[:,:t]
            d4_X_pb.columns = ['f'+str(ti) for ti in range(d4_X_pb.shape[1])]
            d4_X_pb.to_csv(path4_s,sep='\t', header=None, index=False)



################################################################################################################## 
##################################################### STEP 2 #####################################################
################################################################################################################## 


def train(arguments):
    k,dataset = arguments
    print('progress train  : ', k/len(datasets))
    train = pd.read_csv(folderRealData+'/'+dataset+'/'+ dataset + '_TRAIN_CLASSIFIERS.tsv', sep='\t', header=None, index_col=None)
    y_train = train.iloc[:, 0].values

    max_t = train.shape[1]-1
    del train
    step = int(max_t*sampling_ratio) if int(max_t*sampling_ratio)>0 else 1
    timestamps= [t for t in range(step, max_t+1, step)]
    clf_xgbs = {}
    for t in timestamps:

        path_train = folderRealData+'/'+dataset+'/'+ dataset + 'train_features_'+str(t)+'.tsv'
        train_feat_X = pd.read_csv(path_train, sep='\t', header=None, index_col=None)
        clf_xgb = XGBClassifier().fit(train_feat_X, y_train)
        clf_xgbs[t] = clf_xgb

    with open('classifiersRealxgb/classifier'+dataset+'.pkl', 'wb') as inp:
        pickle.dump(clf_xgbs, inp)

Parallel(n_jobs=nb_core)(delayed(train)([k,dataset]) for k,dataset in enumerate(datasets))





################################################################################################################## 
##################################################### STEP 3 #####################################################
################################################################################################################## 

# make predictions
for k,dataset in enumerate(datasets):
    print('progress preds  : ', k/len(datasets))

    print('DATASET ::::::::::::::::: ', dataset)
    with open('classifiersRealxgb/classifier'+dataset+'.pkl', 'rb') as inp:
        clf_xgbs = pickle.load(inp)

    test_preds = {}
    test_probas = {}
    val_preds = {}
    val_probas = {}
    ep_preds = {}
    ep_probas = {}

    timestamps= clf_xgbs.keys()
    for t in timestamps:

        print(t)
        path1 = folderRealData+'/'+dataset+'/'+ dataset + 'test_features_'+str(t)+'.tsv'
        path2 = folderRealData+'/'+dataset+'/'+ dataset + 'val_features_'+str(t)+'.tsv'
        path3 = folderRealData+'/'+dataset+'/'+ dataset + 'ep_features_'+str(t)+'.tsv'

        d1 = pd.read_csv(path1, sep='\t', header=None, index_col=None)
        d2 = pd.read_csv(path2, sep='\t', header=None, index_col=None)
        d3 = pd.read_csv(path3, sep='\t', header=None, index_col=None)

        path1 = folderRealData+'/'+dataset+'/'+  'test_preds_'+str(t)+'.pkl'
        path2 = folderRealData+'/'+dataset+'/'+  'val_preds_'+str(t)+'.pkl'
        path3 = folderRealData+'/'+dataset+'/'+  'ep_preds_'+str(t)+'.pkl'
        path4 = folderRealData+'/'+dataset+'/'+  'test_probas_'+str(t)+'.pkl'
        path5 = folderRealData+'/'+dataset+'/'+  'val_probas_'+str(t)+'.pkl'
        path6 = folderRealData+'/'+dataset+'/'+  'ep_probas_'+str(t)+'.pkl'

        preds_test = clf_xgbs[t].predict(d1)

        assert len(preds_test) == d1.shape[0]

        test_preds[t] = preds_test
        probas_test = clf_xgbs[t].predict_proba(d1)[:,1]
        assert len(probas_test) == d1.shape[0]

        test_probas[t] = probas_test

        preds_val = clf_xgbs[t].predict(d2)
        assert len(preds_val) == d2.shape[0]
        val_preds[t] = preds_val
        probas_val = clf_xgbs[t].predict_proba(d2)[:,1]
        assert len(preds_val) == d2.shape[0]
        val_probas[t] = probas_val


        preds_ep = clf_xgbs[t].predict(d3)
        assert len(preds_ep) == d3.shape[0]
        ep_preds[t] = preds_ep
        probas_ep = clf_xgbs[t].predict_proba(d3)[:,1]
        assert len(probas_ep) == d3.shape[0]
        ep_probas[t] = probas_ep

        with open(path1, 'wb') as inp:
            pickle.dump(preds_test, inp)
        with open(path2, 'wb') as inp:
            pickle.dump(preds_val, inp)
        with open(path3, 'wb') as inp:
            pickle.dump(preds_ep, inp)
        with open(path4, 'wb') as inp:
            pickle.dump(probas_test, inp)
        with open(path5, 'wb') as inp:
            pickle.dump(probas_val, inp)
        with open(path6, 'wb') as inp:
            pickle.dump(probas_ep, inp)

    with open(folderRealData+'/'+dataset+'/'+  'test_preds.pkl', 'wb') as inp:
        pickle.dump(test_preds, inp)
    with open(folderRealData+'/'+dataset+'/'+  'test_probas.pkl', 'wb') as inp:
        pickle.dump(test_probas, inp)
    with open(folderRealData+'/'+dataset+'/'+  'ep_preds.pkl', 'wb') as inp:
        pickle.dump(ep_preds, inp)
    with open(folderRealData+'/'+dataset+'/'+  'ep_probas.pkl', 'wb') as inp:
        pickle.dump(ep_probas, inp)
    with open(folderRealData+'/'+dataset+'/'+  'val_preds.pkl', 'wb') as inp:
        pickle.dump(val_preds, inp)
    with open(folderRealData+'/'+dataset+'/'+  'val_probas.pkl', 'wb') as inp:
        pickle.dump(val_probas, inp)

