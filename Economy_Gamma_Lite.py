import numpy as np
from scipy.spatial.distance import cdist
from Economy import Economy
from sklearn.base import clone
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from utils import generatePossibleSequences
import os
import json
import multiprocessing
try:
    import cPickle as pickle
except ImportError:
    import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from utils import transform_to_format_fears, numpy_to_df
import json



class Economy_Gamma_Lite(Economy):

    """
    Economy_Gamma inherits from Economy

    ATTRIBUTES :

        - nbIntervals : number of intervals.
        - order       : order of marcov chain.
        - thresholds  : dictionary of thresholds for each time step.
        - transitionMatrices   : transition matrices for each sequence (t,t+1).
        - complexTransitionMatrices : transition matrices for each sequence (t-ordre..t,t+1).
        - indices     : indices of data associated to each time step and each interval
        - labels     : list of labels observed on the data set.
    """
    def __init__(self, misClassificationCost, timeCost, min_t, classifier, nbIntervals, order, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset , feat):
        super().__init__(misClassificationCost, timeCost, min_t, classifier)
        self.nbIntervals = nbIntervals
        self.order = order
        self.thresholds = {}
        self.P_tplusTau_gamma_k = {}
        self.indices = {}
        self.P_y_gamma = {}
        self.use_complete_ECO_model = use_complete_ECO_model
        self.pathECOmodel = pathECOmodel
        self.P_yhat_y = {}
        self.sampling_ratio = sampling_ratio
        self.fears = fears
        self.dataset = dataset
        self.feat = feat


    def fit(self, train_classifiers, estimate_probas, ratioVal, path, usePretrainedClassifs=True):

        """
           This procedure fits our model Economy

           INPUTS :
                X : Independent variables
                Y : Dependent variable

           OUTPUTS :
                self.thresholds : dictionary of thresholds for each time step.
                self.transitionMatrices   : transition matrices for each sequence (t,t+1).
                self.P_yhat_y : confusion matrices
                self.indices  : indices associated to each time step and each interval
        """


        self.max_t = train_classifiers.shape[1] - 1
        step = int(self.max_t*self.sampling_ratio) if int(self.max_t*self.sampling_ratio)>0 else 1
        self.timestamps = [t for t in range(step, self.max_t + 1, step)]

        if self.use_complete_ECO_model:

            with open(self.pathECOmodel, 'rb') as input:
                fullmodel = pickle.load(input)



            for t in self.timestamps:
                self.classifiers[t] = fullmodel.classifiers[t]
            self.labels = fullmodel.labels
            # iterate over all time steps 1..max_t
            for t in self.timestamps:
                #compute thresholds and indices
                self.thresholds[t] = fullmodel.thresholds[t]
                for interval in range(1, self.nbIntervals+1):
                    self.P_yhat_y[str(t)+str(interval)] = fullmodel.P_yhat_y[str(t)+str(interval)]
                    self.P_y_gamma[str(t)+str(interval)] = fullmodel.P_y_gamma[str(t)+str(interval)]

            self.recodedTS = fullmodel.recodedTS

        else:


            ## Split into train and val
            Y_train = train_classifiers.iloc[:, 0]
            X_train = train_classifiers.loc[:, train_classifiers.columns != train_classifiers.columns[0]]
            Y_val = estimate_probas.iloc[:, 0]
            X_val = estimate_probas.loc[:, estimate_probas.columns != estimate_probas.columns[0]]

            # labels seen on the data set
            self.labels = Y_train.unique()


            # time step to start train classifiers
            starting_timestep = self.min_t-self.order+1 if self.min_t-self.order>=0 else 1

            # fit classifiers
            self.fit_classifiers(X_train, Y_train, starting_timestep, usePretrainedClassifs, path)

            #



            # iterate over all time steps 1..max_t
            for t in self.timestamps:
                #compute thresholds and indices
                self.thresholds[t] = self.computeThresholdsAndindices(X_val.iloc[:, :t])

            self.recodeTS(X_val)
            for t in self.timestamps:

                for interval in range(1, self.nbIntervals+1):
                    indices_x_t = self.recodedTS[self.recodedTS[t-1]==interval].index.values
                    X_val_x_t = X_val.loc[indices_x_t, :]
                    Y_val_x_t = Y_val.loc[indices_x_t]
                    self.P_yhat_y[str(t)+str(interval)] = self.compute_P_yhat_y_gammak(X_val_x_t, Y_val_x_t, t)
                    self.P_y_gamma[str(t)+str(interval)], _ = self.compute_P_y_gamma(Y_val_x_t, t)




    # TODO : vectorize the procedure
    def recodeTS(self, X_val):
        X_val_new = X_val.copy()
        nb_observations, _ = X_val.shape

        if not self.feat:
            # We predict for every time series [label, tau*]
            for i in range(nb_observations):
                x = X_val.iloc[i, :].values
                intervals = self.findIntervals(x, self.max_t)
                for j in range(len(intervals)):
                    X_val_new.iloc[i, j] = intervals[j]
            X_val_new = X_val_new.astype(int)
        else:
            for t in self.timestamps:
                with open('RealData/'+self.dataset+'/ep_probas_'+str(t)+'.pkl' ,'rb') as inp:
                    X_val_t = pickle.load(inp)

                X_val_new.at[:,t] = self.findIntervalDataset(X_val_t,t)

        self.recodedTS = X_val_new
        self.recodedTS.columns = [i for i,col in enumerate(self.recodedTS.columns)]




    def findIntervalDataset(self, X,t):
        l=[]
        for proba in X:
            l.append(self.determineInterval(t, proba))
        return l



    def determineInterval(self, t_current, proba):
        # search for interval given probability
        ths = self.thresholds[t_current]
        for i,e in enumerate(sorted(ths, reverse=False)):
            if (proba <= e):
                return self.nbIntervals - i
        return 1


    def computeThresholdsAndindices(self, X_val):

        """
           This procedure computes thresholds and indices of data associatied
           to each interval.

           INPUTS :
                X_val : validation data

           OUTPUTS :
                thresholds : dictionary of thresholds for each time step.
                indices  : indices associated to each time step and each interval
        """

        _, t = X_val.shape
        # Predict classes
        if self.fears:
            predictions = self.handle_my_classifier(t, transform_to_format_fears(X_val), proba=True)
            predictions = predictions.values # todo ProbNone1
        elif self.feat:
            with open('RealData/'+self.dataset+'/ep_probas_'+str(t)+'.pkl' ,'rb') as inp:
                predictions = pickle.load(inp)
        else:
            predictions = self.classifiers[t].predict_proba(X_val)
        # Sort according to probabilities
        if self.feat:
            sortedProbabilities = [(i,val) for i,val in zip(np.argsort(predictions)[::-1], sorted(predictions, reverse=True))]
        else:
            sortedProbabilities = [(i,val) for i,val in zip(np.argsort(predictions[:, 1])[::-1], sorted(predictions[:, 1], reverse=True))]
        # equal frequence
        frequence = len(sortedProbabilities) // self.nbIntervals
        #compute thresholds
        thresholds = []
        for i in range(1, self.nbIntervals):
            thresholds.append(sortedProbabilities[i*frequence][1])
        return thresholds


    def compute_P_yhat_y_gammak(self, X_val, Y_val, timestep):
        """
           This function computes P_t(ŷ/y,c_k)


           INPUTS :
                X_val, Y_val : valdiation data
                timestep     : timestep reached
                indicesData  : indices of data associated to each interval / timestep

           OUTPUTS :
                probabilities : P_t(ŷ/y,gamma_k)

        """

        occurences = {}

        # initialise probabilities to 0
        probabilities = {(gamma_k, y, y_hat):0 for y in self.labels for y_hat in self.labels for gamma_k in range(self.nbIntervals)}
        rec = self.recodedTS.loc[X_val.index.values,:]
        # Iterate over intervals
        for gamma_k in range(self.nbIntervals):

            indices_gamma_k = rec[rec[timestep-1]==gamma_k+1].index.values
            # Subset of Validation set in interval gamma_k


            X_val_ck = X_val.loc[indices_gamma_k,:]
            Y_val_ck = Y_val.loc[indices_gamma_k]
            # Subset of Validation set in interval gamma_k
            if (len(Y_val_ck)>0):
                if self.fears:
                    predictions = self.handle_my_classifier(timestep, transform_to_format_fears(X_val_ck.iloc[:, :timestep]))
                elif self.feat:
                    with open('RealData/'+self.dataset+'/ep_preds_'+str(timestep)+'.pkl' ,'rb') as inp:
                        predictions = pickle.load(inp)
                        predictions = [predictions[ii] for ii in indices_gamma_k]
                else:
                    predictions = self.classifiers[timestep].predict(X_val_ck.iloc[:, :timestep])
                for y_hat, y in zip(predictions, Y_val_ck):
                    # frequenceuence
                    probabilities[gamma_k, y, y_hat] += 1
        # normalize
        for gamma_k, y, y_hat in probabilities.keys():
            indices_gamma_k = rec[rec[timestep-1]==gamma_k+1].index.values
            Y_val_gamma = Y_val.loc[indices_gamma_k]

            # number of observations in gammak knowing y
            sizeCluster_gamma = len(Y_val_gamma[Y_val_gamma==y])
            try:
                if (sizeCluster_gamma != 0):
                    probabilities[gamma_k, y, y_hat] /= sizeCluster_gamma
            except ZeroDivisionError:
                print("Zero")
        return probabilities




    def compute_P_y_gamma(self, Y_val, timestep):
        """
           This function computes P_t(y|gamma_k)


           INPUTS :
                X_val, Y_val : valdiation data
                timestep     : timestep reached
                indicesData  : indices of data associated to each interval / timestep
           OUTPUTS :
                probabilities : P_t(ŷ/y,gamma_k)

        """

        # Initialize all probabilities with 0
        probabilities = {(gamma_k, y):0 for y in self.labels for gamma_k in range(self.nbIntervals)}
        occ = {gamma_k:0 for gamma_k in range(self.nbIntervals)}
        rec = self.recodedTS.loc[Y_val.index.values]
        indicesData = [rec[rec[timestep-1]==gamma_k+1].index.values for gamma_k in range(self.nbIntervals)]
        somme = 0
        for gamma_k,e in enumerate(indicesData):
            somme += len(e)
            occ[gamma_k] = len(e)
            for ts in e:
                probabilities[gamma_k, Y_val.loc[ts]] += 1
            try:
                if len(e) != 0:
                    for y in self.labels:
                        probabilities[gamma_k, y] /= len(e)
            except ZeroDivisionError:
                print("Zero")

        for gamma_k in range(self.nbIntervals):
            try:
                if somme != 0:
                    occ[gamma_k] /= somme
            except ZeroDivisionError:
                print("Zero")


        return probabilities, occ


    def filterIndices(self, indicesTofilter, filter):

        results = []

        for l in indicesTofilter:
            element = []
            for e in l:
                if e in filter:
                    element.append(e)
            results.append(element)
        return results

    def forecastExpectedCost(self, x_t, pb=None):
        """
           This function computes expected cost for future time steps given
           a time series xt


           INPUTS :
                x_t : time series

           OUTPUTS :
                totalCosts : list of (max_t - t) values that contains total cost
                             for future time steps.
        """
        t_current = len(x_t)

        # we initialize total costs with time cost
        forecastedCosts = [self.timeCost[t] for t in self.timestamps[self.timestamps.index(t_current):]]

        send_alert = True
        if self.feat:
            interval = self.findInterval(x_t, pb)
        else:
            interval = self.findInterval(x_t)

        for i,t in enumerate(self.timestamps[self.timestamps.index(t_current):]):
            # Compute p_transition
            # compute confusion matricies

            #iterate over intervals
            for gamma_k in range(self.nbIntervals):
                #iterate over possible labels
                for y in self.labels:
                    # iterate over possible predictions
                    for y_hat in self.labels:
                        forecastedCosts[i] +=  self.P_yhat_y[str(t)+str(interval)][gamma_k, y, y_hat] * self.P_y_gamma[str(t)+str(interval)][gamma_k, y] * self.misClassificationCost[int(y_hat)][int(y)]

            if (i>0):
                if (forecastedCosts[i] < forecastedCosts[0]):
                    send_alert = False
                    break
        return send_alert, forecastedCosts[0]



    def findInterval(self, x_t, pb=None):
        """
           This function finds interval associated with a timeseries given its
           probability


           INPUTS :
                proba : probability given by the classifier at time step t

           OUTPUTS :
                interval of x_t
        """
        # we could use binary search for better perf
        t_current = len(x_t)
        # predict probability
        if self.fears:
            probadf = self.handle_my_classifier(t_current, transform_to_format_fears(numpy_to_df(x_t)), proba=True)
            proba = probadf['ProbNone1'].values[0]
        elif self.feat:
            proba = pb
        else:
            probadf = self.classifiers[t_current].predict_proba(x_t.reshape(1, -1))
            proba = probadf[0][1] # a verifier





        # search for interval given probability
        ths = self.thresholds[t_current]
        for i,e in enumerate(sorted(ths, reverse=False)):
            if (proba <= e):
                return self.nbIntervals - i
        return 1
        # nbInt = 3
        # 1
        # 2
        # 3



    def findIntervals(self, x_t, nbSteps):
        """
           This function finds intervals associated with a timeseries given its
           probabilities for last nbSteps values


           INPUTS :
                proba : probability given by the classifier at time step t

           OUTPUTS :
                interval of x_t
        """
        intervals = np.zeros(nbSteps)

        for t in self.timestamps:
            intervals[t-1] = self.findInterval(x_t[:t])
        return intervals
