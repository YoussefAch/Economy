
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist
from Economy import Economy
from sklearn.base import clone
from scipy.linalg import norm
import multiprocessing
from joblib import Parallel, delayed
try:
    import cPickle as pickle
except ImportError:
    import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from utils import transform_to_format_fears
import json



class Economy_K_multiClustering(Economy):

    """
    Economy_K inherits from Economy

    ATTRIBUTES :

        - nbClusters : number of clusters.
        - lmbda      : parameter of our model estimating the membership probability
                       to a given cluster
        - clustering : clustering model
        - clusters   : list of clusters
        - P_y_ck     : prior probabilities of a label y given a cluster.
        - labels     : list of labels observed on the data set.

    """

    def __init__(self, misClassificationCost, timeCost, min_t, classifier, nbClusters, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset , feat):
        super().__init__(misClassificationCost, timeCost, min_t, classifier)
        self.nbClusters = nbClusters
        self.P_y_ck = {}
        self.use_complete_ECO_model = use_complete_ECO_model
        self.pathECOmodel = pathECOmodel
        self.sampling_ratio = sampling_ratio
        self.fears = fears
        self.dataset = dataset
        self.feat = feat

    def clusteringTimestamp(self, X_train):
        clustering = KMeans(n_clusters=np.min([self.nbClusters,X_train.shape[0]]), init="k-means++", algorithm="elkan")
        clustering.fit(X_train)
        return clustering




    def clusteringTrainset(self, X_train):
        """
           This procedure performs a clustering of our train set

           INPUTS :
                X_train : train set

           OUTPUTS :
                self.clustering : clustering model.
                self.clusters : list of clusters.
        """
        ## Identify a finite set of clusters (Kmeans)
        mx_t = X_train.shape[1]
        self.clustering = []
        for t in self.timestamps:
            self.clustering.append(self.clusteringTimestamp(X_train.iloc[:,:t]))
        self.clusters = range(np.min([self.nbClusters,X_train.shape[0]]))




    def fit(self, train_classifiers, estimate_probas, ratioVal, path, usePretrainedClassifs=True):
        """
           This procedure fits our model Economy

           INPUTS :
                X : Independent variables
                Y : Dependent variable

           OUTPUTS :
                self.P_yhat_y : dictionary which contains for every time step the confusion matrix
                                associated to a given cluster at time step t.
                self.P_y_ck   : prior probabilities of a label y given a cluster.
        """
        self.max_t = train_classifiers.shape[1] - 1

        step = int(self.max_t*self.sampling_ratio) if int(self.max_t*self.sampling_ratio)>0 else 1
        self.timestamps = [t for t in range(step, self.max_t + 1, step)]

        if self.use_complete_ECO_model:

            with open(self.pathECOmodel, 'rb') as input:
                fullmodel = pickle.load(input)

            self.clusters = fullmodel.clusters
            self.clustering = fullmodel.clustering

            for t in self.timestamps:
                self.classifiers[t] = fullmodel.classifiers[t]
            self.labels = fullmodel.labels

            for t in self.timestamps:
                self.P_yhat_y[t] = fullmodel.P_yhat_y[t]
            self.P_y_ck = fullmodel.P_y_ck


        else:

            ## Split into train and val
            Y_train = train_classifiers.iloc[:, 0]
            X_train = train_classifiers.loc[:, train_classifiers.columns != train_classifiers.columns[0]]
            Y_val = estimate_probas.iloc[:, 0]
            X_val = estimate_probas.loc[:, estimate_probas.columns != estimate_probas.columns[0]]

            # labels seen on the data set
            self.labels = Y_train.unique()

            # perform clustering
            self.clusteringTrainset(X_train)

            # fit classifiers
            self.fit_classifiers(X_train, Y_train, self.min_t, usePretrainedClassifs, path)


            # Compute probabilities (confusion matricies) for each time step
            for t in self.timestamps:
                self.P_yhat_y[t] = self.compute_P_yhat_y_ck(X_val, Y_val, t)

                # Compute prior probabilities given a cluster
                self.P_y_ck[t] = self.compute_P_y_ck(X_val, Y_val, t)


    def compute_P_ck_xt(self, x_t, clusters):
        """
           This function computes membership probablities to a cluster given a sequence
           of values.

           INPUTS :
                x_t      : sequence of values (time series).
                clusters : centers of our clusters.
                lmbda    : parameter of the logistic function

           OUTPUTS :
                probabilities : list of nbClusers values which contains the probabilty
                                of membership to the different clusters identified.
        """
        # Compute the average of distances beween x_t and all the clusters
        # using the euclidean distance.


        # the distances between x_t and all the clusters
        distances = cdist([x_t], clusters, metric='euclidean')
        for i,e in enumerate(distances[0]):
            if e < 0.000001:
                distances[0] = 0.000001

        distances = 1./distances
        probabilities = distances / np.sum(distances)
        return probabilities[0]







    def compute_P_y_ck(self, X_train, Y_train, timestep):
        """
           This function computes prior probabilities of label y given a cluster ck


           INPUTS :
                X_train, Y_train : train data

           OUTPUTS :
                probabilities : P(y|ck) prior probabilities of label y given a cluster ck
        """

        occurences = {}
        # Initialize all probabilities with 0
        probabilities = {(c_k, y):0 for y in self.labels for c_k in self.clusters}

        # for every observation in train set
        cl = self.clustering[self.timestamps.index(timestep)].predict(X_train.iloc[:, :timestep])
        for c_k, y in zip(cl, Y_train):
            # compute frequence of (ck,y)
            probabilities[c_k, y] += 1
            # compute size of cluster ck
            occurences[c_k] = occurences.get((c_k), 0) + 1

        # normalize
        for c_k, y in probabilities.keys():
            # avoid div by 0
            if (c_k in occurences.keys()):
                probabilities[c_k, y] /= occurences[c_k]

        return probabilities







    def compute_P_yhat_y_ck(self, X_val, Y_val, timestep):
        """
           This function computes P_t(ŷ/y,c_k)


           INPUTS :
                X_val, Y_val : valdiation data
                timestep     : timestep reached

           OUTPUTS :
                probabilities : probabilities of label y given a cluster ck.
        """

        ############## INITS
        occurences = {}
        probabilities = {}
        subsets = {}


        # clusters associated to time series
        # à modifier les noms de variables & noms de fonctions (id ou clusters etc)
        clusters_data = self.clustering[self.timestamps.index(timestep)].predict(X_val.iloc[:, :timestep])

        # initialise probabilities to 0
        probabilities = {(c_k, y, y_hat):0 for y in self.labels for y_hat in self.labels for c_k in self.clusters}

        # for each cluster we associate indices of data corresponding to this cluster
        indices_data_cluster = {c_k:[] for c_k in self.clusters}

        for index, value in enumerate(clusters_data):
            # indices id ?
            indices_data_cluster[value].append(index)

        ############## OCCURENCES
        for c_k in self.clusters:

            indices_ck = indices_data_cluster[c_k]

            # Subset of Validation set in cluster C_k
            X_val_ck = X_val.iloc[indices_ck]

            if (len(indices_ck)>0):
                # predict labels for this subset
                if self.fears:
                    predictions = self.handle_my_classifier(timestep, transform_to_format_fears(X_val_ck.iloc[:, :timestep]))

                elif self.feat:
                    with open('RealData/'+self.dataset+'/ep_preds_'+str(timestep)+'.pkl' ,'rb') as inp:
                        predictions = pickle.load(inp)
                        predictions = [predictions[ii] for ii in indices_ck]
                else:
                    predictions = self.classifiers[timestep].predict(X_val_ck.iloc[:, :timestep])
                for y_hat, y in zip(predictions, Y_val.iloc[indices_ck]):
                    # compute frequence
                    probabilities[c_k, y, y_hat] += 1

        ############## NORMALIZATION KNOWING Y
        for c_k, y, y_hat in probabilities.keys():

            # subset ck
            Y_val_ck = Y_val.iloc[indices_data_cluster[c_k]]
            # number of observations in this subset that have label y
            sizeCluster_y = len(Y_val_ck[Y_val_ck==y])
            if sizeCluster_y != 0:
                probabilities[c_k, y, y_hat] /= sizeCluster_y

        return probabilities



    def forecastExpectedCost(self, x_t, index=None):
        """
           This function computes expected cost for future time steps given
           a time series xt


           INPUTS :
                x_t : time series

           OUTPUTS :
                forecastedCosts : list of (max_t - t) values that contains total cost
                             for future time steps.
        """

        t_current = len(x_t)

        # compute membership probabilites
        P_ck_xt = self.compute_P_ck_xt(x_t, self.clustering[self.timestamps.index(t_current)].cluster_centers_)
        send_alert = True
        # we initialize total costs with time cost
        forecastedCosts = [self.timeCost[t] for t in self.timestamps[self.timestamps.index(t_current):]]

        # iterate over future time steps
        for i,t in enumerate(self.timestamps[self.timestamps.index(t_current):]):
            # iterate over clusters

            for c_k, P_ck in zip(self.clusters, P_ck_xt):
                # iterate over possible labels
                for y in self.labels:
                    # iterate over possible predictions
                    for y_hat in self.labels:
                        forecastedCosts[i] += P_ck * self.P_y_ck[t][c_k, y] * self.P_yhat_y[t][c_k, y, y_hat] * self.misClassificationCost[y_hat, y]

            if (i>0):
                if (forecastedCosts[i] < forecastedCosts[0]):
                    send_alert = False
                    break

        return send_alert, forecastedCosts[0]
