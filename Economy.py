from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import clone
import multiprocessing
try:
    import cPickle as pickle
except ImportError:
    import pickle
import json

class Economy(ABC):

    """
    Abstract class Economy

    ATTRIBUTES :

        - misClassificationCost : function that takes the label and the prediction as parameters
                                  and computes the cost of misclassification.
        - timeCost   : function that takes t as parameter and computes the cost of delaying the decision.
        - min_t      : time step from which we can make a decision.
        - max_t      : maximum length of time series.
        - classifier : classifier chosen.
        - classifiers: dictionary of classifiers for each time step.
        - P_yhat_y   : dictionary of conditional probabilities P(y_hat/y,ck) for each time step.

    """

    @abstractmethod
    def __init__(self, misClassificationCost, timeCost, min_t, classifier):
        self.misClassificationCost = misClassificationCost
        self.timeCost = timeCost
        self.min_t = min_t
        self.classifier = classifier
        self.classifiers = {}
        self.P_yhat_y = {}



    def fit_classifiers(self, X_train, Y_train, starting_timestep, usePretrainedClassifiers, path=None):

        """
        This function trains the classifiers for each time step

        INPUTS :
            - X : Independent variables
            - Y : Dependent variable
            - starting_timestep : timestep from which we start to train classifiers

        """



        # load or save the classifiers
        if not usePretrainedClassifiers:
            ## Train classifiers for each time step
            for t in self.timestamps:

                # use the same type classifier for each time step
                classifier_t = clone(self.classifier)
                # fit the classifier
                classifier_t.fit(X_train.iloc[:, :t], Y_train)
                # save it in memory
                self.classifiers[t] = classifier_t

            with open(path, 'wb') as output:
                pickle.dump(self.classifiers, output)
        else:

            if self.fears:
                self.classifiers = {}
                for t in self.timestamps:
                    self.classifiers[t] = path+str(t)+'.pkl'
            else:
                with open(path+'.pkl', 'rb') as inp:
                    self.classifiers = pickle.load(inp)




    def predict(self, X_test, donnes=None):
        """
        This function predicts for every time series in X_test the optimal time
        to make the prediction and the associated label.

        INPUTS :
            - X_test : Independent variables to test the model

        OUTPUTS :
            - predictions : list that contains [label, tau*] for every
                            time series in X_test.

        """
        nb_observations, _ = X_test.shape
        predictions = []
        if donnes != None:
            test_probas, test_preds = donnes

        # We predict for every time series [label, tau*]



        for i in range(nb_observations):
            # The approach is non-moyopic, for each time step we predict the optimal
            # time to make the prediction in the future.
            for t in self.timestamps:
                # first t values of x
                #x = np.array(list(X_test.iloc[i, :t]))
                x = X_test.iloc[i, :t].values
                # compute cost of future timesteps (max_t - t)
                if self.feat:
                    probass = test_probas[t]
                    proba = probass[i]
                    send_alert, cost = self.forecastExpectedCost(x,proba)
                else:
                    send_alert, cost = self.forecastExpectedCost(x)
                if send_alert:
                    if self.fears:
                        predictions.append([t, cost, self.handle_my_classifier(t,self.transform_to_format_fears(x.reshape(1, -1)))[0]])
                    elif self.feat:
                        predictions.append([t, cost, test_preds[t][i]])
                    else:
                        predictions.append([t, cost, self.classifiers[t].predict(x.reshape(1, -1))[0]])
                    break

        return predictions

    def handle_my_classifier(self, t, inputX, proba=False):
        with open(self.classifiers[t], 'rb') as inp:
             clf = pickle.load(inp)
        if proba:
            return clf.predict_proba(inputX)
        else:
            return clf.predict(inputX)


    def predict_post_tau_stars(self, X_test, donnes=None):
        """
        This function predicts for every time series in X_test the optimal time
        to make the prediction if we had all the values available

        INPUTS :
            - X_test : Independent variables to test the model

        OUTPUTS :
            - tau_post_star_s : list that contains [label, tau*] for every
                            time series in X_test.

        """
        #nbObs
        nb_observations, _ = X_test.shape
        tau_post_star_s = []
        # We predict for every time series [label, tau*]
        if donnes != None:
            test_probas, test_preds = donnes
        # We predict for every time series [label, tau*]

        for i in range(nb_observations):
            post_costs = []
            timestamps_pred = []
            for t in self.timestamps:
                x = X_test.iloc[i, :t].values
                if self.feat:
                    probass = test_probas[t]
                    proba = probass[i]
                    send_alert, cost = self.forecastExpectedCost(x,proba)
                else:
                    send_alert, cost = self.forecastExpectedCost(x)
                post_costs.append(cost)
                timestamps_pred.append(t)
            tau_post_star = timestamps_pred[np.argmin(post_costs)]
            x = X_test.iloc[i, :tau_post_star].values
            #tau_post_star_s.append([self.classifiers[tau_post_star].predict(x.reshape(1, -1))[0], tau_post_star, post_costs[tau_post_star-self.min_t]])
            if self.fears:
                tau_post_star_s.append([tau_post_star, np.min(post_costs), self.handle_my_classifier(tau_post_star, self.transform_to_format_fears(x.reshape(1, -1)))[0]])
            elif self.feat:
                tau_post_star_s.append([tau_post_star, np.min(post_costs), test_preds[tau_post_star][i]])
            else:
                tau_post_star_s.append([tau_post_star, np.min(post_costs), self.classifiers[tau_post_star].predict(x.reshape(1, -1))[0]])

        return tau_post_star_s

    def predict_optimal_algo(self, X_test, y_test, donnes=None):
        """
        This function predicts for every time series in X_test the optimal time
        to make the prediction if we had all the values available (Optimal algorithm)

        INPUTS :
            - X_test : Independent variables to test the model

        OUTPUTS :
            - tau_post_star_s : list that contains [label, tau*] for every
                            time series in X_test.
        """

        #nbObs
        nb_observations, _ = X_test.shape
        tau_post_star_s = []
        tau_post_star_s_feat = np.zeros((nb_observations,3))
        # We predict for every time series [label, tau*]


        if not self.feat:
            for i in range(nb_observations):
                tau_star = self.timestamps[-1]
                cost_tau_star = self.timeCost[self.timestamps[-1]] + self.misClassificationCost[0][1]

                for t in self.timestamps:
                    x = X_test.iloc[i, :t].values
                    if self.fears:
                        pred = self.handle_my_classifier(t, self.transform_to_format_fears(x.reshape(1, -1)))[0]
                    else:
                        pred = self.classifiers[t].predict(x.reshape(1, -1))[0]
                    if (pred == y_test[i]):
                        if (self.timeCost[self.timestamps[0]] + self.misClassificationCost[0][1] > self.timeCost[t]):
                            tau_star = t
                            cost_tau_star = self.timeCost[t]
                        else:
                            tau_star = self.timestamps[0]
                            cost_tau_star = self.timeCost[self.timestamps[0]] + self.misClassificationCost[0][1]
                        if self.fears:
                            tau_post_star_s.append([tau_star, cost_tau_star, self.handle_my_classifier(t, self.transform_to_format_fears(x.reshape(1, -1)))[0]])
                        else:
                            tau_post_star_s.append([tau_star, cost_tau_star, self.classifiers[t].predict(x.reshape(1, -1))[0]])
                        break
                    if (t == self.timestamps[-1]):
                        if self.fears:
                            tau_post_star_s.append([tau_star, cost_tau_star, self.handle_my_classifier(t, self.transform_to_format_fears(x.reshape(1, -1)))[0]])
                        else:
                            tau_post_star_s.append([tau_star, cost_tau_star, self.classifiers[t].predict(x.reshape(1, -1))[0]])
                        break
            return tau_post_star_s
        else:


            for t in self.timestamps:

                tau_star = self.timestamps[-1]
                cost_tau_star = self.timeCost[self.timestamps[-1]] + self.misClassificationCost[0][1]


                for i in range(nb_observations):

                    x = X_test.iloc[i, :t].values

                    pred = donnes[t][i]

                    if (pred == y_test[i]):
                        if (self.timeCost[self.timestamps[0]] + self.misClassificationCost[0][1] > self.timeCost[t]):
                            tau_star = t
                            cost_tau_star = self.timeCost[t]
                        else:
                            tau_star = self.timestamps[0]
                            cost_tau_star = self.timeCost[self.timestamps[0]] + self.misClassificationCost[0][1]
                        tau_post_star_s_feat[i][0] = tau_star
                        tau_post_star_s_feat[i][1] = cost_tau_star
                        tau_post_star_s_feat[i][2] = donnes[tau_star][i]
                        break
                    if (t == self.timestamps[-1]):
                        tau_post_star_s_feat[i][0] = tau_star
                        tau_post_star_s_feat[i][1] = cost_tau_star
                        tau_post_star_s_feat[i][2] = donnes[t][i]
                        break


            return list(map(list, tau_post_star_s_feat))
