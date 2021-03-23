#!/usr/bin/env bash


# exctract features, learn classifiers, and save predictions
python extract_learn_store.py


# save models ECONOMY and predictions (adaptive)
python test.py --pathToInputParams experiments/experiment1/inputParams.json --pathToOutputs ./experiments/experiment1/


# save models ECONOMY and predictions by forcing nbGourps = 1 (non adaptive)
python test.py --pathToInputParams experiments/experiment2/inputParams.json --pathToOutputs ./experiments/experiment2/


# grid search on the SR approach and making predictions on test set (the results are already available in the folder ./experiments/experiment_SR_approach)
# if you wan te reperform the grid search and predictions decomment the line bellow
#python train_optimize_predict_SR_approach.py


# statistical tests and plots available at experiments/experiment1/PlotsPaper after running the following command
python reproduce_plots_tests.py