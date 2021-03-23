import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Stats_tests_utils import matrix_ranking_vsSR, matrix_ranking, wilco_approche_adapt_vs_nonadapt, wilco_approches_vs_SR
import numpy as np 
import pandas as pd
try:
    import cPickle as pickle
except ImportError:
    import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# load files
# adaptive version
filename = "experiments/experiment1/df_metrics.pkl"
with open(filename, 'rb') as input:
    df_metrics_opt = pickle.load(input)

# non adaptive version
filename = "experiments/experiment2/df_metrics.pkl"
with open(filename, 'rb') as input:
    df_metrics_one = pickle.load(input)

# results of the SR approach
with open('experiments/experiment_SR_approach/df_metrics_SR.pkl', 'rb') as outfile:
    df_metrics_SR = pickle.load(outfile)

summary = pd.read_excel('RealData/DataSummaryExpanded_NoMissingVals.xlsx')





df_metrics_opt['diff_score'] = abs(df_metrics_opt['Score'] - df_metrics_opt['Score_post'])



summary = summary[['Name', 'Length']]
lengths = {}
for e in summary.values:
    lengths[e[0]] = e[1]
lengths['MixedShapesRegularTrain'] = lengths['MixedShapes']



df_metrics_opt['lengths'] = 1
def construct(row):
    return lengths[row['Dataset']]
def toInt(x):
    return int(x)
df_metrics_opt['lengths'] = df_metrics_opt.apply(lambda row: construct(row) , axis=1)


# noramlize time features 
metrics_to_normalize = ['meanTauStar','stdTauStar','meanTauPost','stdTauPost','meanTauOPT','stdTauOPT','mean_diff_tauStar_tauPost','std_diff_tauStar_tauPost', 'mean_diff_tauStar_tauOpt','std_diff_tauStar_tauOpt', 'median_tau_et', 'median_tau_post', 'median_tau_opt', 'median_diff_tau_et_post']
for m in metrics_to_normalize:
    df_metrics_opt[m] = df_metrics_opt[m]/df_metrics_opt['lengths']




methods = np.unique(df_metrics_opt['Method'])
datasets = list(np.unique(df_metrics_opt['Dataset']))

timeParams = list(np.unique(df_metrics_opt['timeParam']))

timeParams = list(map(float, timeParams))
timeParams.sort()
timeParams = list(map(str, timeParams))
timeParams.pop()
timeParams.append('1')






################################################################## Fig 3. paper ##################################################################


df_adapt_non_adapt = wilco_approche_adapt_vs_nonadapt(df_metrics_opt, df_metrics_one, 'Score', timeParams, datasets, methods)


methodds = [r'Eco-$\gamma$', r'Eco-$\gamma$-lite', 'Eco-K', 'Eco-multi-K' ]
markers = ['o','o','o','o']
plt.figure(figsize=(5,2))


for i,method in enumerate(methods):
    print(method)


    yy = (i+1) / 2 * np.array(list(map(toInt, list(df_adapt_non_adapt[df_adapt_non_adapt['method'] == method]['null_hypothesis_rejected'].values))))
    
    indicesTrue = np.argwhere(yy>0).flatten()
    indicesFalse = np.argwhere(yy<0.1).flatten()

    

    yyTrue = [yy[j] for j in indicesTrue]
    valeursTrue = [timeParams[j] for j in indicesTrue]
    yyFalse = [yy[indicesTrue[0]] for j in indicesFalse]
    valeursFalse = [timeParams[j] for j in indicesFalse]


    plt.scatter(valeursFalse, yyFalse, marker=markers[i], color='black', facecolor='white')
    plt.scatter(valeursTrue, yyTrue, marker=markers[i], color='black', label=method) 
    



plt.xlabel(r'$\alpha$', fontsize=18)
y_axis = np.arange(1/2, 2.5, 1/2)
plt.yticks(y_axis,methodds)
plt.xticks(timeParams, rotation=90)

plt.savefig('experiments/PlotsPaper/figure_3.png', bbox_inches='tight')
    
plt.close()







################################################################## Figs 4. and 5. paper ##################################################################


# first, compute values for alpha that maximizes the difference in score between 4 approaches
dataset_tempcost = {}
for dataset in datasets:
    diff = []
    for timeparam in timeParams:
        scores = list(df_metrics_opt[(df_metrics_opt['Dataset']==dataset) & (df_metrics_opt['timeParam']==timeparam)]['Score'].values)
        diff.append(max(scores) - min(scores))
    
    dataset_tempcost[dataset] = timeParams[diff.index(max(diff))]

# then we perform our test
nameFig = ['4_a','4_b']
matrix_ranking(methods, 'Score', 'petit', datasets, df_metrics_opt, dataset_tempcost, nameFig)

# earliness 
nameFig = ['5_a','5_b']
matrix_ranking(methods, 'median_tau_et', 'petit', datasets, df_metrics_opt, dataset_tempcost, nameFig)

#kappa
nameFig = ['5_c','5_d']
matrix_ranking(methods, 'kappa_star', 'grand', datasets, df_metrics_opt, dataset_tempcost, nameFig)


################################################################## Fig 6. paper ##################################################################



valuesKappa = np.zeros((4, len(timeParams)))
valuesTau = np.zeros((4, len(timeParams)))
for j,timeParam in enumerate(timeParams):
        
    for i,method in enumerate(methods):
        valuesKappa[i][j] = np.mean(df_metrics_opt[(df_metrics_opt['timeParam']==timeParam) & (df_metrics_opt['Method']==method)]['kappa_star'].values)
        valuesTau[i][j] = np.mean(df_metrics_opt[(df_metrics_opt['timeParam']==timeParam) & (df_metrics_opt['Method']==method)]['median_tau_et'].values)

fil = Line2D.fillStyles[0]

for y, fill_style in enumerate(Line2D.fillStyles):
    print(y, fill_style)
print(methods)
markers = ['d' ,'d','o','o']
alp = r'$\alpha$ = '
for k,color in enumerate(markers):
    if k%2==0:
        plt.plot(valuesTau[k,:], valuesKappa[k,:], color='grey', label=methodds[k], marker=markers[k], markerfacecolor='black')
    else:
        plt.plot(valuesTau[k,:], valuesKappa[k,:], color='grey', label=methodds[k], marker=markers[k], markerfacecolor='white')

for i in [0, 5, 8, 10, 15, 18, 26]:
    plt.annotate(alp + timeParams[i], (valuesTau[2,i]+0.005, valuesKappa[2,i]-0.01))

plt.legend()
plt.xlabel(r'$Earliness$', fontsize=16)
plt.ylabel(r'$Kappa$', fontsize=16)


plt.savefig("experiments/PlotsPaper/figure_6.png")
plt.close()



################################################################## Fig 7. paper ##################################################################
nameFig = ['7_a','7_b']
matrix_ranking(methods, 'diff_score', 'petit', datasets, df_metrics_opt, dataset_tempcost, nameFig)





################################################################## Fig 8. paper ##################################################################
#choose alpha in favor of SR approach for each dataset 
dataset_tempcost_SR = {}

for dataset in datasets:
    diff = []
    for timeparam in timeParams:
        scoreGamma = df_metrics_opt[(df_metrics_opt['Dataset']==dataset) & (df_metrics_opt['timeParam']==timeparam) & (df_metrics_opt['Method']=='Gamma')]['Score'].values[0]
        scoreSR = df_metrics_SR[(df_metrics_SR['timeParam']==float(timeparam)) & (df_metrics_SR['Dataset']==dataset)]['Score'].values
        diff.append(scoreGamma - scoreSR)
    
    dataset_tempcost_SR[dataset] = timeParams[diff.index(max(diff))]
print(dataset_tempcost_SR)
# then we perform our test
nameFig = ['8_a', '8_b']
matrix_ranking_vsSR(['Gamma'], 'Score', 'petit',  datasets, df_metrics_opt, df_metrics_SR, dataset_tempcost_SR, nameFig)







################################################################## Fig 9. paper ##################################################################
df_mori_vs_approches = wilco_approches_vs_SR(df_metrics_opt, df_metrics_SR, 'Score', timeParams, datasets, methods)



methodds = [r'Eco-$\gamma$', r'Eco-$\gamma$-lite', 'Eco-K', 'Eco-multi-K' ]
markers = ['o','o','o','o']
plt.figure(figsize=(15,2))
for i,method in enumerate(methods):


    yy = (i+1) / 2 * np.array(list(map(toInt, list(df_mori_vs_approches[df_mori_vs_approches['method'] == method]['null_hypothesis_rejected'].values))))

    yz =  np.array(df_mori_vs_approches[df_mori_vs_approches['method'] == method]['z'].values)

    
    indicesTrue = np.argwhere(yy>0).flatten()
    indicesFalse = np.argwhere(yy<0.1).flatten()

    Gagne = np.argwhere(yz>0).flatten()
    Perdu = np.argwhere(yz<0).flatten()
    indicesTrueGagne = [ind for ind in Gagne if ind in indicesTrue]
    indicesTruePerdu = [ind for ind in Perdu if ind in indicesTrue]

    yyTrue = [yy[j] for j in indicesTrue]
    valeursTrue = [timeParams[j] for j in indicesTrue]

    yyTrueGagne = [yy[j] for j in indicesTrueGagne]
    valeursTrueGagne = [timeParams[j] for j in indicesTrueGagne]
    yyTruePerdu = [yy[j] for j in indicesTruePerdu]
    valeursTruePerdu = [timeParams[j] for j in indicesTruePerdu]


    yyFalse = [yy[indicesTrue[0]] for j in indicesFalse]
    valeursFalse = [timeParams[j] for j in indicesFalse]

    
    plt.scatter(valeursTrueGagne, yyTrueGagne, marker='+', color='black') #markerfacecolor='black'
    plt.scatter(valeursTruePerdu, yyTruePerdu, marker='>', color='black')
    plt.scatter(valeursFalse, yyFalse, marker=markers[i], color='black', facecolor='white')



plt.xlabel(r'$\alpha$', fontsize=18)
y_axis = np.arange(1/2, 2.5, 1/2)
plt.yticks(y_axis,methodds)
plt.xticks(timeParams, rotation=90)
print(y_axis)

plt.savefig('experiments/PlotsPaper/figure_9.png', bbox_inches='tight')
    
plt.close()