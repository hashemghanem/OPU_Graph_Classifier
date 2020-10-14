import numpy as np 
from matplotlib import pyplot as plt
import matplotlib.ticker as mtic
from scipy import stats
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from RF_modules.sampling import graph_sampler
from RF_modules.RFMapping import *
from RF_modules.Datasets import dataset_loading
from RF_modules.graph_representation import graphlet_avg_features
from RF_modules.sampling import graph_sampler

'''
In this code, we do the following:
    1. setting up the random feature mapping parameters.
    2. choosing the graphlet sampler we want.
    3. we vary the value of inter-class similarity parameter r, and for each 
    value we:
        1. generate the correspondent SBM-based dataset (training/testing).
        2. calculate a representation vector for each graph in it.
        3. Train an SVM (linear kernel) classifier on the  resulted dataset
        4. Evaluate the trained model on the new testing dataset
    4. Plot the accuracy curve
'''


# the solution: model selection
def run_grid(z_train, z_test, y_train, y_test, C_range = 10. ** np.arange(-2, 6)):       
    param_grid = dict(C=C_range)
    grid = GridSearchCV(SVC(kernel='linear', gamma='auto'),
                        param_grid=param_grid, cv=StratifiedKFold())
    print('Fit...')
    grid.fit(z_train, y_train)
    # Training error
    y_pred = grid.predict(z_test)

    # Computes and prints the classification accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", str(round(acc*100, 2)) + "%")
    return acc




# setting up the parameters k, m , s, r, S_k
k, features_num,samples_num= 5 ,5000, 1           
r= 1+np.linspace(0.2,2,4)
sampler_type= "simple_random_sampling"                      

# creating an instance of the required sampler
sampler=graph_sampler(sampler_type, k) 

# creating an instance of the required feature mapping class (OPU RFs)
feat_map=Lighton_random_features(k**2, features_num)       
                                                          
accur=np.zeros(len(r))
for (f_ind, factor) in enumerate(r):
    print('Processing r={}, Remaining experiments: {}/{}'.format(factor,len(r)-f_ind-1,len(r)))
    # generate a new synthetic dataset (SBM generator)
    (G_train,y_train),(G_test,y_test) = dataset_loading().generate_SBM(r= factor) 
    # calculating a representation vector z for each graph 
    graphletRF = graphlet_avg_features(samples_num, sampler, feat_map, batch_size=None, verbose=True)
    z_train=graphletRF.apply(G_train)
    z_test=graphletRF.apply(G_test)
    #Training, then evaluating a linear SVM model on the training/testing data
    accur[f_ind]= run_grid(z_train.T, z_test.T, y_train, y_test)
    # cutting access to the OPU
    feat_map.close()

fig, ax = plt.subplots()
fmt='%.0f%%'
yticks=mtic.FormatStrFormatter(fmt)
ax.yaxis.set_major_formatter(yticks)
plt. plot(r, accur, 's', linewidth=2.8)
plt.xlabel('Inter-class similarity parameter (r)')
plt.ylabel('Test accuracy')
plt.grid()
plt.show()