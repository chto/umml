import numpy as np
import glob 
import matplotlib.pyplot as plt
import pickle
from helpers import SimulationAnalysis
import pandas as pd 
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
import seaborn as sns 
total_cat_mergedmask = pd.read_csv("./data/merged_vpeak_150.cache")
features = ['Mpeak','Macc', 'Vpeak', 'Vacc','Spin','upid','mvir','Halfmass_Scale']
H = total_cat_mergedmask[features]
M = total_cat_mergedmask[['obs_SM']]
training_size = 0.8
H_train, H_test, M_train, M_test = cross_validation.train_test_split(H, M, train_size=training_size, random_state=23)
#Hyperparameters to try:
#parameters = {'n_estimators':(100,500,750,1000), "max_features": ["auto"], "min_samples_leaf": [1,2,4]}
parameters = {"max_features": ["auto"],"min_samples_leaf": [4],"max_depth":(5,10)}

# Do a grid search to find the highest n-fold cross-validation score:
n = 5
rf_tuned = GridSearchCV(RandomForestRegressor(n_estimators=500), parameters, cv=n, verbose=1)
RFselector = rf_tuned.fit(H_train, M_train)
pickle.dump(rf_tuned, open("./model/rf_tuned_test.pkl", "wb"))
pickle.dump(RFselector, open("./model/RFselector_test.pkl", "wb"))
#"""
#rf = pickle.load(open(filename, 'r'))
