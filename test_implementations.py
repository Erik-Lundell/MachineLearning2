import json
import numpy as np
import pandas as pd

import sklearn as skl
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats.stats import pearsonr   

import time

#Our implementations
import kNN as OurKNN
import RidgeRegression as OurRidge

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error, mean_absolute_error
#
import seaborn as sns
import matplotlib.pyplot as plt

##### Dataset 1: Wikipedia maths
'''Json structure:
{"edges":[[x,y]... array of ordered tuples of links],
"weights":[(ordered) array of weights for the edges, I guess they represent the number of times one topic is linked to another],
"node_ids":{"topic name": id, set of ids for all topics},
"time_periods":731,
"0": {"index":0, "year":2019, "month": 3, "day":16, "y":[124, 1240, 123...]},
"1":{"index":1, ...}
...
}'''

# Load data, process, potentially plot some distributions

with open('data/wikivital_mathematics.json') as data_file:    
    math_data = json.load(data_file)
    
keys = math_data.keys()

topics_id = math_data["node_ids"]
topics = math_data["node_ids"].keys()

#### Dataset 2: Energy efficiency

# Load data, process, potentially plot some distributions

df_raw = pd.read_csv('data/energy_efficiency_data.csv')
num_entries = df_raw.shape[0]
num_attributes = df_raw.shape[1]

#df_shuffeled = df_raw.sample(frac=1,random_state=42)

#Preprocess target: Pick one of two possible targets, mean = 0 
X_energy = df_raw.iloc[:,:-2]
y_energy = df_raw.iloc[:,-2]
#Y = np.reshape(Y, len(Y))
y_energy = y_energy - np.mean(y_energy)





#Preprocess features. Scale all to [0,1]
'''X = df_shuffeled.values[:,:-2]
scaler = preprocessing.MinMaxScaler()
scaler.fit(X)
df_shuffeled = scaler.transform(X)'''

# Ridge regression, own and available implementation + plot

learning_rate = 0.000000002
lambda_parameter = 100.00 
max_iterations = 400

cv=KFold(n_splits=5, shuffle=True, random_state=42)


our_time=[]
skl_time=[]

pred_df=pd.DataFrame(columns =[0,1,2])



# Perform cross validation Ridge
if False:
    for train_index, test_index in cv.split(X_energy):
        
        X_train, X_test = X_energy.iloc[train_index], X_energy.iloc[test_index]
        y_train, y_test = y_energy.iloc[train_index], y_energy.iloc[test_index]
        
        # Fit models
        start=time.time()
        OurRidge.fit(X_train.to_numpy(),y_train.to_numpy(), lambda_parameter, learning_rate, max_iter = max_iterations, quiet=True)
        our_pred=OurRidge.regress_all(X_test.to_numpy())
        our_time.append(time.time()-start)
        
        start=time.time()
        availableRidge = Ridge(lambda_parameter, fit_intercept=False, max_iter=max_iterations)
        availableRidge.fit(X_train,y_train)
        skl_pred=availableRidge.predict(X_test)
        skl_time.append(time.time()-start)
        
        
        
        
        pred_df=pd.concat([pred_df,pd.DataFrame(list(zip(y_test,our_pred,skl_pred)))],axis=0)
        
        break
    

    pred_df.columns=["actual", "our_prediction","sklearn_prediction"]
    
    pred_df=pred_df.astype(float)
    
    correlation= (pearsonr(pred_df["our_prediction"],pred_df["sklearn_prediction"])[0])
    our_mse = mean_squared_error(pred_df["our_prediction"],pred_df["actual"])
    skl_mse =mean_squared_error(pred_df["sklearn_prediction"],pred_df["actual"])
    
    pred_df["our_dif"]=pred_df["actual"]-pred_df["our_prediction"]
    pred_df["skl_dif"]=pred_df["actual"]-pred_df["sklearn_prediction"]
    
    sns.regplot(x=pred_df["our_prediction"], y=pred_df["sklearn_prediction"])
    
    plt.figure()
    
    sns.distplot(pred_df['our_dif'], hist=True, kde=False, 
                 bins=int(180/5), color = 'blue',
                 hist_kws={'edgecolor':'black'})
    plt.axvline(0, color='black')
    # Add labels
    plt.title('Ridge: Histogram Our Prediction')
    plt.xlabel('Diffrence between Our Prediction and Actual Observation')
    plt.ylabel('Observations')
    
    plt.figure()
    
    sns.distplot(pred_df['skl_dif'], hist=True, kde=False, 
                 bins=int(180/5), color = 'blue',
                 hist_kws={'edgecolor':'black'})
    plt.axvline(0, color='black')
    # Add labels
    plt.title('Ridge: Histogram SKlearn Prediction')
    plt.xlabel('Diffrence between SKlearn Prediction and Actual Observation')
    plt.ylabel('Observations')
    
    plt.figure()



    
# Perform cross validation KNN
if True:
    our_time=[]
    skl_time=[]


    pred_df=pd.DataFrame(columns =[0,1,2])
    
    
    for train_index, test_index in cv.split(X_energy):
        X_train, X_test = X_energy.iloc[train_index], X_energy.iloc[test_index]
        y_train, y_test = y_energy.iloc[train_index], y_energy.iloc[test_index]
        
        # Fit models
        start=time.time()
        OurKNN.fit(X_train.to_numpy(),y_train.to_numpy())
        our_pred=OurKNN.kNNVector(X_test.to_numpy(),5)
        our_time.append(time.time()-start)
        
        start=time.time()
        KNN = KNeighborsRegressor(5)
        KNN.fit(X_train,y_train)
        skl_pred=KNN.predict(X_test)
        skl_time.append(time.time()-start)
        
        
        pred_df=pd.concat([pred_df,pd.DataFrame(list(zip(y_test,our_pred,skl_pred)))],axis=0)
    
    pred_df.columns=["actual", "our_prediction","sklearn_prediction"]
    
    pred_df=pred_df.astype(float)
    
    correlation= (pearsonr(pred_df["our_prediction"],pred_df["sklearn_prediction"])[0])
    our_mse = mean_squared_error(pred_df["our_prediction"],pred_df["actual"])
    skl_mse =mean_squared_error(pred_df["sklearn_prediction"],pred_df["actual"])
    
    pred_df["our_dif"]=pred_df["actual"]-pred_df["our_prediction"]
    pred_df["skl_dif"]=pred_df["actual"]-pred_df["sklearn_prediction"]
    
    sns.regplot(x=pred_df["our_prediction"], y=pred_df["sklearn_prediction"])
    
    plt.figure()
    
    sns.distplot(pred_df['our_dif'], hist=True, kde=False, 
                 bins=int(180/5), color = 'blue',
                 hist_kws={'edgecolor':'black'})
    plt.axvline(0, color='black')
    # Add labels
    plt.title('KNN: Histogram Our Prediction')
    plt.xlabel('Diffrence between Our Prediction and Actual Observation')
    plt.ylabel('Observations')
    
    plt.figure()
    
    sns.distplot(pred_df['skl_dif'], hist=True, kde=False, 
                 bins=int(180/5), color = 'blue',
                 hist_kws={'edgecolor':'black'})
    plt.axvline(0, color='black')
    # Add labels
    plt.title('KNN: Histogram SKlearn Prediction')
    plt.xlabel('Diffrence between SKlearn Prediction and Actual Observation')
    plt.ylabel('Observations')
    
    plt.figure()



