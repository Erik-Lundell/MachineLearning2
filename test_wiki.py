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

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR



with open('data/wikivital_mathematics.json') as data_file:    
    math_data = json.load(data_file)
    
keys = math_data.keys()

topics_id = math_data["node_ids"]
topics = math_data["node_ids"].keys()

X = []
Y = []

prev_total_visitors = 0
for day in range(int(math_data['time_periods'])):
    day_data = math_data[str(day)]
    
    index = day_data['index']
    year = day_data['year']
    month = day_data['month']
    date = day_data['day']
    weekday = index % 7
    
    #Calculate the number of visitors this day as target + feature for next day.
    total_visitors = 0
    for visitors in day_data['y']:
        total_visitors = total_visitors + int(visitors)
    
    #We can't use first day since 
    if(index>0):
        # 
        x = [index, year, month, date]
        
        # One hot encode weekday
        for i in range(7):
            if(i == weekday):
                x.append(1)
            else:
                x.append(0)
        
        x.append(prev_total_visitors)
                  
        X.append(x)
        Y.append(total_visitors)
        
    prev_total_visitors = total_visitors

X = np.array(X)
Y = np.array(Y)

#Preprocess features. scale to [-1, 1]
scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
scaler.fit(X)
X = scaler.transform(X)

#Preprocess target. Subtract mean.
Y = Y - np.mean(Y)


X=pd.DataFrame(X)
y = pd.Series(Y)

#y.columns="target"


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
if True:
    for train_index, test_index in cv.split(X):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
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
        
    

    pred_df.columns=["actual", "our_prediction","sklearn_prediction"]
    
    pred_df=pred_df.astype(float)
    
    correlation= (pearsonr(pred_df["our_prediction"],pred_df["sklearn_prediction"])[0])
    our_mse = mean_squared_error(pred_df["our_prediction"],pred_df["actual"])
    skl_mse =mean_squared_error(pred_df["sklearn_prediction"],pred_df["actual"])
    
    our_time= sum(our_time)/len(our_time)
    skl_time= sum(skl_time)/len(skl_time)
    
    our_r2 = r2_score(pred_df["our_prediction"],pred_df["actual"])
    skl_r2 =r2_score(pred_df["sklearn_prediction"],pred_df["actual"])
    
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
if False:
    our_time=[]
    skl_time=[]


    pred_df=pd.DataFrame(columns =[0,1,2])
    
    
    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
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
    
    our_time= sum(our_time)/len(our_time)
    skl_time= sum(skl_time)/len(skl_time)
    
    our_r2 = r2_score(pred_df["our_prediction"],pred_df["actual"])
    skl_r2 =r2_score(pred_df["sklearn_prediction"],pred_df["actual"])
    
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
    
    
    
# Perform cross validation RandomForestRegressor
if False:
    skl_time=[]


    pred_df=pd.DataFrame(columns =[0,1])
    
    
    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        
        start=time.time()
        mod = RandomForestRegressor()
        mod.fit(X_train,y_train)
        skl_pred=mod.predict(X_test)
        skl_time.append(time.time()-start)
        
        
        pred_df=pd.concat([pred_df,pd.DataFrame(list(zip(y_test,skl_pred)))],axis=0)
    
    pred_df.columns=["actual","sklearn_prediction"]
    
    pred_df=pred_df.astype(float)
    
    skl_mse =mean_squared_error(pred_df["sklearn_prediction"],pred_df["actual"])
    skl_r2 =r2_score(pred_df["sklearn_prediction"],pred_df["actual"])
    
    skl_time= sum(skl_time)/len(skl_time)
    
    pred_df["skl_dif"]=pred_df["actual"]-pred_df["sklearn_prediction"]
    
    sns.distplot(pred_df['skl_dif'], hist=True, kde=False, 
                 bins=int(180/5), color = 'blue',
                 hist_kws={'edgecolor':'black'})
    plt.axvline(0, color='black')
    # Add labels
    plt.title('Random Forest: Histogram SKlearn Prediction')
    plt.xlabel('Diffrence between SKlearn Prediction and Actual Observation')
    plt.ylabel('Observations')
    
    plt.figure()
    
    
    features = X_train.columns
    importances = mod.feature_importances_
    indices = np.argsort(importances)

    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    
    plt.figure()
    
    
# Perform cross validation SVM
if False:
    skl_time=[]


    pred_df=pd.DataFrame(columns =[0,1])
    
    
    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        
        start=time.time()
        mod = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
        mod.fit(X_train,y_train)
        skl_pred=mod.predict(X_test)
        skl_time.append(time.time()-start)
        
        
        pred_df=pd.concat([pred_df,pd.DataFrame(list(zip(y_test,skl_pred)))],axis=0)
    
    pred_df.columns=["actual","sklearn_prediction"]
    
    pred_df=pred_df.astype(float)
    
    skl_mse =mean_squared_error(pred_df["sklearn_prediction"],pred_df["actual"])
    skl_r2 =r2_score(pred_df["sklearn_prediction"],pred_df["actual"])
    
    skl_time= sum(skl_time)/len(skl_time)
    
    pred_df["skl_dif"]=pred_df["actual"]-pred_df["sklearn_prediction"]
    
    sns.distplot(pred_df['skl_dif'], hist=True, kde=False, 
                 bins=int(180/5), color = 'blue',
                 hist_kws={'edgecolor':'black'})
    plt.axvline(0, color='black')
    # Add labels
    plt.title('SVM: Histogram SKlearn Prediction')
    plt.xlabel('Diffrence between SKlearn Prediction and Actual Observation')
    plt.ylabel('Observations')
    
    plt.figure()
    
    
