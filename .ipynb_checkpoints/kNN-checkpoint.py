import numpy as np
from collections import deque
import math
import random

from sklearn.model_selection import KFold

_x = None
_y = None
_has_fitted = False

def L2(x1, x2):
    return np.linalg.norm(x1-x2)

def L1(x1, x2):
    return np.sum(np.abs(x1-x2))

def L_inf(x1, x2):
    d = np.abs(np.max(x1-x2))
    if(d==0):
        d = 0.000001
    return d

distance_functions = {
    'L1': L1, 
    'L2': L2,
    'L_inf' : L_inf
}

def mean_weighting(dist, k):
    return 1

def distance_weighting(dist, k):
    return 1/dist

# x is assumed to be numpy array [[row1], [row2]...] with row instances
# y is the regression value
def fit(x, y):
    global _x
    global _y
    global _has_fitted
    _x = x
    _y = y
    _has_fitted = True
    

### do kNN for one instance x1
def kNN(x1, k, norm = L2, weighting = mean_weighting):
    if(not _has_fitted):
        print("No fitted data yet.")
        return 0
    if(k>len(_x)):
        print("Warning: k>number of instances.")
    
    #Touple of (distance, regression value) of closest neighbours this far
    #Sorted low -> high.
    neighbours = deque(k*[(math.inf,0)], k)
    
    ### Loop through instances and find k neighbours
    for i, x2 in enumerate(_x):
        dist = norm(x1, x2)

        insert_at = -1
        for j, neigh in enumerate(neighbours):
            if(dist<=neigh[0]):
                insert_at = j
                break
        if(insert_at > -1):
            neighbours.pop()
            neighbours.insert(insert_at, (dist, _y[i]))
                
    ### Perform regression using the found neighbours
    
    # Calculate total weight
    total_weight = 0
    for neigh in neighbours:
        total_weight = total_weight + weighting(neigh[0], k)
    
    #Compute regression
    y = 0
    for neigh in neighbours:
        y = y + neigh[1]*weighting(neigh[0], k)/ total_weight 
    
    return y

### do kNN for vector of X, return predicted values
def kNNVector(x, k, norm = L2, weighting = mean_weighting):
    y_pred = []
    for row in x:
        y_pred.append(kNN(row,k,norm,weighting))
    
    return y_pred

### Find best k and distance function
# Brute force: just try all combinations
# Each test performed with cross-evaluatoin but with different samples
def metaKNN(x, y, min_k = 1, max_k = 10, folds = 5):
    
    #initialize pseudonumber generator with some seed each time to get consistent results.
    random.seed(27)
    
    best_k = 0
    best_distance = None
    bestRMS = np.Inf
    
    num_k = max_k - min_k + 1
    save_RMS = {
    'L1':num_k*[0],
    'L2':num_k*[0],
    'L_inf':num_k*[0]
    }
    
    for k in range(min_k, max_k+1):
        print("Testing k = " + str(k))
        for distanceKey in distance_functions.keys():
            distance = distance_functions[distanceKey]
            
            #perform test
            cv=KFold(n_splits=folds, shuffle=True, random_state=random.randint(1,1000))

            total_RMS = 0
            for train_index, test_index in cv.split(x):
                X_train, X_test = x[train_index], x[test_index]
                Y_train, Y_test = y[train_index], y[test_index]
            
                fit(X_train, Y_train)
                
                score, RMS, meanResidual, maxRes = report(X_test, Y_test, k, distance, distance_weighting)
                
                total_RMS = total_RMS + RMS
                
            save_RMS[distanceKey][k-min_k] = total_RMS/folds
            
            #if score is better, update best k and distance.
            if(total_RMS < bestRMS):
                best_k = k
                best_distance = distanceKey
                bestRMS = total_RMS
    
    print(f"Best k is {best_k}, best distance_function is {best_distance}")
    
    return save_RMS

### Calculate various metrics: 
#  score = 1 - u/w used by scipy
#  RMS 
#  average absolute of residuals
#  max residual
def report(x, y, k, norm = L2, weighting = mean_weighting):
    y_reg = kNNVector(x,k,norm, weighting)
    
    residuals = y - y_reg
    
    u = np.inner(residuals, residuals)
    w = np.inner(y,y)
    
    score = 1 - u/w
    RMS = np.sqrt(u)
    meanResidual = np.sum(np.abs(residuals)) / len(y)
    maxRes = np.max(residuals)
    
    return score, RMS, meanResidual, maxRes

    
