import numpy as np
from collections import deque
import math

_x = None
_y = None
_has_fitted = False

def L2(x1, x2):
    return np.linalg.norm(x1-x2)

def L1(x1, x2):
    return np.sum(np.abs(x1-x2))

def L_inf():
    return np.max(x1-x2)

distance_functions = [L1, L2, L_inf]

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
def metaKNN(x, y, min_k = 1, max_k = 10):
    for k in range(min_k, max_k):
        for distance in distance_functions:
            
            #perform test
            
            #if score is better, update best k and distance.
            best_k = k
            best_distance = distance
    
    
    best_k = 0
    best_distance = L2
    
    return best_k, best_distance

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

    
