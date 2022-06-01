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