from numpy import polynomial as poly
import numpy as np

#Weights representing the model
_w = None
_w_old = None

#Calculate the actual regression at point x1
def regress(x1):
    #Return regression with current model
    #Just linear??
    return np.inner(_w, x1)


#Calculate the actual regression at point x1
def regress_all(X):
    predictions=[]
    #Return regression with current model
    for x1 in X:
        predictions.append(regress(x1))
    return predictions
    

#Calculate the partial derivative of the regression at point x1 with respect to weight
def pd_regress(x1, i_w):
    return x1[i_w]

#Calculate sum of squared residuals
def RSS(x,y):
    rss_sum = 0
    for i, x1 in enumerate(x):
        y_est = regress(x1)
        residual = y[i] - y_est
        rss_sum = rss_sum + residual*residual
        
    return rss_sum

def meanResidual(x,y):
    residual_sum = 0
    for i, x1 in enumerate(x):
        y_est = regress(x1)
        residual = y[i] - y_est
        residual_sum = residual_sum + np.sqrt(residual*residual)
        
    return residual_sum/len(y)

#Calculate partial derivative of RSS with respect to weight _w[i_w]
def pd_RSS(x,y, i_w):
    pd_rss_sum = 0
    for i, x1 in enumerate(x):
        y_est = regress(x1)
        residual = y[i] - y_est
        pd_rss_sum = pd_rss_sum - 2*residual*pd_regress(x1, i_w)
        
    return pd_rss_sum

#Calculate cost_function
def cost_function(x,y, lambda_parameter):
    return RSS(x,y) + lambda_parameter*np.inner(_w,_w)

#Calculate partial derivate of cost function with respect to weight _w[i_w]
def pd_cost_function(x,y, lambda_parameter, i_w):
    return pd_RSS(x,y,i_w) + lambda_parameter*2*_w[i_w]

#Perform one step of gradiant_
def grad_descent_iteration(x,y, lambda_parameter, learning_rate):
    global _w
    
    new_w = len(_w) * [0]
    for i in range(len(_w)):
        new_w[i] = _w[i] - learning_rate*pd_cost_function(x,y,lambda_parameter, i)
        
    #Update all weights simultanously (pocket old weights)
    _w_old = _w
    _w = new_w

# Fits the model using RidgeRegression. Stops when cost_function  
def fit(x,y, lambda_parameter, learning_rate, epsilon = 0.0001, initial_weights = None, max_iter = 200, weight_threshold = 0.01, quiet = False):
    global _w
    #Initialize weights
    if(initial_weights is None):
        
        #Size of w not clear since model is yet unknown
        _w = np.random.rand(len(x[0,:]))
    else:
        _w = initial_weights
        
    old_cost = cost_function(x,y, lambda_parameter)
    print(f"Initial cost:{old_cost:.2f}")
 
    iter_counter = 0
    while(True):
        iter_counter = iter_counter + 1
        grad_descent_iteration(x,y,lambda_parameter, learning_rate)
        new_cost = cost_function(x,y, lambda_parameter)
        
        improvement_absolute = old_cost-new_cost
        improvement_relative = improvement_absolute/old_cost
        
        if(iter_counter<5 or iter_counter%10==0):
            if(not quiet):
                print(f"Iteration {iter_counter}: cost = {new_cost:.2f} (improvement: {improvement_relative*100:.2f}%).")
        
        #Stopping condition 1. Cost got worse
        if(new_cost > old_cost):
            print("Terminating fitting since result got worse. Decrease learning rate?")
            break
            
        #Stopping condition 2. convergence
        if(improvement_relative < epsilon):
            print("Terminating fitting since relative improvement dropped below threshold.")
            break
            
        #Stopping condition 3. To many iterations
        if(iter_counter>=max_iter):
            print("Terminating fitting since maximum number of iterations was reached.")
            break
            
        old_cost = new_cost
      
    if(not quiet):
        print("")
        
    #Remove features with too little influence
    counter =  0
    for i in range(len(_w)):
        if(np.abs(_w[i])<=weight_threshold):
            counter = counter + 1
            _w[i] = 0
    if(counter>0 and not quiet):
        print(f"Removed {counter} features due to low weight.")
        
    #Compute some metrics
    finalRSS =RSS(x,y)
    finalCost = cost_function(x,y, lambda_parameter)
    finalMeanResidual = meanResidual(x,y)
    diffY = np.max(y)-np.min(y)
    
    #Report
    if(not quiet):
        print("Final weights: " + str(_w))
        print("L2-norm of weights: " + str(np.sqrt(np.inner(_w,_w))))
        print(f"RSS split of cost: {finalRSS/finalCost*100:.2f}%")
        print(f"Mean residual: {finalMeanResidual}, compare to difference between max and min y: {diffY}")
