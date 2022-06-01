from numpy import polynomial as poly

#Weights representing the model
_w = None
_w_old = None

#Calculate the actual regression at point x1
def regress(x1):
    #Return regression with current model
    #Just linear??
    pass

#Calculate the partial derivative of the regression at point x1 with respect to weight
def pd_regress(x1, i_w):
    pass

#Calculate sum of squared residuals
def RSS(x,y):
    rss_sum = 0
    for i, x1 in enumerate(x):
        y_est = regress(x1, _w)
        residual = y_est - y[i]
        rss_sum = rss_sum + residual*residual
        
    return rss_sum

#Calculate partial derivative of RSS with respect to weight _w[i_w]
def pd_RSS(x,y, i_w):
    pd_rss_sum = 0
    for i, x1 in enumerate(x):
        y_est = regress(x1, _w)
        residual = y_est - y[i]
        pd_rss_sum = od_rss_sum - 2*residual*pd_regress(x1, i_w)
        
    return pd_rss_sum

#Calculate cost_function
def cost_function(x,y, lambda_parameter):
    return RSS(x,y) + lambda_parameter*np.inner(_w,_w)

#Calculate partial derivate of cost function with respect to weight _w[i_w]
def pd_cost_function(x,y, lambda_parameter, i_w):
    return pd_RSS(x,y,_iw) + lambda_parameter*2*_w[i_w]

#Perform one step of gradiant_
def grad_descent_iteration(x,y, lambda_parameter learning_rate):
    global _w
    
    new_w = len(_w) * [0]
    for i in range(_w):
        new_w[i] = _w[i] - learning_rate*pd_cost_function(x,y,lambda_parameter, i)
        
    #Update all weights simultanously (pocket old weights)
    _w_old = _w
    _w = new_w

# Fits the model using RidgeRegression. Stops when cost_function  
def fit(x,y, intial_weights = None, lambda_parameter, learning_rate, epsilon = 0.001):
    
    #Initialize weights
    if(initial_weights is None):
        
        #Size of w not clear since model is yet unknown
        _w = numpy.rand(len(y))
    else:
        _w = initial_weights
        
    old_cost = cost_function(x,y, lambda_parameter)
    inter_counter = 0
    while(True):
        inter_counter = inter_counter + 1
        grad_descent_iteration(x,y,lambda_parameter, learning_rate)
        new_cost = cost_function(x,y, lambda_parameter)
        
        improvement_absolute = old_RSS-new_RSS
        improvement_relative = improvement_absolute/oldRSS
        
        print(f"Iteration {inter_counter}: cost = {new_cost:.2f} (improvement: {improvement_relative*100:.2f}%).")
        
        #Stopping condition 1. Cost got worse
        if(new_cost > old_cost):
            print("Terminating fitting since result got worse. Decrease learning rate?")
            break
            
        #Stopping condition 2. convergence
        if(improvement_relative < epsilon):
            print("Terminating fitting since relative improvement dropped below threshold.")
            break
    
    
