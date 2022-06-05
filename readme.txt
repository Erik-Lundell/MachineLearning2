// For writing done thoughts about the exercise

Questions:
    - What is the actual model in Ridge regression, linear? polynomial of some degree? (linear now)
    - Stopping condition in Ridge? (handled with three diffrent now)
    - Should bias be handled in RidgeRegression? Or do we just assume data to have mean=0 (if that even works)
    - Some decisions, see below...
    - How to process data in math data-set

Tasks:

1. Pick 3 data sets
    a) Math dataset from ex0
    b) Energy efficiency: straight forward, 768 instances with 8 features, 2 values to predict. No missing values.
    c) ...
2. Implement ridge regression gradient descent
3. Implement k-nn for regression
    3b Automatically find best k + distance function
    Do not use same test set for different tests
    
using CROSS-VALIDATION, two performance metrics
    a) R2: sklearn score 1-u/v  (sum of squared residuals / sum of squared y deviation from mean)
    b) Mean squared Error
    
4. Compare performance with existing implementations (defualt parameters ok)
5. Compare performance with 2 regression techniques:
    a) Random Forest
    b) ...
    
Conclusions
– How efficient are your algorithms
– Performance of your algorithms regarding performance metrics for
regression
– Impact of learning rate, regularization constant
– Impact of pre-processing
– Other findings
powerpoint ~25,40 slides

