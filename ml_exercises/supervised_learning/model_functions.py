import numpy as np

'''
Initialization function for starting values of model parameters.

@param: n number of features in data
@return: dictionary where w is array of doubles for weights (=representation of importance) 
        & b is double value for the bias (location on the y-axis)
'''
def init(n, w, b):
    return {"w": np.array([float(w)]).repeat(n), "b": float(b)}


'''
Prediction function for linear regression model: y = wx + b
@param: x vector x of features representing a data sample (e.g. single apartment)
@param: parameters dictionary which stores parameters of the model along with their current state

'''
def predict(x, parameters):
    # Prediction initial value
    prediction = 0

    # Adding multiplication of each feature with it's weight
    for weight, feature in zip(parameters["w"], x):
        prediction += weight * feature
        
    # Adding bias
    prediction += parameters["b"]
        
    return prediction