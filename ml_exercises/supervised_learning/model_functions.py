import numpy as np


def init_parameters(n, w, b):
    '''
    Initialization function for starting values of model parameters.

    Parameters:
        n (integer): number of features in data
    Returns:
        (dict): dictionary where w is array of doubles for weights (=representation of importance) & b is double value for the bias (location on the y-axis)
    '''
    return {"w": np.array([float(w)]).repeat(n), "b": float(b)}



def predict(x, parameters):
    '''
    Prediction function for linear regression model: y = wx + b

    Parameters:
        x (array): vector x of features representing a data sample (e.g. single apartment)
        parameters (dict): dictionary which stores parameters of the model along with their current state

    Returns:
        (double): predicted value y (= wx + b)

    '''
    # Prediction initial value
    prediction = 0.0

    # Adding multiplication of each feature with it's weight
    for weight, feature in zip(parameters["w"], x):
        prediction += weight * feature
        
    # Adding bias
    prediction += parameters["b"]
        
    return prediction