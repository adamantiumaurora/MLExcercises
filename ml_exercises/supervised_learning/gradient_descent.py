from model_functions import init, predict


'''
Iterative process for training the parameters by performing three actions on the values: increase, decrease, no changes.

Parameters:
    x (array): vector x of features representing a data sample (e.g. single apartment)
    model_parameters (dict): dictionary representing parameters of the model along with their current state
    step (double): double value representing the amount of decrease/increase
    iterations (integer): integer value representing amount of iterations

'''
def train_parameters(x, model_parameters, step=0.1, iterations=100):


    number_of_params = len(model_parameters)


    return