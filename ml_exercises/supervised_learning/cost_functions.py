def mean_absolute_error(predictions, observations):
    '''
    Mean absolute error expressing the arithmetic average of the absolute errors.
    All individual deviations have even importance.

    Parameters:
        predictions (array): array of double values representing predicted values
        observations (array): array of double values representing observed values

    Returns:
        (double): single, cumulative value representing the average of the absolute errors

    '''
    number_of_points = len(predictions)

    sum_of_errors = 0
    for predicted, observed in zip(predictions, observations):
        sum_of_errors += abs(predicted - observed)
    
    mae = (1.0 / number_of_points) * sum_of_errors
    return mae


def mean_squared_error(predictions, observations):
    '''
    Average squared error expressing the arithmetic squared value of difference between the predictions and expected results.
    Each individual deviation is equivalent to the area of the square created out of the geometrical distance between the measured points.

    Parameters:
        predictions (array): array of double values representing predicted values
        observations (array): array of double values representing observed values

    Returns:
        (double): single, cumulative value representing the average of the squared errors

    '''
    number_of_points = len(predictions)
    
    sum_of_squared_errors = 0
    for predicted, observed in zip(predictions, observations):
        sum_of_squared_errors += abs(predicted - observed) **2
    
    # division by two in the averaging denominator as its presence makes MSE derivation calculus cleaner
    mse = (1.0 / (number_of_points * 2)) * sum_of_squared_errors
    return mse