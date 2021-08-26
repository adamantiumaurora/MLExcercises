from cost_functions import *


df_data = pd.read_csv("../data/cracow_apartments.csv", sep=",")
# Used features and target value
features = ["size"]
target = ["price"]


'''
Initialization function for starting values of model parameters.

@param: n number of features in data
@return: dictionary where w is array of doubles for weights (=representation of importance) 
        & b is double value for the bias (location on the y-axis)
'''
def init(n, w, b):
    return {"w": np.array([float(w)]).repeat(n), "b": float(b)}


'''
Prediction function for linear regression model
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



# Initialize model parameters
n = len(features)
# Slice Dataframe to separate feature vectors and target value
x = df_data[features].values
y = df_data[target].values


mae = []
mse = []

for w in range(-3,7):
    model_parameters = init(n, w, 0)

    predictions = [predict(i, model_parameters) for i in x]

    mae.append(mean_absolute_error(predictions, y))
    mse.append(mean_squared_error(predictions, y))



# Creating plot
'''
plt.scatter(df_data["size"], df_data["price"])
plt.title("y = wx + b, [w={}, b={}]".format(model_parameters['w'], model_parameters['b']))
plt.xlabel("Size [m^2]")
plt.ylabel("Price [k zł]")
'''
plt.scatter(mse, list(range(-3,7)), c='yellow')
plt.scatter(mae, list(range(-3,7)), c='orange')
plt.plot(mse, c='yellow')
plt.plot(mae, c='orange')
#plt.plot(np.arange(0, df_data["size"].max(), 0.1), [predict([x], model_parameters) for x in np.arange(0, df_data["size"].max(), 0.1)], c="red")

plt.show()

