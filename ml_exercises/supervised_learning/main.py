import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model_functions import init, predict
from cost_functions import mean_absolute_error, mean_squared_error


df_data = pd.read_csv("ml_exercises/data/cracow_apartments.csv", sep=",")



# Used features and target value
features = ["size"]
target = ["price"]



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

