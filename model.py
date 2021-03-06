# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QGQnyD40ILl16wH0SzMNPvBBeUwhwPvr
"""

# Import our dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score

# !wget https://github.com/plotly/orca/releases/download/v1.2.1/orca-1.2.1-x86_64.AppImage -O /usr/local/bin/orca
# !chmod +x /usr/local/bin/orca
# !apt-get install xvfb libgtk2.0-0 libgconf-2-4

#Import the dataset that we will train
stock_top_df = pd.read_csv("dataframes_top.csv")
stock_top_df

# '.values' need the 2nd Column Opening Price as a Numpy array (not vector)
training_set = stock_top_df.iloc[:, 1:2].values
training_set

# 'feature_range = (0,1)' makes sure that training data is scaled to have values between 0 and 1

scaler = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = scaler.fit_transform(training_set)

# Using Recurrent Neural Network (RNN) Deep Learning technique for continuous data pattern recognition. RNN takes into account how data changes over time.
# For that we need to Create a data structure with 60 timesteps (look back 60 days) and 1 output, telling RNN what to remember (Number of timesteps) when predicting the next Stock Price.

X_train = []

# 'y_train' Output with next day's stock price
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Keras RNNs expects an input shape (Batch Size, Timesteps, input_dim)
# .shape[0]: number of rows --> Batch Size
# .shape[1]: number of columns --> Timesteps
# 'input_dim': the number of factors that may affect stock prices

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#the x-train will have to reshaped

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
# 'return_sequences = True' because we will add more stacked LSTM Layers
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
# Dropout is included to avoid overfitting. 20% of Neurons will be ignored (10 out of 50 Neurons) to prevent Overfitting
model.add(Dropout(0.2))

# Adding second layer
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

# Adding third layer
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

# Adding fourth layer
# This is the last LSTM Layer. 'return_sequences = false' by default so we leave it out.
model.add(LSTM(units = 50))
model.add(Dropout(0.2))

# Adding the output layer
# 'units = 1' because Output layer has one dimension
model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Making the predictions and visualising the results

apple_test = pd.read_csv("apple_test.csv")
real_apple_stock_price = apple_test.iloc[:, 1:2].values

# We need 60 previous inputs for each day of the apple_set in 2021
# Combine 'dataset_train' and 'apple_test'
dataset_total = pd.concat((stock_top_df['Open'], apple_test['Open']), axis = 0)

# Extract Stock Prices for Test time period, plus 60 days previous
inputs = dataset_total[len(dataset_total) - len(apple_test) - 60:].values
# 'reshape' function to get it into a NumPy format
inputs = inputs.reshape(-1,1)
# Scaling the input
inputs = scaler.transform(inputs)

X_test = []

for i in range(60, 374):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
# Making the input in 3D format
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predict the Stock Price
predicted_stock_price = model.predict(X_test)
# We need to inverse the scaling of our prediction to get a Dollar amount
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

len(inputs)

# Visualising the results
from matplotlib.pyplot import figure

figure(figsize=(10, 20), dpi=80)

plt.plot(real_apple_stock_price, color = 'red', label = 'Real Apple Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Apple Stock Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Apple Stock Price')
plt.legend()
plt.show()
plt.savefig('apple.pdf')

# Evaluating the model

import math

from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(real_apple_stock_price, predicted_stock_price))

print(f"rmse: {round(rmse,2)}")

import joblib

joblib.dump(model, "model.pkl")

