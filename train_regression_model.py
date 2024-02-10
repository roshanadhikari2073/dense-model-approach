import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers, losses
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np

# Fetching real-time stock data
api_key = 'LGGPOMUFZTVCMEI0'  
symbol = 'GOOGL' 

# Initialize Alpha Vantage TimeSeries
ts = TimeSeries(key=api_key, output_format='pandas')

# Fetch stock data (example: intraday data)
data, meta_data = ts.get_intraday(symbol=symbol, interval='60min', outputsize='full')

# Basic Preprocessing
close_prices = data['4. close']
normalized_data = (close_prices - close_prices.min()) / (close_prices.max() - close_prices.min())
train_data = np.array(normalized_data)

# Assuming we're using past 60 points to predict the next point
X = []
y = []
for i in range(60, len(train_data)):
    X.append(train_data[i-60:i])
    y.append(train_data[i])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM

# Splitting the dataset into Training and Test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Building the model using LSTM layers
def build_lstm_model():
    model = models.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    model.compile(optimizer=optimizers.Adam(), loss=losses.MeanSquaredError(), metrics=['mae'])
    return model

# Preparing the model
model = build_lstm_model()

# Training the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Saving the model
model.save('/workspaces/dense-model-approach')
