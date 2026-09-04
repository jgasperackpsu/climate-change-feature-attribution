"""
reference:Hochreiter S, Schmidhuber J. Long short-term memory[J]. Neural computation, 1997, 9(8): 1735-1780.
reference code link:https://github.com/jgpavez/LSTM---Stock-prediction
"""
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

os.environ['PYTHONHASHSEED'] = '0'
random.seed(12345)
tf.random.set_seed(1234)
np.random.seed(1337)

file_path = ""
df = pd.read_excel(file_path)

X = df.drop('', axis=1).values
Y = df[''].values.reshape(-1, 1)

feature_weights = np.ones(X.shape[1])
X_weighted = X * feature_weights

scX = MinMaxScaler(feature_range=(0, 1))
scY = MinMaxScaler(feature_range=(0, 1))
X_scaled = scX.fit_transform(X_weighted)
Y_scaled = scY.fit_transform(Y)

# split data sets
seq_len = 1
data_len = len(df)
train_len = int(data_len - 1)
X_train, Y_train = X_scaled[:train_len], Y_scaled[:train_len]
X_test, Y_test = X_scaled[train_len:], Y_scaled[train_len:]


X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_len, X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test), verbose=2)


Y_pred = model.predict(X_test)
Y_pred = scY.inverse_transform(Y_pred)
Y_test_original = scY.inverse_transform(Y_test)

mape = mean_absolute_percentage_error(Y_test_original, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test_original, Y_pred))

print(f"{mape:.3f}")
print(f"{rmse:.3f}")

