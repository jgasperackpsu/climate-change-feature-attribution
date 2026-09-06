"""
reference:Chen T, Guestrin C. Xgboost: A scalable tree boosting system[C]//Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining. 2016: 785-794.
reference code link:https://github.com/Jenniferz28/Time-Series-ARIMA-XGBOOST-RNN
"""
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import math


def mape(y_true, y_pred):
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# read data
read_path = ""
data = pd.read_excel(read_path)

# Split features and target values
X = data.drop('', axis=1)
y = data['']


y_mean = y.mean()
y_std = y.std()

y_scaled = (y - y_mean) / y_std

# Divide the dataset into training and testing sets
X_train = X[:-1]
X_test = X[-1:]
y_train = y_scaled[:-1]
y_test = y_scaled[-1:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

learning_rate = 0.08
n_estimators = 100
model = xgb.XGBRegressor(max_depth=5, learning_rate=learning_rate, n_estimators=n_estimators,
                         objective='reg:squarederror', random_state=101)
model.fit(X_train_scaled, y_train)

y_pred_scaled = model.predict(X_test_scaled)

y_pred_original = y_pred_scaled * y_std + y_mean
y_test_original = y_test.values * y_std + y_mean

# calculate RMSE and MAPE
rmse = math.sqrt(mean_squared_error(y_test_original, y_pred_original))
mape_value = mape(y_test_original, y_pred_original)


print(f"{mape_value}")
print(f"{rmse}")


