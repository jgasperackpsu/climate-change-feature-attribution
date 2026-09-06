"""
reference:O'shea K, Nash R. An introduction to convolutional neural networks[J]. arXiv preprint arXiv:1511.08458, 2015.
reference code link:https://github.com/dennybritz/cnn-text-classification-tf
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

data_path = ""
learning_rate = 0.005
epochs = 100
window_size = 12
df = pd.read_excel(data_path)

target = ''
features = df.columns.drop([target]).tolist()

scaler_features = MinMaxScaler(feature_range=(-1, 1))
scaler_target = MinMaxScaler(feature_range=(-1, 1))
df[features] = scaler_features.fit_transform(df[features])
df[[target]] = scaler_target.fit_transform(df[[target]])

def input_data(seq, ws):
    out = []
    L = len(seq)
    for i in range(L - ws):
        window = seq.iloc[i:i + ws][features].values
        label = seq.iloc[i + ws][target]
        out.append((window, label))
    return out

class MultiFeatureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=len(features), out_channels=16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * window_size, 100)
        self.fc2 = nn.Linear(100, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def root_mean_squared_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

current_data = df.copy()
model = MultiFeatureCNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_data = input_data(current_data, window_size)

for epoch in range(epochs):
    for seq, y_train in train_data:
        optimizer.zero_grad()
        model.train()
        y_pred = model(torch.FloatTensor(seq).unsqueeze(0))
        loss = criterion(y_pred, torch.FloatTensor([y_train]))
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    test_seq = torch.FloatTensor(current_data.iloc[-window_size:][features].values).unsqueeze(0)
    pred = model(test_seq).item()

pred_transformed = scaler_target.inverse_transform([[pred]])[0][0]
actual_transformed = scaler_target.inverse_transform([[current_data[target].iloc[-1]]])[0][0]

mape = mean_absolute_percentage_error([actual_transformed], [pred_transformed])
rmse = root_mean_squared_error([actual_transformed], [pred_transformed])

print(f"Predicted Value: {pred_transformed}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")
