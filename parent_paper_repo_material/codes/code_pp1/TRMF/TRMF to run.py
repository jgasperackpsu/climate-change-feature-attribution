
import numpy as np
import pandas as pd
import random

from trmf import trmf
from RollingCV import RollingCV


random.seed(1)
np.random.seed(1)

# Load data
electricity = pd.read_csv(r"")
data = electricity.values.T
N = data.shape[0]
T = data.shape[1]
K = 4
lags = [1, 2, 3]
L = len(lags)

# TRMF model parameters
lambda_f = 2
lambda_x = 2
lambda_w = 2
eta = 2
alpha = 1000.
max_iter = 1000
T_train = data.shape[1] - 1
T_test = 1
T_step = 1
T_start = 0

# Initialize model
model = trmf(lags, K, lambda_f, lambda_x, lambda_w, alpha, eta, max_iter)

# Perform rolling cross-validation
for h in [1]:
    scores_mape = RollingCV(model, data, T_train, h, T_step=1, metric='MAPE')
    scores_rmse = RollingCV(model, data, T_train, h, T_step=1, metric='NRMSE')

    # Output MAPE and RMSE
    print(f"{scores_mape[0]:.4f}")
    print(f"{scores_rmse[0]:.4f}")
