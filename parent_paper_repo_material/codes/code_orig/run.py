"""
run.py — Master runner for all ML models in code_orig/
Loads real BP CO2 data, runs LSTM, CNN, XGBoost, TCN, TRMF, and DE (ensemble),
then prints a consolidated metrics table.

Data: Carbon Dioxide Emissions from Energy (1965-2022), BP Statistical Review
Target: US CO2 emissions (Mt)
Features: other countries' CO2 emissions in the same dataset
"""

import os
import sys
import warnings
import tempfile
import math
import random as rd

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = "0"

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "Hierarchical BP datasets(from energyinst.org)",
    "Carbon_Doixide",
    "Carbon_Dioxide_Emissions_from_Energy(1965-2022).xlsx",
)
DATA_PATH = os.path.normpath(DATA_PATH)


def load_co2_dataframe() -> pd.DataFrame:
    """
    Load the BP CO2 dataset and return a clean DataFrame.
    Rows = years (1965–2022), columns = countries/regions + target column 'US'.
    All columns are fully numeric with no NaNs.
    """
    raw = pd.read_excel(DATA_PATH, sheet_name=0, header=2)
    country_col = raw.columns[0]
    year_cols = [
        c
        for c in raw.columns
        if isinstance(c, (int, float)) and c == int(c) and 1965 <= c <= 2022
    ]
    df = raw[raw[country_col].notna()].copy()
    df = df[[country_col] + year_cols].set_index(country_col).T
    # Keep only fully-numeric, NaN-free columns
    df = df.select_dtypes(include=[np.number]).dropna(axis=1)
    df.index = df.index.astype(int)
    df.index.name = "Year"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ─────────────────────────────────────────────────────────────────────────────
# XGBOOST
# ─────────────────────────────────────────────────────────────────────────────

def run_xgboost(df: pd.DataFrame):
    """
    Reference: Chen & Guestrin, XGBoost, KDD 2016.
    Train on all years except the last; predict the last year.
    """
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler

    TARGET = "US"
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    y_mean, y_std = y.mean(), y.std()
    y_scaled = (y - y_mean) / y_std

    X_train, X_test = X.iloc[:-1], X.iloc[-1:]
    y_train, y_test = y_scaled.iloc[:-1], y_scaled.iloc[-1:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = xgb.XGBRegressor(
        max_depth=5,
        learning_rate=0.08,
        n_estimators=100,
        objective="reg:squarederror",
        random_state=101,
        verbosity=0,
    )
    model.fit(X_train_s, y_train, verbose=False)

    y_pred_s = model.predict(X_test_s)
    y_pred_orig = y_pred_s * y_std + y_mean
    y_test_orig = y_test.values * y_std + y_mean

    return mape(y_test_orig, y_pred_orig), rmse(y_test_orig, y_pred_orig), y_pred_orig, y_test_orig


# ─────────────────────────────────────────────────────────────────────────────
# LSTM (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def run_lstm(df: pd.DataFrame):
    """
    Reference: Hochreiter & Schmidhuber, Neural Computation 1997.
    Two-layer stacked LSTM with Dropout; train on all-but-last year, predict last year.
    Implemented natively in PyTorch to avoid TensorFlow/Keras C++ mutex lock issues.
    """
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler

    torch.manual_seed(1234)
    np.random.seed(1337)

    TARGET = "US"
    X = df.drop(TARGET, axis=1).values
    Y = df[TARGET].values.reshape(-1, 1)

    scX = MinMaxScaler(feature_range=(0, 1))
    scY = MinMaxScaler(feature_range=(0, 1))
    X_s = scX.fit_transform(X)
    Y_s = scY.fit_transform(Y)

    train_len = len(df) - 1
    X_train, Y_train = X_s[:train_len], Y_s[:train_len]
    X_test, Y_test = X_s[train_len:], Y_s[train_len:]

    X_train_t = torch.FloatTensor(X_train).unsqueeze(1)  # (N, 1, num_features)
    Y_train_t = torch.FloatTensor(Y_train)               # (N, 1)
    X_test_t = torch.FloatTensor(X_test).unsqueeze(1)
    Y_test_t = torch.FloatTensor(Y_test)

    class StackedLSTM(nn.Module):
        def __init__(self, in_features, hidden_dim=50):
            super().__init__()
            self.lstm1 = nn.LSTM(in_features, hidden_dim, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            out, _ = self.lstm1(x)
            out = self.relu(out)
            out = self.dropout(out)
            out, _ = self.lstm2(out)
            out = self.relu(out[:, -1, :])
            return self.fc(out)

    model = StackedLSTM(X_train.shape[1], hidden_dim=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for _ in range(100):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_pred_s = model(X_test_t).cpu().numpy()

    Y_pred = scY.inverse_transform(test_pred_s)
    Y_true = scY.inverse_transform(Y_test)

    return mape(Y_true, Y_pred), rmse(Y_true, Y_pred), Y_pred, Y_true


# ─────────────────────────────────────────────────────────────────────────────
# CNN (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def run_cnn(df: pd.DataFrame):
    """
    Reference: O'Shea & Nash, arXiv 2015.
    1D-CNN over a rolling window; train on all windows except the last.
    """
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler

    TARGET = "US"
    features = [c for c in df.columns if c != TARGET]

    df_c = df.copy()
    scaler_feat = MinMaxScaler(feature_range=(-1, 1))
    scaler_tgt = MinMaxScaler(feature_range=(-1, 1))
    df_c[features] = scaler_feat.fit_transform(df_c[features])
    df_c[[TARGET]] = scaler_tgt.fit_transform(df_c[[TARGET]])

    window_size = min(12, len(df_c) - 2)

    def make_windows(seq, ws):
        out = []
        for i in range(len(seq) - ws):
            w = seq.iloc[i : i + ws][features].values
            lbl = seq.iloc[i + ws][TARGET]
            out.append((w, lbl))
        return out

    class MultiFeatureCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(len(features), 16, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(16 * window_size, 100)
            self.fc2 = nn.Linear(100, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.relu(self.conv1(x))
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    model = MultiFeatureCNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # Train on all windows except the very last one (which we predict)
    train_data = make_windows(df_c.iloc[:-1], window_size)
    for _ in range(100):
        for seq, y_val in train_data:
            optimizer.zero_grad()
            model.train()
            y_pred = model(torch.FloatTensor(seq).unsqueeze(0))
            loss = criterion(y_pred, torch.FloatTensor([y_val]))
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_seq = torch.FloatTensor(
            df_c.iloc[-1 - window_size : -1][features].values
        ).unsqueeze(0)
        pred_s = model(test_seq).item()

    pred_orig = scaler_tgt.inverse_transform([[pred_s]])[0][0]
    actual_s = df_c[TARGET].iloc[-1]
    actual_orig = scaler_tgt.inverse_transform([[actual_s]])[0][0]

    return mape([actual_orig], [pred_orig]), rmse([actual_orig], [pred_orig]), np.array([pred_orig]), np.array([actual_orig])


# ─────────────────────────────────────────────────────────────────────────────
# TCN (Temporal Convolutional Network, PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def run_tcn(df: pd.DataFrame):
    """
    Reference: Bai et al., 'An Empirical Evaluation of Generic Convolutional
    and Recurrent Networks for Sequence Modeling', arXiv 2018.
    Dilated causal convolutional network with residual connections.
    """
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler

    torch.manual_seed(1234)
    np.random.seed(1337)

    TARGET = "US"
    window_size = 10
    nb_filters = 5
    kernel_size = 5
    dilations = [2, 4, 8, 16]

    scaler = MinMaxScaler()
    df_s = scaler.fit_transform(df.values)
    column_len = df.shape[1]
    tgt_idx = df.columns.tolist().index(TARGET)

    def make_windows(data, ws):
        X_list, y_list = [], []
        for i in range(len(data) - ws):
            X_list.append(data[i : i + ws, :])
            y_list.append(data[i + ws, tgt_idx])
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    train_s = df_s[:-1]
    test_s = df_s[-(window_size + 1):]

    x_train, y_train = make_windows(train_s, window_size)
    x_test, y_test_s = make_windows(test_s, window_size)

    class DilatedResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, k_size, dilation):
            super().__init__()
            self.trim = (k_size - 1) * dilation
            self.conv1 = nn.Conv1d(
                in_channels, out_channels, k_size,
                dilation=dilation, padding=self.trim
            )
            self.conv2 = nn.Conv1d(
                out_channels, out_channels, k_size,
                dilation=dilation, padding=self.trim
            )
            self.relu = nn.ReLU()
            self.proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        def forward(self, x):
            res = x if self.proj is None else self.proj(x)
            out = self.relu(self.conv1(x))
            if self.trim > 0:
                out = out[:, :, :-self.trim]
            out = self.relu(self.conv2(out))
            if self.trim > 0:
                out = out[:, :, :-self.trim]
            return out + res

    class PyTorchTCN(nn.Module):
        def __init__(self, in_features, nb_filters=5, k_size=5, dilations=[2, 4, 8, 16]):
            super().__init__()
            blocks = []
            c_in = in_features
            for d in dilations:
                blocks.append(DilatedResidualBlock(c_in, nb_filters, k_size, d))
                c_in = nb_filters
            self.network = nn.Sequential(*blocks)
            self.fc1 = nn.Linear(nb_filters, 13)
            self.fc2 = nn.Linear(13, 1)

        def forward(self, x):
            # x: (B, T, C) -> Conv1d: (B, C, T)
            x = x.transpose(1, 2)
            feat = self.network(x)
            pooled = feat.mean(dim=-1)
            h = torch.sigmoid(self.fc1(pooled))
            return torch.sigmoid(self.fc2(h))

    model = PyTorchTCN(column_len, nb_filters=nb_filters, k_size=kernel_size, dilations=dilations)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()

    x_train_t = torch.FloatTensor(x_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    x_test_t = torch.FloatTensor(x_test)

    dataset = torch.utils.data.TensorDataset(x_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)

    for _ in range(100):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        predict = model(x_test_t).cpu().numpy()

    if predict.ndim == 1:
        predict = predict.reshape(-1, 1)

    pre_copies = np.repeat(predict, column_len, axis=-1)
    pred_orig = scaler.inverse_transform(pre_copies)[:, tgt_idx]
    true_copies = np.repeat(y_test_s.reshape(-1, 1), column_len, axis=-1)
    true_orig = scaler.inverse_transform(true_copies)[:, tgt_idx]

    return mape(true_orig, pred_orig), rmse(true_orig, pred_orig), pred_orig, true_orig


# ─────────────────────────────────────────────────────────────────────────────
# TRMF (Temporal Regularized Matrix Factorization)
# ─────────────────────────────────────────────────────────────────────────────

def run_trmf(df: pd.DataFrame):
    """
    Reference: Yu et al., 'Temporal Regularized Matrix Factorization for
    High-dimensional Time Series Prediction', NeurIPS 2016.
    Uses the local trmf.py + RollingCV.py implementation.
    """
    import sys

    TRMF_DIR = os.path.join(os.path.dirname(__file__), "TRMF")
    if TRMF_DIR not in sys.path:
        sys.path.insert(0, TRMF_DIR)
    from trmf import trmf
    from RollingCV import RollingCV

    random = __import__("random")
    random.seed(1)
    np.random.seed(1)

    TARGET = "US"
    # TRMF expects shape (N_series, T_timepoints)
    # Use all series; target is the US column
    data = df.values.T.astype(float)  # (N, T)

    T_train = data.shape[1] - 1
    lags = [1, 2, 3]
    K = 4
    model = trmf(
        lags=lags,
        K=K,
        lambda_f=2,
        lambda_x=2,
        lambda_w=2,
        alpha=1000.0,
        eta=2,
        max_iter=200,
    )

    # RollingCV returns (test_preds, test) — shape (N, h)
    preds, test = RollingCV(model, data, T_train, T_test=1, T_step=1, metric="MAPE")

    tgt_idx = df.columns.tolist().index(TARGET)
    y_pred = preds[tgt_idx, :].ravel()
    y_true = test[tgt_idx, :].ravel()

    # Compute metrics on un-normalised values
    # RollingCV normalises internally; back-transform using train stats
    train_col = df[TARGET].values[:-1]
    mean_t, std_t = train_col.mean(), train_col.std()
    if std_t == 0:
        std_t = 1.0
    y_pred_orig = y_pred * std_t + mean_t
    y_true_orig = y_true * std_t + mean_t

    return mape(y_true_orig, y_pred_orig), rmse(y_true_orig, y_pred_orig), y_pred_orig, y_true_orig


# ─────────────────────────────────────────────────────────────────────────────
# DE — Differential Evolution Ensemble Weighter
# ─────────────────────────────────────────────────────────────────────────────

def run_de(predictions: dict, y_true_arr: np.ndarray):
    """
    Reference: Das & Suganthan, IEEE TEVC 2010.
    Uses DE to find optimal ensemble weights for the four main models
    (XGBoost, LSTM, CNN, TRMF) then reports the ensemble metrics.

    predictions : dict mapping model name → 1-D numpy array of predictions
    y_true_arr  : 1-D numpy array of ground truth values (common length)
    """
    from numpy import (
        array, zeros, shape, random as nprand, clip, vstack,
        mean, abs as npabs, argmin
    )

    NP = 500       # population size
    size = len(predictions)
    xMin, xMax = 0.001, 1.0
    F = 0.5
    CR = 0.8
    max_gen = 200

    pred_keys = list(predictions.keys())
    # Stack predictions into (n_models, n_test_points)
    pred_matrix = np.vstack([predictions[k] for k in pred_keys])  # (size, T)
    y_true = y_true_arr.ravel()

    def cal_fitness(weights):
        # weights shape (size,), normalise so they sum to 1
        w = np.array(weights)
        w = np.clip(w, xMin, xMax)
        w = w / w.sum()
        ens = pred_matrix.T.dot(w)  # (T,)
        valid = y_true != 0
        return float(np.mean(np.abs((y_true[valid] - ens[valid]) / y_true[valid])) * 100)

    # Initialise population
    pop = np.random.uniform(xMin, xMax, (NP, size))
    pop = pop / pop.sum(axis=1, keepdims=True)

    fitness = np.array([cal_fitness(pop[i]) for i in range(NP)])

    for _ in range(max_gen):
        for i in range(NP):
            candidates = [j for j in range(NP) if j != i]
            r1, r2, r3 = rd.sample(candidates, 3)
            mutant = np.clip(pop[r1] + F * (pop[r2] - pop[r3]), xMin, xMax)
            trial = np.where(np.random.rand(size) <= CR, mutant, pop[i])
            f_trial = cal_fitness(trial)
            if f_trial < fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial

    best_idx = int(np.argmin(fitness))
    best_w = pop[best_idx]
    best_w = best_w / best_w.sum()

    ens_pred = pred_matrix.T.dot(best_w)

    print(f"\n  DE best weights: { {pred_keys[i]: round(float(best_w[i]), 4) for i in range(size)} }")

    return mape(y_true, ens_pred), rmse(y_true, ens_pred)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Corporate Carbon ML Prediction — Model Runner")
    print("=" * 60)
    print(f"\nLoading data from:\n  {DATA_PATH}\n")

    df = load_co2_dataframe()
    print(f"Dataset: {df.shape[0]} years × {df.shape[1]} series/features")
    print(f"Target : US CO2 (Mt),  years {df.index[0]}–{df.index[-1]}\n")

    results = {}
    model_preds = {}    # for DE ensemble
    y_true_common = None  # ground truth (from the last test point)

    # ── XGBoost ────────────────────────────────────────────────────────────
    print("Running XGBoost...", end=" ", flush=True)
    try:
        m, r, yp, yt = run_xgboost(df)
        results["XGBoost"] = (m, r)
        model_preds["XGBoost"] = yp
        y_true_common = yt
        print(f"MAPE={m:.3f}%  RMSE={r:.3f}")
    except Exception as e:
        results["XGBoost"] = (float("nan"), float("nan"))
        print(f"ERROR: {e}")

    # ── LSTM ───────────────────────────────────────────────────────────────
    print("Running LSTM...", end=" ", flush=True)
    try:
        m, r, yp, yt = run_lstm(df)
        results["LSTM"] = (m, r)
        model_preds["LSTM"] = yp
        if y_true_common is None:
            y_true_common = yt
        print(f"MAPE={m:.3f}%  RMSE={r:.3f}")
    except Exception as e:
        results["LSTM"] = (float("nan"), float("nan"))
        print(f"ERROR: {e}")

    # ── CNN ────────────────────────────────────────────────────────────────
    print("Running CNN...", end=" ", flush=True)
    try:
        m, r, yp, yt = run_cnn(df)
        results["CNN"] = (m, r)
        model_preds["CNN"] = yp
        if y_true_common is None:
            y_true_common = yt
        print(f"MAPE={m:.3f}%  RMSE={r:.3f}")
    except Exception as e:
        results["CNN"] = (float("nan"), float("nan"))
        print(f"ERROR: {e}")

    # ── TCN ────────────────────────────────────────────────────────────────
    print("Running TCN...", end=" ", flush=True)
    try:
        m, r, yp, yt = run_tcn(df)
        results["TCN"] = (m, r)
        model_preds["TCN"] = yp
        print(f"MAPE={m:.3f}%  RMSE={r:.3f}")
    except Exception as e:
        results["TCN"] = (float("nan"), float("nan"))
        print(f"ERROR: {e}")

    # ── TRMF ───────────────────────────────────────────────────────────────
    print("Running TRMF...", end=" ", flush=True)
    try:
        m, r, yp, yt = run_trmf(df)
        results["TRMF"] = (m, r)
        model_preds["TRMF"] = yp
        print(f"MAPE={m:.3f}%  RMSE={r:.3f}")
    except Exception as e:
        results["TRMF"] = (float("nan"), float("nan"))
        print(f"ERROR: {e}")

    # ── DE Ensemble ────────────────────────────────────────────────────────
    # Only run DE if we have at least 2 successful models with matching pred lengths
    print("Running DE (Differential Evolution Ensemble)...", end=" ", flush=True)
    try:
        if y_true_common is None:
            raise ValueError("No base model succeeded; cannot run DE.")
        n = len(y_true_common)
        # Filter to models whose predictions match the ground truth length
        valid_preds = {
            k: v for k, v in model_preds.items()
            if v is not None and len(v.ravel()) == n
        }
        if len(valid_preds) < 2:
            raise ValueError(f"Need ≥2 models for DE ensemble; got {len(valid_preds)}.")
        m, r = run_de(valid_preds, y_true_common)
        results["DE Ensemble"] = (m, r)
        print(f"MAPE={m:.3f}%  RMSE={r:.3f}")
    except Exception as e:
        results["DE Ensemble"] = (float("nan"), float("nan"))
        print(f"ERROR: {e}")

    # ── Metrics Table ──────────────────────────────────────────────────────
    print("\n")
    print("╔" + "═" * 16 + "╦" + "═" * 12 + "╦" + "═" * 14 + "╗")
    print("║{:^16}║{:^12}║{:^14}║".format("Model", "MAPE (%)", "RMSE (Mt CO2)"))
    print("╠" + "═" * 16 + "╬" + "═" * 12 + "╬" + "═" * 14 + "╣")
    for model_name, (m_val, r_val) in results.items():
        m_str = f"{m_val:.4f}" if not math.isnan(m_val) else "  ERROR "
        r_str = f"{r_val:.4f}" if not math.isnan(r_val) else "  ERROR "
        print("║{:^16}║{:^12}║{:^14}║".format(model_name, m_str, r_str))
    print("╚" + "═" * 16 + "╩" + "═" * 12 + "╩" + "═" * 14 + "╝")
    print("\nDone. All models evaluated on US CO2 emissions (last year held out).")


if __name__ == "__main__":
    main()
