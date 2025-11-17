# src/lstm_model.py
import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# LSTM model definition
# ---------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)              # out: (batch, seq_len, hidden)
        last = out[:, -1, :]               # last timestep
        return self.head(last).squeeze(-1) # (batch,)

# ---------------------------
# Dataset: sliding windows
# ---------------------------
class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], target_col: str,
                 seq_len: int = 50, stride: int = 1):
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        Xs, ys = [], []
        arr_x = df[feature_cols].values
        arr_y = df[target_col].values if target_col in df.columns else np.zeros(len(df))
        n = len(df)
        for start in range(0, n - seq_len + 1, stride):
            end = start + seq_len
            Xs.append(arr_x[start:end])
            ys.append(arr_y[end - 1])  # label = target at last timestep of window
        self.X = np.array(Xs, dtype=np.float32)
        self.y = np.array(ys, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------
# Helpers to load feature list & scaler
# ---------------------------
def load_feature_and_scaler(models_dir: str):
    with open(os.path.join(models_dir, "feature_columns.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    with open(os.path.join(models_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return feature_cols, scaler

# ---------------------------
# Train function
# ---------------------------
def train_from_dataframe(df: pd.DataFrame,
                         models_dir: str,
                         target_col: str = "target",
                         seq_len: int = 50,
                         batch_size: int = 64,
                         epochs: int = 20,
                         lr: float = 1e-3,
                         hidden_size: int = 128,
                         num_layers: int = 2,
                         device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feature_cols, scaler = load_feature_and_scaler(models_dir)

    # scale features (scaler expected pre-fitted). If not, you'll need to fit and save one separately.
    X = df[feature_cols].values
    Xs = scaler.transform(X)
    df_scaled = df.copy()
    df_scaled[feature_cols] = Xs

    dataset = SequenceDataset(df_scaled, feature_cols, target_col, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = LSTMRegressor(input_size=len(feature_cols), hidden_size=hidden_size, num_layers=num_layers)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    save_path = os.path.join(models_dir, "lstm_model.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = float(np.mean(losses)) if losses else 0.0
        print(f"[Train] Epoch {epoch}/{epochs} — loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state": model.state_dict(),
                "input_size": len(feature_cols),
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "seq_len": seq_len
            }, save_path)
            print(f"Saved best model to: {save_path} (loss={best_loss:.6f})")

    return save_path

# ---------------------------
# Load model + predict utility
# ---------------------------
def load_model(models_dir: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(os.path.join(models_dir, "lstm_model.pt"), map_location=device)
    input_size = ckpt["input_size"]
    hidden_size = ckpt["hidden_size"]
    num_layers = ckpt["num_layers"]
    seq_len = ckpt.get("seq_len", 50)

    model = LSTMRegressor(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    feature_cols, scaler = load_feature_and_scaler(models_dir)
    return model, scaler, feature_cols, seq_len, device

def predict_dataframe(df: pd.DataFrame, models_dir: str, target_col: str = "target"):
    model, scaler, feature_cols, seq_len, device = load_model(models_dir)
    X = df[feature_cols].values
    Xs = scaler.transform(X)
    df_scaled = df.copy()
    df_scaled[feature_cols] = Xs

    dataset = SequenceDataset(df_scaled, feature_cols, target_col=target_col, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds.extend(out.tolist())
    return np.array(preds)
