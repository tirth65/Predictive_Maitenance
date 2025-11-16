# src/train_lstm.py
"""
LSTM trainer for predictive maintenance.

Usage (train):
    python -m src.train_lstm --csv ./data/equipment_data.csv --seq_len 32 --epochs 20 --batch_size 128

Usage (predict sliding windows):
    python -m src.train_lstm --csv ./data/equipment_data.csv --predict --seq_len 32

Usage (predict one-per-machine):
    python -m src.train_lstm --csv ./data/equipment_data.csv --predict_one_per_machine --seq_len 32 --id_col machine_id

Notes:
- Requires: torch, sklearn, pandas, joblib, src.preprocess.preprocess_data, src.features.add_features
- This script intentionally filters out columns that contain 'fail', 'maint', 'maintenance', 'label', 'target', 'cum'
  to avoid target leakage.
"""
from __future__ import annotations
import os
import json
import argparse
import joblib
import math
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from src.preprocess import preprocess_data
from src.features import add_features

# ---------------- Dataset & Model ----------------
class SequenceDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series, seq_len: int):
        self.X = X.values.astype("float32")
        self.y = y.values.astype("float32")
        self.seq_len = int(seq_len)
        self.n = max(0, len(self.X) - self.seq_len + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        s = int(idx)
        e = s + self.seq_len
        x = self.X[s:e]
        y = self.y[e - 1]  # label for last t in window
        return torch.from_numpy(x), torch.tensor([y], dtype=torch.float32)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        mid = max(1, hidden_size // 2)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, mid),
            nn.ReLU(),
            nn.Linear(mid, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.head(last)
        return logits.squeeze(-1)


# ---------------- Helpers ----------------
def ensure_models_dir(models_dir: str):
    os.makedirs(models_dir, exist_ok=True)


def detect_target_and_attach(raw_train: pd.DataFrame, raw_test: pd.DataFrame,
                             proc_train: pd.DataFrame, proc_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Prefer 'failure' if created by preprocess_data(); otherwise reattach raw 'Maintenance Required'.
    Returns (proc_train, proc_test, TARGET_COL)
    """
    if "failure" in proc_train.columns:
        return proc_train, proc_test, "failure"

    raw_name = "Maintenance Required"
    if raw_name in raw_train.columns:
        # attach as safe column
        SAFE = "_raw_failure_"
        proc_train = proc_train.copy()
        proc_train[SAFE] = raw_train[raw_name].reset_index(drop=True).astype("float32")
        if raw_name in raw_test.columns:
            proc_test = proc_test.copy()
            proc_test[SAFE] = raw_test[raw_name].reset_index(drop=True).astype("float32")
        return proc_train, proc_test, SAFE

    # fallback: try to auto-detect
    candidates = [c for c in proc_train.columns if "fail" in c.lower() or "maint" in c.lower()]
    if candidates:
        return proc_train, proc_test, candidates[0]

    raise KeyError("No target found. Expected processed 'failure' or raw 'Maintenance Required'.")


def filter_features(df: pd.DataFrame, extra_allow: list | None = None) -> list:
    """
    Return list of safe feature names (drop target-like and id/time).
    extra_allow: list of column names to force-include (e.g. sensor names)
    """
    leak_subs = ["fail", "failure", "maint", "maintenance", "label", "target", "cum"]
    drop_cols = set()
    for c in df.columns:
        low = c.lower()
        for s in leak_subs:
            if s in low:
                drop_cols.add(c)
                break
    # also drop timestamp/id-like
    for c in ["timestamp", "time", "machine_id", "id"]:
        if c in df.columns:
            drop_cols.add(c)
    allow = set(extra_allow or [])
    features = [c for c in df.columns if c not in drop_cols or c in allow]
    # final safety: keep only numeric columns
    features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    return features


def save_artifacts(models_dir: str, model: nn.Module, scaler: StandardScaler, feature_columns: list, meta: dict):
    torch.save(model.state_dict(), os.path.join(models_dir, "lstm_model.pt"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(list(feature_columns), os.path.join(models_dir, "feature_columns.pkl"))
    with open(os.path.join(models_dir, "lstm_meta.json"), "w") as f:
        json.dump(meta, f)
    print("Saved artifacts to:", models_dir)


def load_for_predict(models_dir: str, device: torch.device):
    meta_path = os.path.join(models_dir, "lstm_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError("Missing lstm_meta.json in models dir. Train first.")
    meta = json.load(open(meta_path, "r"))
    feature_cols = joblib.load(os.path.join(models_dir, "feature_columns.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    input_size = meta["input_size"]
    model = LSTMClassifier(input_size=input_size,
                           hidden_size=meta.get("hidden_size", 64),
                           num_layers=meta.get("num_layers", 1),
                           dropout=meta.get("dropout", 0.0))
    state = torch.load(os.path.join(models_dir, "lstm_model.pt"), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, scaler, feature_cols, meta.get("seq_len", 32)


# ---------------- Training ----------------
def train_lstm(csv_path: str,
               seq_len: int = 32,
               epochs: int = 20,
               batch_size: int = 128,
               hidden_size: int = 64,
               num_layers: int = 1,
               dropout: float = 0.2,
               lr: float = 1e-3,
               holdout_frac: float = 0.2,
               limit_rows: int | None = None,
               keep_features: list | None = None,
               models_dir: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = pd.read_csv(csv_path)
    if limit_rows and limit_rows > 0:
        df = df.head(limit_rows).reset_index(drop=True)
        print("Using top rows:", len(df))

    # chronological holdout
    split_idx = int(len(df) * (1.0 - holdout_frac))
    df_train_raw = df.iloc[:split_idx].reset_index(drop=True)
    df_test_raw = df.iloc[split_idx:].reset_index(drop=True)
    print(f"Train rows: {len(df_train_raw)}, Test rows: {len(df_test_raw)}")

    # preprocess + features
    print("Preprocessing + feature engineering (train)...")
    df_train_proc = add_features(preprocess_data(df_train_raw.copy()), include_trend=False)
    print("Preprocessing + feature engineering (test)...")
    df_test_proc = add_features(preprocess_data(df_test_raw.copy()), include_trend=False)

    # target detection / attach
    df_train_proc, df_test_proc, TARGET_COL = detect_target_and_attach(df_train_raw, df_test_raw, df_train_proc, df_test_proc)
    print("TARGET_COL:", TARGET_COL)

    # choose safe feature set
    if keep_features:
        feature_cols = [c for c in keep_features if c in df_train_proc.columns]
        if not feature_cols:
            raise ValueError("keep_features specified but none present in processed dataframe.")
    else:
        # auto-filter leakage features; allow core sensors if present
        core_allow = [c for c in ["temperature", "vibration", "pressure", "rpm"] if c in df_train_proc.columns]
        feature_cols = filter_features(df_train_proc, extra_allow=core_allow)
    print("Using feature columns (sample):", feature_cols[:10])

    # prepare X,y and safety cleaning
    y_train = df_train_proc[TARGET_COL].astype("float32")
    y_test  = df_test_proc[TARGET_COL].astype("float32")

    X_train = df_train_proc[feature_cols].replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    X_test  = df_test_proc[feature_cols].replace([float("inf"), float("-inf")], pd.NA).fillna(0)

    # scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    # sequence datasets
    train_ds = SequenceDataset(X_train_scaled, y_train, seq_len=seq_len)
    test_ds  = SequenceDataset(X_test_scaled, y_test, seq_len=seq_len)
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise ValueError("Sequence length too large for dataset. Reduce seq_len")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # model, loss, optimizer
    input_size = X_train_scaled.shape[1]
    model = LSTMClassifier(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)

    pos = max(1.0, float(y_train.sum()))
    neg = max(1.0, float((y_train == 0).sum()))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_f1 = -1.0
    patience_ctr = 0
    patience_limit = 6  # early stopping on val_f1

    models_dir = models_dir or os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models"))
    ensure_models_dir(models_dir)

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).squeeze(1)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * xb.size(0)
        avg_loss = total_loss / (len(train_loader.dataset) if len(train_loader.dataset) > 0 else 1)

        # evaluate
        model.eval()
        all_probs, all_lbls = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_lbls.append(yb.numpy().squeeze(1))
        y_proba = np.concatenate(all_probs)
        y_true = np.concatenate(all_lbls)

        if len(np.unique(y_true)) > 1:
            y_pred = (y_proba >= 0.5).astype(int)
            val_f1 = f1_score(y_true, y_pred)
            try:
                val_auc = roc_auc_score(y_true, y_proba)
            except Exception:
                val_auc = float("nan")
        else:
            val_f1 = 0.0
            val_auc = float("nan")

        print(f"Epoch {epoch}/{epochs}  loss:{avg_loss:.6f}  val_f1:{val_f1:.4f}  val_auc:{val_auc:.4f}")

        scheduler.step(val_f1)

        # early stopping & save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_ctr = 0
            meta = {"input_size": input_size, "hidden_size": hidden_size, "num_layers": num_layers, "seq_len": seq_len, "dropout": dropout}
            save_artifacts(models_dir, model, scaler, feature_cols, meta)
        else:
            patience_ctr += 1
            if patience_ctr >= patience_limit:
                print("Early stopping triggered. Stopping training.")
                break

    print("Training finished. Best val_f1:", best_val_f1)
    return model


# ---------------- Prediction helpers ----------------
def predict_sliding(csv_path: str, seq_len: int = 32, models_dir: str | None = None, device: str | None = None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models_dir = models_dir or os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models"))
    model, scaler, feature_cols, saved_seq_len = load_for_predict(models_dir, device)
    if seq_len != saved_seq_len:
        print(f"Warning: requested seq_len {seq_len} != saved {saved_seq_len}. Using saved.")
        seq_len = saved_seq_len

    df = pd.read_csv(csv_path)
    df_proc = add_features(preprocess_data(df.copy()), include_trend=False)

    missing = [c for c in feature_cols if c not in df_proc.columns]
    if missing:
        raise KeyError(f"Missing feature columns in data for prediction: {missing}")

    X = df_proc[feature_cols].replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    Xs = pd.DataFrame(scaler.transform(X), columns=feature_cols)
    ds = SequenceDataset(Xs, pd.Series([0] * len(Xs)), seq_len=seq_len)
    loader = DataLoader(ds, batch_size=256, shuffle=False, drop_last=False)

    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.extend(p.tolist())
    probs = np.array(probs)
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "predictions_lstm.csv")
    pd.DataFrame({"probability": probs}).to_csv(out_path, index=False)
    print("Saved sliding-window probabilities to:", out_path)
    return probs


def predict_one_per_machine(csv_path: str, seq_len: int = 32, models_dir: str | None = None, device: str | None = None, id_col: str = "machine_id"):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models_dir = models_dir or os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models"))
    model, scaler, feature_cols, saved_seq_len = load_for_predict(models_dir, device)
    seq_len = saved_seq_len

    df = pd.read_csv(csv_path)
    if id_col not in df.columns:
        raise KeyError(f"id_col '{id_col}' not found in CSV. Use sliding-window predict instead.")

    rows = []
    for mid, g in df.groupby(id_col):
        g = g.sort_values("Timestamp").reset_index(drop=True) if "Timestamp" in g.columns else g.reset_index(drop=True)
        if len(g) < seq_len:
            continue
        last = g.tail(seq_len)
        last_proc = add_features(preprocess_data(last.copy()), include_trend=False)
        missing = [c for c in feature_cols if c not in last_proc.columns]
        if missing:
            print(f"Skipping {mid}: missing features {missing}")
            continue
        X = last_proc[feature_cols].replace([float("inf"), float("-inf")], pd.NA).fillna(0)
        Xs = pd.DataFrame(scaler.transform(X), columns=feature_cols).values.astype("float32")
        xb = torch.from_numpy(Xs).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(xb)
            prob = torch.sigmoid(logit).item()
        rows.append({"machine_id": mid, "probability": prob})
    out = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "predictions_per_machine.csv")
    out.to_csv(out_path, index=False)
    print("Saved per-machine predictions to:", out_path)
    return out


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--holdout_frac", type=float, default=0.2)
    p.add_argument("--limit_rows", type=int, default=0)
    p.add_argument("--predict", action="store_true")
    p.add_argument("--predict_one_per_machine", action="store_true")
    p.add_argument("--id_col", type=str, default="machine_id")
    p.add_argument("--keep_features", type=str, default="", help="Comma-separated feature names to force-include (optional)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    csv_path = os.path.abspath(args.csv)
    keep_features = [s.strip() for s in args.keep_features.split(",")] if args.keep_features else None

    if args.predict:
        print("Running sliding-window prediction...")
        predict_sliding(csv_path, seq_len=args.seq_len)
        if args.predict_one_per_machine:
            try:
                predict_one_per_machine(csv_path, seq_len=args.seq_len, id_col=args.id_col)
            except Exception as e:
                print("predict_one_per_machine failed:", e)
    else:
        train_lstm(csv_path,
                   seq_len=args.seq_len,
                   epochs=args.epochs,
                   batch_size=args.batch_size,
                   hidden_size=args.hidden_size,
                   num_layers=args.num_layers,
                   dropout=args.dropout,
                   lr=args.lr,
                   holdout_frac=args.holdout_frac,
                   limit_rows=(args.limit_rows or None),
                   keep_features=keep_features)
