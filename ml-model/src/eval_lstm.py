# src/eval_lstm.py
import os, json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix, precision_recall_curve, auc
import torch
from src.preprocess import preprocess_data
from src.features import add_features
from src.train_lstm import load_for_predict  # uses function from train_lstm module

def load_df(csv_path):
    df = pd.read_csv(csv_path)
    return df

def sliding_predict(csv_path, models_dir="models"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler, feature_cols, saved_seq_len = load_for_predict(models_dir, device)
    df = pd.read_csv(csv_path)
    df_proc = add_features(preprocess_data(df.copy()), include_trend=False)
    # ensure features present
    missing = [c for c in feature_cols if c not in df_proc.columns]
    if missing:
        raise KeyError("Missing feature columns:", missing)
    X = df_proc[feature_cols].replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    Xs = pd.DataFrame(scaler.transform(X), columns=feature_cols)
    # build sliding windows
    seq_len = saved_seq_len
    n_windows = max(0, len(Xs) - seq_len + 1)
    probs = []
    idxs = []
    with torch.no_grad():
        for i in range(n_windows):
            block = Xs.iloc[i:i+seq_len].values.astype("float32")
            xb = torch.from_numpy(block).unsqueeze(0).to(device)
            logit = model(xb)
            p = torch.sigmoid(logit).cpu().numpy().item()
            probs.append(p)
            idxs.append(i + seq_len - 1)  # label aligned to last timestep
    out_df = pd.DataFrame({"index": idxs, "prob": probs})
    return out_df, df_proc

def find_best_threshold_and_report(out_df, df_proc, target_col="failure"):
    # align probs with ground truth labels at the last timestep index
    y_true = df_proc[target_col].values
    indices = out_df["index"].values
    y_true_win = y_true[indices]
    y_proba = out_df["prob"].values

    # find best threshold by F1
    best_th = 0.5
    best_f1 = -1
    for th in np.linspace(0.01, 0.99, 99):
        y_pred = (y_proba >= th).astype(int)
        f = f1_score(y_true_win, y_pred)
        if f > best_f1:
            best_f1 = f
            best_th = th

    y_pred_best = (y_proba >= best_th).astype(int)
    print("Best threshold by F1:", best_th, "F1:", best_f1)
    print("Classification report (best-threshold):")
    print(classification_report(y_true_win, y_pred_best))
    print("ROC-AUC:", roc_auc_score(y_true_win, y_proba))
    cm = confusion_matrix(y_true_win, y_pred_best)
    print("Confusion matrix:\n", cm)

    # PR-AUC
    precision, recall, _ = precision_recall_curve(y_true_win, y_proba)
    pr_auc = auc(recall, precision)
    print("PR-AUC:", pr_auc)

    # Save CSV of predictions (prob + pred + true)
    save_df = pd.DataFrame({"idx": indices, "prob": y_proba, "pred": y_pred_best, "true": y_true_win})
    save_path = os.path.join("data", "predictions_lstm_eval.csv")
    save_df.to_csv(save_path, index=False)
    print("Saved eval predictions to:", save_path)

    # plot ROC
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true_win, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true_win, y_proba):.3f}")
        plt.plot([0,1],[0,1],"--", color="gray")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC")
        plt.legend()
        plt.savefig(os.path.join("models", "roc_eval.png"))
        plt.close()
        print("Saved ROC plot -> models/roc_eval.png")
    except Exception as e:
        print("Could not plot ROC:", e)

if __name__ == "__main__":
    csv = os.path.join("data", "equipment_data.csv")
    out_df, df_proc = sliding_predict(csv, models_dir="models")
    find_best_threshold_and_report(out_df, df_proc, target_col="failure")
