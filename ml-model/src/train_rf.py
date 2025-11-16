import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import preprocess_data
from src.features import add_features


# ============================================================
# CLEAN WORKING RANDOM FOREST TRAINER
# ============================================================
def train_random_forest(csv_path: str,
                        holdout_frac: float = 0.2,
                        n_estimators: int = 200,
                        max_depth=None,
                        plot=True):
    
    print("Loading data:", csv_path)
    df_raw = pd.read_csv(csv_path)

    # ---------------------- Split ----------------------
    split_idx = int(len(df_raw) * (1 - holdout_frac))
    df_train_raw = df_raw.iloc[:split_idx].reset_index(drop=True)
    df_test_raw = df_raw.iloc[split_idx:].reset_index(drop=True)
    print(f"Train rows: {len(df_train_raw)}, Test rows: {len(df_test_raw)}")

    # ---------------------- Preprocess ----------------------
    print("Preprocessing train...")
    df_train = add_features(preprocess_data(df_train_raw), include_trend=False)

    print("Preprocessing test...")
    df_test = add_features(preprocess_data(df_test_raw), include_trend=False)

    # ---------------------- Target ----------------------
    if "failure" in df_train.columns:
        TARGET_COL = "failure"
        print("Using processed target: failure")
    else:
        raise KeyError("Processed dataframe does NOT contain 'failure' column.")

    y_train = df_train[TARGET_COL]
    y_test = df_test[TARGET_COL]

    # ---------------------- Feature selection ----------------------
    # ---------------------- Feature selection (CLEAN: remove target-derived features) ----------------------
# Always drop any columns that contain substrings suggesting they are target-derived or leaked.
    leak_substrings = ["fail", "failure", "maintenance", "maint", "target", "cum", "_label", "label"]
# Also drop obvious timestamp or ID columns if present (not used as features here)
    drop_if_present = ["Timestamp", "timestamp", "time", "machine_id", "id"]

# start from processed df
    all_cols = list(df_train.columns)

# remove exact target column if present
    candidates_drop = set()
    if "failure" in all_cols:
        candidates_drop.add("failure")
# add any column that matches leak substrings
    for c in all_cols:
        lower = c.lower()
    for s in leak_substrings:
        if s in lower:
            candidates_drop.add(c)
            break
# remove common id/time cols
    for c in drop_if_present:
        if c in all_cols:
            candidates_drop.add(c)

# Now build feature lists
    feature_cols = [c for c in all_cols if c not in candidates_drop]
    print("Dropping leaked/target-like columns:", sorted(list(candidates_drop)))
    print("Using feature columns (sample 20):", feature_cols[:20])

    X_train = df_train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test  = df_test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)


    # Clean NaN/inf
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ---------------------- Pipeline ----------------------
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",   # handle 0/1 imbalance
            n_jobs=-1,
            random_state=42
        ))
    ])

    print("Training Random Forest...")
    pipeline.fit(X_train, y_train)

    # ---------------------- Predictions ----------------------
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    print("\n==== RESULTS ====")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("F1 Score:", f1_score(y_test, y_pred))

    # ---------------------- Plot ----------------------
    if plot:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Random Forest - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        models_dir = os.path.join("models")
        os.makedirs(models_dir, exist_ok=True)

        plot_path = os.path.join(models_dir, "rf_confusion_matrix.png")
        plt.savefig(plot_path, bbox_inches="tight")
        print("Saved confusion matrix to:", plot_path)
        plt.close()

    # ---------------------- Save Model ----------------------
    models_dir = os.path.join("models")
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(X_train.columns.tolist(), os.path.join(models_dir, "feature_columns.pkl"))
    joblib.dump(pipeline, os.path.join(models_dir, "rf_model.pkl"))

    print("Model saved at models/rf_model.pkl")

    return pipeline


# ============================================================
# MAIN RUNNER
# ============================================================
if __name__ == "__main__":
    train_random_forest(
        csv_path="./data/equipment_data.csv",
        holdout_frac=0.2,
        n_estimators=200,
        max_depth=None
    )
