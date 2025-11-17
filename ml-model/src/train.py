import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix

# Optional dependencies – guard so Random Forest path still works without them
try:
    from imblearn.over_sampling import SMOTE  # noqa: F401
except Exception:
    SMOTE = None  # type: ignore

try:
    import xgboost as xgb  # noqa: F401
except Exception:
    xgb = None  # type: ignore

try:
    import lightgbm as lgb  # noqa: F401
except Exception:
    lgb = None  # type: ignore

try:
    from catboost import CatBoostClassifier  # noqa: F401
except Exception:
    CatBoostClassifier = None  # type: ignore

try:
    import optuna  # noqa: F401
except Exception:
    optuna = None  # type: ignore

import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import preprocess_data
from src.features import add_features

from sklearn.ensemble import RandomForestClassifier

# Optional: PyTorch for LSTM
try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = None  # type: ignore




# ==============================================================
# Random Forest Training (n_estimators=5)
# ==============================================================
def train_random_forest(
    csv_path: str,
    n_splits: int = 5,
    n_estimators: int = 5,   # ⬅️ Using 5 trees ("epochs")
    max_depth: int = None,
    limit_rows: int | None = None,
    plot: bool = False,
    holdout_frac: float = 0.0,
    optimize_threshold: bool = False,
    optimize_for: str = "f1",
):
    print(f"Loading data from: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    if limit_rows is not None and limit_rows > 0:
        df_raw = df_raw.head(limit_rows)
        print(f"Using top {len(df_raw)} rows for a quick run")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = None
    # Prepare model output directory early (also used to save plots)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.normpath(os.path.join(base_dir, "..", "models"))
    os.makedirs(models_dir, exist_ok=True)

    # Holdout mode: single chronological split
    if holdout_frac and holdout_frac > 0 and holdout_frac < 0.9:
        split_idx = int(len(df_raw) * (1.0 - holdout_frac))
        print(f"\n===== Random Forest Holdout ({(1-holdout_frac):.0%} train / {holdout_frac:.0%} test) =====")
        df_train_raw = df_raw.iloc[:split_idx].reset_index(drop=True)
        df_test_raw = df_raw.iloc[split_idx:].reset_index(drop=True)

        print("Preprocessing and feature engineering (train)...")
        df_train = add_features(preprocess_data(df_train_raw), include_trend=False)
        print("Preprocessing and feature engineering (test)...")
        df_test = add_features(preprocess_data(df_test_raw), include_trend=False)

        y_train = df_train["failure"]
        y_test = df_test["failure"]

        drop_cols = [c for c in df_train.columns if c.startswith("failure")]
        X_train = df_train.drop(columns=drop_cols, errors="ignore")
        X_test = df_test.drop(columns=drop_cols, errors="ignore")
        # Safety: remove inf/nan
        X_train = X_train.replace([float("inf"), float("-inf")], pd.NA).fillna(0)
        X_test = X_test.replace([float("inf"), float("-inf")], pd.NA).fillna(0)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ])

        print("Fitting model...")
        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        if optimize_threshold:
            thresholds = np.linspace(0.05, 0.95, 181)
            best_score = -1.0
            best_th = 0.5
            for th in thresholds:
                pred = (y_proba >= th).astype(int)
                if optimize_for == "accuracy":
                    score = (pred == y_test.values).mean()
                else:
                    score = f1_score(y_test, pred)
                if score > best_score:
                    best_score = score
                    best_th = th
            print(f"Best threshold by {optimize_for}: {best_th:.3f} ({optimize_for}={best_score:.3f})")
            y_pred = (y_proba >= best_th).astype(int)
        else:
            y_pred = (y_proba >= 0.5).astype(int)

        print(classification_report(y_test, y_pred))
        print("ROC-AUC:", roc_auc_score(y_test, y_proba))
        print("F1 Score:", f1_score(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        if plot:
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Random Forest Confusion Matrix - Holdout")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plot_path = os.path.join(models_dir, f"rf_confusion_matrix_holdout.png")
            try:
                plt.savefig(plot_path, bbox_inches="tight")
                print(f"Saved confusion matrix to: {plot_path}")
            except Exception:
                pass
            finally:
                plt.close()

        model = pipeline

    else:
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df_raw)):
            print(f"\n===== Random Forest Fold {fold+1} =====")
            df_train_raw = df_raw.iloc[train_idx].reset_index(drop=True)
            df_test_raw = df_raw.iloc[test_idx].reset_index(drop=True)

            print("Preprocessing and feature engineering (train)...")
            df_train = add_features(preprocess_data(df_train_raw), include_trend=False)
            print("Preprocessing and feature engineering (test)...")
            df_test = add_features(preprocess_data(df_test_raw), include_trend=False)

            y_train = df_train["failure"]
            y_test = df_test["failure"]

            drop_cols = [c for c in df_train.columns if c.startswith("failure")]
            X_train = df_train.drop(columns=drop_cols, errors="ignore")
            X_test = df_test.drop(columns=drop_cols, errors="ignore")
            # Safety: remove inf/nan
            X_train = X_train.replace([float("inf"), float("-inf")], pd.NA).fillna(0)
            X_test = X_test.replace([float("inf"), float("-inf")], pd.NA).fillna(0)

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                ))
            ])

            print("Fitting model...")
            pipeline.fit(X_train, y_train)

            y_proba = pipeline.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            print(classification_report(y_test, y_pred))
            print("ROC-AUC:", roc_auc_score(y_test, y_proba))
            print("F1 Score:", f1_score(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            if plot:
                plt.figure(figsize=(5,4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title(f"Random Forest Confusion Matrix - Fold {fold+1}")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plot_path = os.path.join(models_dir, f"rf_confusion_matrix_fold_{fold+1}.png")
                try:
                    plt.savefig(plot_path, bbox_inches="tight")
                    print(f"Saved confusion matrix to: {plot_path}")
                except Exception:
                    pass
                finally:
                    plt.close()

            model = pipeline

    feature_columns = X_train.columns.tolist()
    joblib.dump(feature_columns, os.path.join(models_dir, "feature_columns.pkl"))
    joblib.dump(model, os.path.join(models_dir, "rf_model.pkl"))
    print(f"✅ Random Forest model saved at {os.path.join(models_dir, 'rf_model.pkl')}")
    return model


# ==============================================================
# LightGBM Training
# ==============================================================
def train_lightgbm(
    csv_path: str,
    n_splits: int = 5,
    n_estimators: int = 2000,
    learning_rate: float = 0.05,
    num_leaves: int = 127,
    min_child_samples: int = 40,
    scale_pos_weight: float | None = None,
    limit_rows: int | None = None,
    plot: bool = False,
    holdout_frac: float = 0.0,
    optimize_threshold: bool = False,
    optimize_for: str = "f1",
):
    if lgb is None:
        raise ImportError("lightgbm is not installed. Install with: pip install lightgbm")

    print(f"Loading data from: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    if limit_rows is not None and limit_rows > 0:
        df_raw = df_raw.head(limit_rows)
        print(f"Using top {len(df_raw)} rows for a quick run")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = None

    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.normpath(os.path.join(base_dir, "..", "models"))
    os.makedirs(models_dir, exist_ok=True)

    if holdout_frac and holdout_frac > 0 and holdout_frac < 0.9:
        split_idx = int(len(df_raw) * (1.0 - holdout_frac))
        print(f"\n===== LightGBM Holdout ({(1-holdout_frac):.0%} train / {holdout_frac:.0%} test) =====")
        df_train_raw = df_raw.iloc[:split_idx].reset_index(drop=True)
        df_test_raw = df_raw.iloc[split_idx:].reset_index(drop=True)

        print("Preprocessing and feature engineering (train)...")
        df_train = add_features(preprocess_data(df_train_raw), include_trend=False)
        print("Preprocessing and feature engineering (test)...")
        df_test = add_features(preprocess_data(df_test_raw), include_trend=False)

        y_train = df_train["failure"]
        y_test = df_test["failure"]

        drop_cols = [c for c in df_train.columns if c.startswith("failure")]
        X_train = df_train.drop(columns=drop_cols, errors="ignore")
        X_test = df_test.drop(columns=drop_cols, errors="ignore")

        # Safety: remove inf/nan
        X_train = X_train.replace([float("inf"), float("-inf")], pd.NA).fillna(0)
        X_test = X_test.replace([float("inf"), float("-inf")], pd.NA).fillna(0)

        # Compute scale_pos_weight if not provided
        if scale_pos_weight is None:
            pos = max(1, int(y_train.sum()))
            neg = max(1, int((y_train == 0).sum()))
            spw = neg / pos
        else:
            spw = scale_pos_weight

        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=spw,
        )

        print("Fitting LightGBM...")
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        if optimize_threshold:
            thresholds = np.linspace(0.05, 0.95, 181)
            best_score = -1.0
            best_th = 0.5
            for th in thresholds:
                pred = (y_proba >= th).astype(int)
                if optimize_for == "accuracy":
                    score = (pred == y_test.values).mean()
                else:
                    score = f1_score(y_test, pred)
                if score > best_score:
                    best_score = score
                    best_th = th
            print(f"Best threshold by {optimize_for}: {best_th:.3f} ({optimize_for}={best_score:.3f})")
            y_pred = (y_proba >= best_th).astype(int)
        else:
            y_pred = (y_proba >= 0.5).astype(int)

        print(classification_report(y_test, y_pred))
        print("ROC-AUC:", roc_auc_score(y_test, y_proba))
        print("F1 Score:", f1_score(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        if plot:
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
            plt.title("LightGBM Confusion Matrix - Holdout")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plot_path = os.path.join(models_dir, f"lgb_confusion_matrix_holdout.png")
            try:
                plt.savefig(plot_path, bbox_inches="tight")
                print(f"Saved confusion matrix to: {plot_path}")
            except Exception:
                pass
            finally:
                plt.close()

    else:
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df_raw)):
            print(f"\n===== LightGBM Fold {fold+1} =====")
            df_train_raw = df_raw.iloc[train_idx].reset_index(drop=True)
            df_test_raw = df_raw.iloc[test_idx].reset_index(drop=True)

            print("Preprocessing and feature engineering (train)...")
            df_train = add_features(preprocess_data(df_train_raw), include_trend=False)
            print("Preprocessing and feature engineering (test)...")
            df_test = add_features(preprocess_data(df_test_raw), include_trend=False)

            y_train = df_train["failure"]
            y_test = df_test["failure"]

            drop_cols = [c for c in df_train.columns if c.startswith("failure")]
            X_train = df_train.drop(columns=drop_cols, errors="ignore")
            X_test = df_test.drop(columns=drop_cols, errors="ignore")

            # Safety: remove inf/nan
            X_train = X_train.replace([float("inf"), float("-inf")], pd.NA).fillna(0)
            X_test = X_test.replace([float("inf"), float("-inf")], pd.NA).fillna(0)

            # Compute scale_pos_weight if not provided
            if scale_pos_weight is None:
                pos = max(1, int(y_train.sum()))
                neg = max(1, int((y_train == 0).sum()))
                spw = neg / pos
            else:
                spw = scale_pos_weight

            model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary",
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=spw,
            )

            print("Fitting LightGBM...")
            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            print(classification_report(y_test, y_pred))
            print("ROC-AUC:", roc_auc_score(y_test, y_proba))
            print("F1 Score:", f1_score(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            if plot:
                plt.figure(figsize=(5,4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
                plt.title(f"LightGBM Confusion Matrix - Fold {fold+1}")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plot_path = os.path.join(models_dir, f"lgb_confusion_matrix_fold_{fold+1}.png")
                try:
                    plt.savefig(plot_path, bbox_inches="tight")
                    print(f"Saved confusion matrix to: {plot_path}")
                except Exception:
                    pass
                finally:
#                     plt.close()

    # Persist last model and feature columns
    feature_columns = X_train.columns.tolist()
    joblib.dump(feature_columns, os.path.join(models_dir, "feature_columns.pkl"))
    joblib.dump(model, os.path.join(models_dir, "lgb_model.pkl"))
    print(f"✅ LightGBM model saved at {os.path.join(models_dir, 'lgb_model.pkl')}")
    return model

