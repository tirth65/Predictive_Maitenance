

# train_model.py

import os
import argparse

# Import training functions from src/train.py
from src.train import train_random_forest
# (Add others here when ready: train_lightgbm, train_catboost, train_xgboost, etc.)


def main():
    parser = argparse.ArgumentParser(description="Train ML models for Predictive Maintenance")
    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        choices=["rf", "lgbm", "lstm"],
        help="Which model to train: rf | lgbm | lstm"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=os.path.join("data", "processed.csv"),
        help="Path to training CSV file"
    )
    parser.add_argument("--limit_rows", type=int, default=0, help="Limit rows for a quick run (0 = no limit)")
    parser.add_argument("--plot", action="store_true", help="Save confusion matrix plots")
    parser.add_argument("--holdout_frac", type=float, default=0.2, help="Fraction of latest data used for test (0 disables holdout)")
    parser.add_argument("--optimize_threshold", action="store_true", help="Optimize decision threshold for accuracy on holdout")
    parser.add_argument("--optimize_for", type=str, default="f1", choices=["accuracy", "f1"], help="Metric to optimize threshold for")
    # LSTM hyperparameters
    parser.add_argument("--seq_len", type=int, default=32, help="LSTM sequence length")
    parser.add_argument("--epochs", type=int, default=15, help="LSTM training epochs")
    parser.add_argument("--hidden_size", type=int, default=64, help="LSTM hidden size")
    parser.add_argument("--lstm_lr", type=float, default=1e-3, help="LSTM learning rate")
    args = parser.parse_args()

    print(f"🚀 Starting training with model: {args.model.upper()}")

    if args.model == "rf":
        model = train_random_forest(
            csv_path=args.csv,
            n_splits=5,
            n_estimators=5,
            max_depth=None,
            limit_rows=(args.limit_rows if args.limit_rows and args.limit_rows > 0 else None),
            plot=bool(args.plot),
            holdout_frac=(args.holdout_frac if args.holdout_frac and args.holdout_frac > 0 else 0.0),
            optimize_threshold=bool(args.optimize_threshold),
            optimize_for=args.optimize_for,
        )

    elif args.model == "lgbm":
        from src.train import train_lightgbm
        model = train_lightgbm(
            csv_path=args.csv,
            n_splits=5,
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=127,
            min_child_samples=40,
            scale_pos_weight=None,
            limit_rows=(args.limit_rows if args.limit_rows and args.limit_rows > 0 else None),
            plot=bool(args.plot),
            holdout_frac=(args.holdout_frac if args.holdout_frac and args.holdout_frac > 0 else 0.0),
            optimize_threshold=bool(args.optimize_threshold),
            optimize_for=args.optimize_for,
        )
    elif args.model == "lstm":
        from src.train import train_lstm
        model = train_lstm(
            csv_path=args.csv,
            holdout_frac=(args.holdout_frac if args.holdout_frac and args.holdout_frac > 0 else 0.2),
            seq_len=args.seq_len,
            epochs=args.epochs,
            batch_size=128,
            hidden_size=args.hidden_size,
            learning_rate=args.lstm_lr,
            optimize_threshold=True,
            optimize_for=args.optimize_for,
            limit_rows=(args.limit_rows if args.limit_rows and args.limit_rows > 0 else None),
            plot=bool(args.plot),
        )

    # Other models can be re-added here when ready

    else:
        raise ValueError(f"❌ Unknown model type: {args.model}")

    print("✅ Training finished successfully!")


if __name__ == "__main__":
    main()
