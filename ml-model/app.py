# ------------------------ app.py (CLEAN + FIXED) ------------------------
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import traceback
from datetime import datetime, timedelta
import logging

# ---------------- Logger Setup ----------------
logger = logging.getLogger("ml-api")
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
logger.addHandler(console)

# ---------------- Flask App ----------------
app = Flask(__name__)
CORS(app)

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORT_PATH = os.path.join(MODELS_DIR, "load_report.txt")

sys.path.append(os.path.join(BASE_DIR, "src"))

from src.preprocess import preprocess_data
from src.features import add_features

lstm_model = None
lstm_scaler = None
lstm_feature_cols = None
lstm_seq_len = None
lstm_device = None

rf_model = None
rf_feature_cols = None

logger.info(f"Models directory: {MODELS_DIR}")
os.makedirs(MODELS_DIR, exist_ok=True)


def write_report(text):
    try:
        with open(REPORT_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.utcnow().isoformat()}Z - {text}\n")
    except:
        logger.warning("Failed to write to load_report.txt")


# ---------------- Load metadata ----------------
meta_path = os.path.join(MODELS_DIR, "lstm_meta.json")
meta = {}

if os.path.exists(meta_path):
    try:
        meta = json.load(open(meta_path, "r"))
        logger.info(f"Loaded lstm_meta.json: {meta}")
    except Exception as e:
        logger.error(f"Failed to load lstm_meta.json: {e}")
else:
    logger.warning("lstm_meta.json not found. Will infer metadata if needed.")

# ---------------- Load Feature Columns ----------------
feat_path = os.path.join(MODELS_DIR, "feature_columns.pkl")
if os.path.exists(feat_path):
    try:
        feature_cols = joblib.load(feat_path)
        # Some workspaces accidentally ship a truncated column file (only base sensors).
        # Treat anything extremely small as suspicious so we can fall back to metadata
        # coming from the serialized models themselves.
        if isinstance(feature_cols, (list, tuple)) and len(feature_cols) >= 16:
            lstm_feature_cols = list(feature_cols)
            rf_feature_cols = list(feature_cols)
            logger.info(f"Loaded {len(feature_cols)} feature columns.")
        else:
            logger.warning(
                "feature_columns.pkl contained only %s columns - ignoring and "
                "using model metadata instead.",
                len(feature_cols) if hasattr(feature_cols, "__len__") else "unknown",
            )
            feature_cols = None
    except Exception as e:
        lstm_feature_cols = None
        rf_feature_cols = None
        logger.error(f"Failed to load feature_columns.pkl: {e}")
else:
    logger.warning("feature_columns.pkl not found.")

# ---------------- Load Scaler ----------------
scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
if os.path.exists(scaler_path):
    try:
        lstm_scaler = joblib.load(scaler_path)
        logger.info("Scaler loaded.")
    except Exception as e:
        logger.error(f"Failed to load scaler.pkl: {e}")
else:
    logger.warning("scaler.pkl not found.")

# ---------------- Load LSTM Model ----------------
lstm_ckpt_path = os.path.join(MODELS_DIR, "lstm_model.pt")

if os.path.exists(lstm_ckpt_path):
    logger.info("Found lstm_model.pt, attempting to load...")

    try:
        import torch
        from src.lstm_model import LSTMRegressor, load_model as helper_load_model, predict_dataframe as helper_predict_dataframe

        try:
            # Use helper load if available
            model_obj, scaler_h, feature_h, seq_len_h, device_h = helper_load_model(MODELS_DIR)
            lstm_model = model_obj
            lstm_scaler = lstm_scaler or scaler_h
            lstm_feature_cols = lstm_feature_cols or feature_h
            lstm_seq_len = seq_len_h
            lstm_device = device_h
            logger.info("Loaded LSTM via helper loader.")
        except Exception as helperErr:
            logger.warning(f"Helper load failed: {helperErr}")

            ckpt = torch.load(lstm_ckpt_path, map_location="cpu")
            state_dict = ckpt.get("model_state_dict") or ckpt

            input_size = meta.get("input_size") or len(lstm_feature_cols)
            hidden_size = meta.get("hidden_size", 64)
            num_layers = meta.get("num_layers", 1)
            dropout = meta.get("dropout", 0.0)

            model = LSTMRegressor(
                input_size=int(input_size),
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                dropout=float(dropout) if num_layers > 1 else 0.0
            )

            model.load_state_dict(state_dict, strict=False)
            lstm_model = model
            lstm_seq_len = meta.get("seq_len")
            lstm_device = "cpu"
            logger.info("Loaded LSTM using fallback loader.")

        lstm_model.eval()

    except Exception as e:
        lstm_model = None
        logger.error(f"Failed to load LSTM: {e}")

else:
    logger.warning("No lstm_model.pt found; LSTM predictions will not work.")

# ---------------- Load Random Forest Model ----------------
rf_model_path = os.path.join(MODELS_DIR, "rf_model.pkl")
if os.path.exists(rf_model_path):
    try:
        rf_model = joblib.load(rf_model_path)
        logger.info("Loaded Random Forest model.")
        if not rf_feature_cols:
            names = getattr(rf_model, "feature_names_in_", None)
            if names is not None:
                rf_feature_cols = list(names)
                logger.info(f"Derived {len(rf_feature_cols)} RF feature columns from model metadata.")
    except Exception as e:
        rf_model = None
        logger.error(f"Failed to load rf_model.pkl: {e}")
else:
    logger.warning("rf_model.pkl not found. RF predictions disabled.")


# ---------------- Helper constants & utilities ----------------
SENSOR_KEYS = ["temperature", "vibration", "pressure", "rpm"]
SENSOR_ALIASES = {
    "temperature": ["temperature", "temp", "tempc", "temperaturec", "temperature_deg_c", "temperaturedegc"],
    "vibration": ["vibration", "vibe", "vibrationlevel", "vibration_mm_s", "vibrationmms"],
    "pressure": ["pressure", "press", "pressurepsi", "psi", "pressurebar", "bar"],
    "rpm": ["rpm", "speed", "rotationspeed", "rotationalspeed", "shaftspeed"],
}

RECOMMENDATIONS = {
    "High": [
        "Immediate inspection required",
        "Reduce load and schedule downtime",
        "Verify lubrication and alignment",
    ],
    "Medium": [
        "Monitor twice daily",
        "Plan maintenance within one week",
        "Check for abnormal vibration spikes",
    ],
    "Low": [
        "Continue normal operation",
        "Review weekly sensor trends",
        "Keep maintenance schedule on track",
    ],
}

HEURISTIC_RULES = {
    "temperature": {"start": 60.0, "span": 15.0, "weight": 0.25},
    "pressure": {"start": 45.0, "span": 10.0, "weight": 0.2},
    "vibration": {"start": 1.2, "span": 1.5, "weight": 0.45},
    "rpm": {"start": 1500.0, "span": 150.0, "weight": 0.1},
}

# ---------------- Helper for scaling sequences ----------------
def apply_lstm_scaler(seq):
    if lstm_scaler is None:
        return seq
    try:
        b, sl, f = seq.shape
        flat = seq.reshape(b * sl, f)
        scaled = lstm_scaler.transform(flat)
        return scaled.reshape(b, sl, f)
    except Exception as e:
        logger.warning(f"Scaler failed: {e}")
        return seq


def normalize_key(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())


def sanitize_number(value):
    if value in ("", None):
        return None
    try:
        num = float(value)
        if np.isnan(num):
            return None
        return float(num)
    except (TypeError, ValueError):
        return None


def normalize_sensors(payload: dict) -> dict:
    sensors = {}
    mapped = {}
    for key, value in (payload or {}).items():
        norm_key = normalize_key(key)
        mapped[norm_key] = sanitize_number(value)

    for field in SENSOR_KEYS:
        aliases = SENSOR_ALIASES.get(field, []) + [field]
        value = None
        for alias in aliases:
            if alias in mapped and mapped[alias] is not None:
                value = mapped[alias]
                break
        sensors[field] = value if value is not None else 0.0
    return sensors


def _excess_ratio(value, start, span):
    if value is None:
        return 0.0
    if value <= start:
        return 0.0
    return min(1.5, (value - start) / span)


def heuristic_probability_from_sensors(sensors: dict | None) -> float | None:
    if not sensors:
        return None

    cleaned = {k: sanitize_number(sensors.get(k)) for k in SENSOR_KEYS}
    score = 0.0
    for key, info in HEURISTIC_RULES.items():
        score += info["weight"] * _excess_ratio(cleaned.get(key), info["start"], info["span"])

    temp = cleaned.get("temperature")
    vib = cleaned.get("vibration")
    pressure = cleaned.get("pressure")
    rpm = cleaned.get("rpm")

    if temp is not None and vib is not None:
        if temp >= 80 and vib >= 3:
            score += 0.12
        elif temp >= 75 and vib >= 2.5:
            score += 0.1
        elif temp >= 70 and vib >= 2:
            score += 0.08
        elif vib >= 1.5 and temp >= 65:
            score += 0.05

    if pressure is not None:
        if pressure >= 58:
            score += 0.1
        elif pressure >= 52:
            score += 0.06
        elif pressure >= 47:
            score += 0.03

    if vib is not None and pressure is not None and vib >= 1.5 and pressure >= 47:
        score += 0.04

    if rpm is not None:
        if rpm >= 1650:
            score += 0.06
        elif rpm >= 1550:
            score += 0.03
        elif rpm >= 1500:
            score += 0.02

    score = max(0.0, min(score, 1.35))

    if score == 0.0:
        return 0.02

    probability = min(0.99, score ** 0.7)
    if score < 0.08:
        probability *= 0.6
    return float(probability)


def build_dataframe_from_sensors(sensors: dict, seq_len: int | None = None) -> pd.DataFrame:
    seq = seq_len or meta.get("seq_len") or 32
    seq = max(int(seq), 1)
    row = {k: sanitize_number(v) if sanitize_number(v) is not None else 0.0 for k, v in sensors.items()}

    df = pd.DataFrame([row] * seq)
    # create synthetic timestamp for feature builders
    now = datetime.utcnow()
    df["timestamp"] = [now - timedelta(minutes=5 * i) for i in range(seq)][::-1]
    # include default columns expected during training
    if "failure" not in df.columns:
        df["failure"] = 0
    return df


def get_rf_feature_columns():
    if rf_feature_cols:
        return rf_feature_cols
    names = getattr(rf_model, "feature_names_in_", None) if rf_model else None
    if names is not None:
        return list(names)
    return lstm_feature_cols


def prepare_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    df_proc = preprocess_data(df)
    df_feat = add_features(df_proc, include_trend=False)
    if "failure" in df_feat.columns:
        df_feat = df_feat.drop(columns=["failure"])
    target_cols = get_rf_feature_columns()
    if target_cols:
        df_feat = df_feat.reindex(columns=target_cols, fill_value=0)
    return df_feat.replace([np.inf, -np.inf], 0).fillna(0)


def derive_risk(probability: float):
    if probability is None:
        return {
            "risk_level": "Unknown",
            "health_score": None,
            "risk_score": None,
            "remaining_days": None,
            "recommendations": [],
        }
    risk_level = "Low"
    if probability > 0.66:
        risk_level = "High"
    elif probability > 0.33:
        risk_level = "Medium"
    health_score = max(0, min(100, round((1 - probability) * 100)))
    remaining_days = 0 if risk_level == "High" else 7 if risk_level == "Medium" else 30
    return {
        "risk_level": risk_level,
        "health_score": health_score,
        "risk_score": probability,
        "remaining_days": remaining_days,
        "recommendations": RECOMMENDATIONS[risk_level],
    }


def predict_rf_from_sensors(sensors: dict):
    heuristic_prob = heuristic_probability_from_sensors(sensors)

    if rf_model is None:
        if heuristic_prob is None:
            return None
        probability = heuristic_prob
        prediction = int(probability >= 0.5)
        risk = derive_risk(probability)
        return {
            "model_used": "Heuristic",
            "prediction": prediction,
            "probability": probability,
            "avg_prediction": probability,
            "binary_prediction": prediction,
            **risk,
            "all_predictions": {"heuristic": prediction},
            "all_probabilities": {"heuristic": probability},
            "heuristic_probability": probability,
        }

    df = build_dataframe_from_sensors(sensors)
    features = prepare_tabular_features(df)
    preds = rf_model.predict(features)
    probs = rf_model.predict_proba(features)[:, 1]
    probability = float(np.mean(probs))

    if heuristic_prob is not None:
        if probability is None:
            probability = heuristic_prob
        else:
            probability = float(
                min(0.99, max(0.0, probability * 0.2 + heuristic_prob * 0.8))
            )

    prediction = int(probability >= 0.5)
    risk = derive_risk(probability)
    return {
        "model_used": "RF",
        "prediction": prediction,
        "probability": probability,
        "avg_prediction": probability,
        "binary_prediction": prediction,
        **risk,
        "all_predictions": {"rf": prediction},
        "all_probabilities": {"rf": probability},
        "heuristic_probability": heuristic_prob,
    }


# ---------------- ENDPOINTS ----------------
@app.get("/")
def home():
    return jsonify({
        "message": "LSTM Predictive Maintenance API",
        "lstm_loaded": lstm_model is not None,
        "seq_len": lstm_seq_len,
        "input_size": len(lstm_feature_cols) if lstm_feature_cols else None,
        "rf_loaded": rf_model is not None,
        "features": len(rf_feature_cols) if rf_feature_cols else None,
    })


@app.post("/predict")
def predict():
    data = request.json
    if not data:
        return jsonify({"error": "Provide request body"}), 400

    # 1) Tabular sensors path (preferred for UI)
    if "sensors" in data:
        sensors = normalize_sensors(data["sensors"])
        rf_result = predict_rf_from_sensors(sensors)
        if rf_result:
            return jsonify(rf_result)
        # If RF unavailable, fall back to LSTM via synthetic sequence
        data["sensors_sequence"] = [list(sensors.values())] * (lstm_seq_len or 32)

    # 2) LSTM path (expects pre-built sequence)
    if lstm_model is None:
        return jsonify({"error": "No ML model available"}), 500

    if "sensors_sequence" not in data:
        return jsonify({"error": "Provide 'sensors' or 'sensors_sequence'"}), 400

    seq = np.array(data["sensors_sequence"], dtype=np.float32)

    if seq.ndim == 2:
        seq = seq[np.newaxis, ...]

    if seq.ndim != 3:
        return jsonify({"error": "Invalid shape"}), 400

    b, seq_len_r, input_size_r = seq.shape

    if lstm_seq_len and seq_len_r != lstm_seq_len:
        return jsonify({"error": f"Expected seq_len={lstm_seq_len}, got {seq_len_r}"}), 400

    if lstm_feature_cols and input_size_r != len(lstm_feature_cols):
        return jsonify({"error": f"Expected input_size={len(lstm_feature_cols)}, got {input_size_r}"}), 400

    seq = apply_lstm_scaler(seq)

    try:
        import torch
        device = lstm_device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = lstm_model.to(device)
        seq_t = torch.tensor(seq, dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(seq_t).cpu().numpy().tolist()

        avg = float(np.mean(out))
        binary = int(avg > 0.5)
        risk = derive_risk(avg)

        return jsonify({
            "model_used": "LSTM",
            "predictions": out,
            "avg_prediction": avg,
            "binary_prediction": binary,
            **risk,
            "probability": avg,
            "all_predictions": {"lstm": binary},
            "all_probabilities": {"lstm": avg},
        })

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return jsonify({"error": "Inference failed", "details": str(e)}), 500


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    logger.info("Starting ML server on port 8000...")
    app.run(host="0.0.0.0", port=8000, debug=True)

# ---------------------------------------------------------------------
