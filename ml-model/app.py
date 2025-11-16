# # # # app.py
# # # from fastapi import FastAPI
# # # from pydantic import BaseModel
# # # import pandas as pd
# # # import joblib
# # # import numpy as np
# # # from src.preprocess import preprocess_data
# # # from src.features import add_features

# # # # -----------------------------
# # # # Load model, scaler, features
# # # # -----------------------------
# # # MODEL_PATH = "models/predictive_model.pkl"
# # # SCALER_PATH = "models/scaler.pkl"
# # # FEATURES_PATH = "models/feature_columns.pkl"

# # # model = joblib.load(MODEL_PATH)
# # # scaler = joblib.load(SCALER_PATH)
# # # feature_columns = joblib.load(FEATURES_PATH)

# # # # -----------------------------
# # # # FastAPI setup
# # # # -----------------------------
# # # app = FastAPI(title="Predictive Maintenance API")

# # # # Pydantic model for input validation
# # # class SensorData(BaseModel):
# # #     temperature: float
# # #     vibration: float
# # #     pressure: float
# # #     rpm: float
# # #     temp_change: float = 0.0
# # #     vib_change: float = 0.0
# # #     Timestamp: str = None
# # #     torque: float = 0.0  # optional, for feature engineering

# # # # -----------------------------
# # # # Helper function
# # # # -----------------------------
# # # def prepare_input(data: pd.DataFrame) -> pd.DataFrame:
# # #     """Preprocess, feature engineer, scale, and align columns."""
# # #     df = preprocess_data(data)
# # #     df = add_features(df)

# # #     # Keep only trained features and fill missing with 0
# # #     df = df.reindex(columns=feature_columns, fill_value=0)

# # #     # Scale
# # #     df_scaled = scaler.transform(df)
# # #     return df_scaled

# # # # -----------------------------
# # # # API endpoint
# # # # -----------------------------
# # # @app.post("/predict")
# # # def predict(data: SensorData):
# # #     df = pd.DataFrame([data.dict()])
# # #     X_prepared = prepare_input(df)

# # #     prediction = model.predict(X_prepared)[0]
# # #     probability = model.predict_proba(X_prepared)[:, 1][0]

# # #     return {
# # #         "prediction": int(prediction),
# # #         "probability": float(probability)
# # #     }

# # # # -----------------------------
# # # # Run server
# # # # -----------------------------
# # # if __name__ == "__main__":
# # #     import uvicorn
# # #     uvicorn.run(app, host="0.0.0.0", port=8000)


# # from fastapi import FastAPI
# # from fastapi.middleware.cors import CORSMiddleware
# # import joblib
# # import os
# # import pandas as pd
# # from pydantic import BaseModel

# # class InputData(BaseModel):
# #     temperature: float
# #     pressure: float
# #     vibration: float
# #     humidity: float
# #     rpm: float
# #     torque: float
# #     timestamp: str
# #     # add raw feaures that your preprocess_data() + add_features() expect

# # app = FastAPI()

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # Build absolute path to models directory
# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # MODELS_DIR = os.path.join(BASE_DIR, "models")

# # # Load all models
# # models = {
# #     "rf": joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl")),
# #     "gb": joblib.load(os.path.join(MODELS_DIR, "gb_model.pkl")),
# #     "xgb": joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl")),
# # }

# # @app.post("/predict/{model_name}")
# # def predict(model_name: str, data: InputData):
# #     if model_name not in models:
# #         return {"error": f"Model '{model_name}' not found. Available: {list(models.keys())}"}
# #     
# #     df = pd.DataFrame([data.dict()])
# #     prediction = models[model_name].predict(df)
# #     return {"model": model_name, "prediction": prediction.tolist()}


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import os
# import joblib
# import pandas as pd

# app = Flask(__name__)
# CORS(app)  # Allow frontend (React) to communicate with backend

# # --- Load trained model and feature columns ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "models", "gb_model.pkl")
# FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")

# try:
# 	model = joblib.load(MODEL_PATH)
# 	print(f"Model loaded from {MODEL_PATH}")
# except FileNotFoundError:
# 	model = None
# 	print(f"Model not found at {MODEL_PATH}. Prediction API will not work until model is loaded.")

# try:
# 	feature_columns = joblib.load(FEATURES_PATH)
# 	if isinstance(feature_columns, dict) and "columns" in feature_columns:
# 		feature_columns = feature_columns["columns"]
# 	# Validate against model expectation
# 	n_expected = getattr(model, "n_features_in_", None) if model is not None else None
# 	if isinstance(feature_columns, (list, tuple)) and n_expected is not None and len(feature_columns) != n_expected:
# 		print(f"Warning: feature_columns length {len(feature_columns)} != model expects {n_expected}. Falling back to model order only.")
# 		feature_columns = None
# 	else:
# 		if isinstance(feature_columns, (list, tuple)):
# 			print(f"Feature columns loaded: {len(feature_columns)}")
# 		else:
# 			feature_columns = None
# 			print("feature_columns.pkl not in expected format. Using fallback.")
# except Exception:
# 	feature_columns = None
# 	print("feature_columns.pkl not found. Falling back to model.n_features_in_.")

# # --- Home route ---
# @app.route("/", methods=["GET"])
# def home():
# 	return jsonify({
# 		"message": "Predictive Maintenance API is running!",
# 		"endpoints": {
# 			"predict": "/predict (POST JSON: {\"sensors\": [...]})",
# 			"predict_file": "/predict_file (POST multipart/form-data with file=CSV)",
# 		}
# 	})


# def _expected_n_features():
# 	return getattr(model, "n_features_in_", None) if model is not None else None


# def _pad_or_trim(values: np.ndarray) -> np.ndarray:
# 	"""Ensure correct feature length by padding with zeros or trimming."""
# 	n = _expected_n_features()
# 	if n is None:
# 		return values.reshape(1, -1)
# 	# flatten to 1D then adjust
# 	arr = values.astype(float).ravel()
# 	if arr.size < n:
# 		arr = np.pad(arr, (0, n - arr.size), mode="constant")
# 	elif arr.size > n:
# 		arr = arr[:n]
# 	return arr.reshape(1, -1)


# def _align_features_from_row(row_like):
# 	"""Aligns incoming row (dict or pandas Series/list) to trained feature order.
# 	- If `feature_columns` list exists, map by column names when possible.
# 	- Otherwise, pad/trim to model.n_features_in_.
# 	"""
# 	if isinstance(row_like, dict):
# 		# Map frontend field names to training data column names
# 		field_mapping = {
# 			'temperature': 'Temperature',
# 			'vibration': 'Vibration', 
# 			'pressure': 'Pressure',
# 			'rpm': 'RPM'
# 		}
		
# 		# Convert frontend field names to training column names
# 		mapped_row = {}
# 		for frontend_key, value in row_like.items():
# 			training_key = field_mapping.get(frontend_key, frontend_key)
# 			mapped_row[training_key] = value
		
# 		if feature_columns is None:
# 			values = list(mapped_row.values())
# 			return _pad_or_trim(np.array(values, dtype=float))
# 		# map by column name; missing -> 0
# 		ordered = [mapped_row.get(col, 0) for col in feature_columns]
# 		return _pad_or_trim(np.array(ordered, dtype=float))

# 	# list/array pathway
# 	values = np.array(row_like, dtype=float)
# 	return _pad_or_trim(values)


# # --- Prediction API (single row) ---
# @app.route("/predict", methods=["POST"])
# def predict():
# 	if model is None:
# 		return jsonify({"error": "Model not loaded"}), 500
# 	try:
# 		data = request.get_json(silent=True) or {}
# 		sensors = data.get("sensors")
# 		if sensors is None:
# 			return jsonify({"error": "Body must include 'sensors' as list or object"}), 400

# 		X = _align_features_from_row(sensors)
# 		prediction = model.predict(X).tolist()[0]

# 		response = {"prediction": prediction}
# 		if hasattr(model, "predict_proba"):
# 			try:
# 				prob = float(model.predict_proba(X)[:, 1][0])
# 				response["probability"] = prob
# 			except Exception:
# 				pass
# 		return jsonify(response)

# 	except Exception as e:
# 		return jsonify({"error": str(e)}), 400


# # --- Batch Prediction via CSV upload ---
# @app.route("/predict_file", methods=["POST"])
# def predict_file():
# 	if model is None:
# 		return jsonify({"error": "Model not loaded"}), 500
# 	try:
# 		if "file" not in request.files:
# 			return jsonify({"error": "No file part in request"}), 400
# 		file = request.files["file"]
# 		if file.filename == "":
# 			return jsonify({"error": "No selected file"}), 400

# 		# Read CSV to DataFrame
# 		df = pd.read_csv(file)

# 		n = _expected_n_features() or df.shape[1]
# 		if feature_columns is not None:
# 			# Keep only trained features and fill missing with 0
# 			df_aligned = df.reindex(columns=feature_columns, fill_value=0)
# 			# If still mismatched, trim/pad
# 			if df_aligned.shape[1] != n:
# 				cols = list(df_aligned.columns)[:n]
# 				while len(cols) < n:
# 					cols.append(f"__pad_{len(cols)}")
# 				df_aligned = df_aligned.reindex(columns=cols, fill_value=0)
# 		else:
# 			# Fall back: take first n columns and pad with zeros if needed
# 			if df.shape[1] >= n:
# 				df_aligned = df.iloc[:, :n]
# 			else:
# 				df_aligned = df.copy()
# 				for i in range(df.shape[1], n):
# 					df_aligned[f"__pad_{i}"] = 0

# 		preds = model.predict(df_aligned)
# 		response = {
# 			"count": int(len(preds)),
# 			"predictions": preds.tolist(),
# 		}
# 		if hasattr(model, "predict_proba"):
# 			try:
# 				probs = model.predict_proba(df_aligned)[:, 1].tolist()
# 				response["probabilities"] = probs
# 			except Exception:
# 				pass

# 		return jsonify(response)

# 	except Exception as e:
# 		return jsonify({"error": str(e)}), 400


# # --- Run Server ---
# if __name__ == "__main__":
#     # force ML server to use port 8000 so it doesn't conflict with Node
#     app.run(host="0.0.0.0", port=8000, debug=True)
# # (from) D:\SGP_5\Predictive_Maitenance\ml-model

# ml-model/app.py
import os
import logging
from typing import Optional, List, Any, Dict

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ml-model")

app = Flask(__name__)
CORS(app)

# --- Paths & model metadata ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILENAME = "gb_model.pkl"  # adjust if different
FEATURES_FILENAME = "feature_columns.pkl"

MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)
FEATURES_PATH = os.path.join(MODELS_DIR, FEATURES_FILENAME)

MODEL_VERSION = MODEL_FILENAME

# --- Load model ---
model = None
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
except FileNotFoundError:
    logger.warning("Model file not found at %s. Prediction endpoints will return error.", MODEL_PATH)
except Exception as e:
    logger.exception("Failed to load model: %s", e)

# --- Load feature columns if present ---
feature_columns: Optional[List[str]] = None
try:
    if os.path.exists(FEATURES_PATH):
        feature_columns_raw = joblib.load(FEATURES_PATH)
        # feature_columns.pkl may be a dict like {"columns": [...]}
        if isinstance(feature_columns_raw, dict) and "columns" in feature_columns_raw:
            feature_columns = list(feature_columns_raw["columns"])
        elif isinstance(feature_columns_raw, (list, tuple, pd.Index)):
            feature_columns = list(feature_columns_raw)
        else:
            # try to coerce pandas Series
            try:
                feature_columns = list(pd.Index(feature_columns_raw))
            except Exception:
                feature_columns = None

        logger.info("Loaded feature_columns: %s (len=%s)", "(available)" if feature_columns else "(none)", len(feature_columns) if feature_columns else 0)
    else:
        logger.info("feature_columns.pkl not found at %s", FEATURES_PATH)
except Exception as e:
    feature_columns = None
    logger.exception("Error loading feature_columns: %s", e)

# Utility to get expected feature count from model
def expected_n_features() -> Optional[int]:
    return getattr(model, "n_features_in_", None) if model is not None else None

# Mapping from frontend field names -> training column names.
# Extend this mapping if your frontend uses different names.
FIELD_MAPPING: Dict[str, str] = {
    "temperature": "Temperature",
    "vibration": "Vibration",
    "pressure": "Pressure",
    "rpm": "RPM",
    "torque": "Torque",
    "humidity": "Humidity",
    # add more mappings as needed
}

def _to_number_safe(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _pad_or_trim_array(arr: np.ndarray) -> np.ndarray:
    """Pad with zeros or trim to expected_n_features()."""
    n = expected_n_features()
    arr = np.asarray(arr).astype(float).ravel()
    if n is None:
        return arr.reshape(1, -1)
    if arr.size < n:
        arr = np.pad(arr, (0, n - arr.size), mode="constant", constant_values=0.0)
    elif arr.size > n:
        arr = arr[:n]
    return arr.reshape(1, -1)

def _align_row_by_feature_columns(mapped_row: Dict[str, Any]) -> np.ndarray:
    """If feature_columns available, return ordered array matching them (missing -> 0)."""
    assert feature_columns is not None
    ordered = []
    for col in feature_columns:
        # try exact match
        if col in mapped_row:
            val = mapped_row[col]
        else:
            # try case-insensitive matches and FIELD_MAPPING inverse lookup
            key_found = None
            low_col = col.lower()
            for k in mapped_row.keys():
                if k.lower() == low_col:
                    key_found = k
                    break
            if key_found is not None:
                val = mapped_row[key_found]
            else:
                # no value provided: default 0
                val = 0
        ordered.append(_to_number_safe(val, 0.0))
    return _pad_or_trim_array(np.array(ordered, dtype=float))

def _normalize_and_map_input(row_like: Any) -> np.ndarray:
    """
    Accepts:
      - dict (key -> value)
      - list/tuple/np.ndarray (ordered values)
      - pandas Series/Row
    Returns a 2D numpy array ready for model (1 x n_features).
    """
    # dict pathway
    if isinstance(row_like, dict):
        # map front-end keys to training keys using FIELD_MAPPING
        mapped_row: Dict[str, Any] = {}
        for k, v in row_like.items():
            mapped_key = FIELD_MAPPING.get(k, k)  # if not in mapping, use as-is
            mapped_row[mapped_key] = v

        # If we have explicit feature_columns, use it to order values
        if feature_columns:
            return _align_row_by_feature_columns(mapped_row)

        # Else no feature_columns: try to rely on model.n_features_in_
        values = []
        # Keep order: try to prefer FIELD_MAPPING order if present
        # Combine mapped_row.values() as fallback
        if mapped_row:
            values = [_to_number_safe(v, 0.0) for v in mapped_row.values()]
        arr = np.array(values, dtype=float)
        return _pad_or_trim_array(arr)

    # list/array pathway
    if isinstance(row_like, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(row_like, dtype=float)
        return _pad_or_trim_array(arr)

    # unknown type: try to coerce into array
    try:
        arr = np.asarray(list(row_like), dtype=float)
        return _pad_or_trim_array(arr)
    except Exception:
        # fallback: empty zeros
        n = expected_n_features() or 0
        return np.zeros((1, n), dtype=float)

# --- Routes ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Predictive Maintenance ML API (Flask)",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION,
        "expected_features": expected_n_features(),
        "feature_columns_provided": bool(feature_columns)
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True, silent=True) or {}
        sensors = data.get("sensors")
        if sensors is None:
            return jsonify({"error": "Body must include 'sensors' (object or array)"}), 400

        X = _normalize_and_map_input(sensors)
        logger.info("Predict: input_shape=%s expected=%s", X.shape, expected_n_features())

        # model predict
        pred_raw = model.predict(X)
        prediction = int(pred_raw.tolist()[0]) if hasattr(pred_raw, "tolist") else int(pred_raw[0])

        response = {
            "prediction": prediction,
            "modelVersion": MODEL_VERSION
        }

        # probability if available
        if hasattr(model, "predict_proba"):
            try:
                prob = float(model.predict_proba(X)[:, 1][0])
                response["probability"] = prob
            except Exception as e:
                logger.warning("predict_proba failed: %s", e)

        return jsonify(response)
    except Exception as e:
        logger.exception("Predict error")
        return jsonify({"error": str(e)}), 500

@app.route("/predict_file", methods=["POST"])
def predict_file():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in request"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Read CSV into DataFrame
        df = pd.read_csv(file)
        logger.info("predict_file: received CSV with shape %s", df.shape)

        n_expected = expected_n_features() or df.shape[1]

        if feature_columns:
            # align to feature_columns (fill missing with 0)
            df_aligned = df.reindex(columns=feature_columns, fill_value=0)
            # if columns still fewer than expected, pad additional __pad columns
            if df_aligned.shape[1] < n_expected:
                for i in range(df_aligned.shape[1], n_expected):
                    df_aligned[f"__pad_{i}"] = 0
            # if more, trim
            if df_aligned.shape[1] > n_expected:
                df_aligned = df_aligned.iloc[:, :n_expected]
        else:
            # fallback: trim or pad DataFrame columns to n_expected
            if df.shape[1] >= n_expected:
                df_aligned = df.iloc[:, :n_expected].copy()
            else:
                df_aligned = df.copy()
                for i in range(df.shape[1], n_expected):
                    df_aligned[f"__pad_{i}"] = 0

        # Ensure numeric where possible
        df_numeric = df_aligned.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        preds = model.predict(df_numeric)
        response = {
            "count": int(len(preds)),
            "predictions": [int(x) for x in preds.tolist()],
            "modelVersion": MODEL_VERSION
        }
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(df_numeric)[:, 1].tolist()
                response["probabilities"] = [float(x) for x in probs]
            except Exception as e:
                logger.warning("predict_proba (batch) failed: %s", e)

        return jsonify(response)
    except Exception as e:
        logger.exception("predict_file error")
        return jsonify({"error": str(e)}), 500

# --- Run server ---
if __name__ == "__main__":
    # Run on port 8000 to avoid conflicts with Node backend on 5000
    logger.info("Starting ML server on port 8000")
    app.run(host="0.0.0.0", port=8000, debug=True)
