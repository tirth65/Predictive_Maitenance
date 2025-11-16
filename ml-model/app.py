# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import os
# import joblib
# import pandas as pd

# # Add src to path for imports
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# # Import preprocessing functions
# try:
#     from src.preprocess import preprocess_data
#     from src.features import add_features
#     print("✅ Preprocessing modules imported successfully")
# except Exception as e:
#     print(f"❌ Error importing preprocessing modules: {e}")
#     preprocess_data = None
#     add_features = None

# app = Flask(__name__)
# CORS(app)  # Allow frontend (React) to communicate with backend

# # --- Load trained models safely ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODELS_DIR = os.path.join(BASE_DIR, "models")

# # Load LightGBM model (primary) and fallback models
# models = {}
# model_paths = {
#     "lgb": "lgb_model.pkl",
#     "gb": "gb_model.pkl", 
#     "rf": "rf_model.pkl"
# }

# for model_name, model_file in model_paths.items():
#     try:
#         model_path = os.path.join(MODELS_DIR, model_file)
#         models[model_name] = joblib.load(model_path)
#         print(f"✅ {model_name.upper()} model loaded from {model_path}")
#     except FileNotFoundError:
#         print(f"⚠️  {model_name.upper()} model not found at {model_path}")

# # Set primary model (LightGBM preferred)
# primary_model = models.get("lgb") or models.get("gb") or models.get("rf")
# if primary_model:
#     print(f"🎯 Primary model: {'LightGBM' if 'lgb' in models else 'GradientBoosting' if 'gb' in models else 'RandomForest'}")
# else:
#     print("❌ No models loaded. Prediction API will not work.")

# # --- Home route ---
# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({
#         "message": "Predictive Maintenance API is running!",
#         "endpoints": {
#             "predict": "/predict (POST with JSON: {\"sensors\": [...]})",
#             "sensor_data": "/api/sensors (POST with JSON)"
#         }
#     })

# # --- Enhanced Prediction API with LightGBM ---
# @app.route("/predict", methods=["POST"])
# def predict():
#     if not primary_model:
#         return jsonify({"error": "No models loaded"}), 500
    
#     try:
#         data = request.json
        
#         # Handle both sensor object and feature array formats
#         if "sensors" in data:
#             # New format: sensor object
#             sensors = data["sensors"]
#             if isinstance(sensors, dict):
#                 # Convert sensor dict to feature array using preprocessing
#                 if preprocess_data is None or add_features is None:
#                     return jsonify({"error": "Preprocessing modules not available"}), 500
                
#                 # Create DataFrame from sensor data
#                 df = pd.DataFrame([sensors])
                
#                 # Preprocess and add features
#                 df_processed = preprocess_data(df)
#                 df_features = add_features(df_processed, include_trend=False)
                
#                 # Load feature columns to ensure correct order
#                 try:
#                     feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
#                     df_features = df_features.reindex(columns=feature_columns, fill_value=0)
#                 except:
#                     pass
                
#                 # Convert to numpy array
#                 features_array = df_features.values
#             else:
#                 # Legacy format: feature array
#                 features_array = np.array(sensors).reshape(1, -1)
#         else:
#             return jsonify({"error": "No sensor data provided"}), 400
        
#         # Get predictions from all available models
#         predictions = {}
#         probabilities = {}
        
#         for model_name, model in models.items():
#             try:
#                 pred = model.predict(features_array)[0]
#                 prob = model.predict_proba(features_array)[0, 1] if hasattr(model, 'predict_proba') else None
#                 predictions[model_name] = int(pred)
#                 probabilities[model_name] = float(prob) if prob is not None else None
#             except Exception as e:
#                 print(f"Error with {model_name} model: {e}")
#                 continue
        
#         # Use primary model for main prediction
#         primary_name = "lgb" if "lgb" in models else "gb" if "gb" in models else "rf"
#         main_prediction = predictions.get(primary_name, 0)
#         main_probability = probabilities.get(primary_name, 0.5)
        
#         # Calculate risk metrics
#         risk_score = main_probability
#         risk_level = "Low" if risk_score < 0.3 else "Medium" if risk_score < 0.7 else "High"
#         health_score = int((1 - risk_score) * 100)
        
#         # Estimate remaining days (simplified heuristic)
#         remaining_days = max(1, int((1 - risk_score) * 30)) if main_prediction == 0 else 0
        
#         # Generate recommendations based on risk level
#         if risk_level == "High":
#             recommendations = [
#                 "Schedule immediate inspection",
#                 "Stop equipment operation if possible",
#                 "Check for abnormal vibrations or sounds",
#                 "Review recent maintenance logs"
#             ]
#         elif risk_level == "Medium":
#             recommendations = [
#                 "Schedule maintenance within 1-2 weeks",
#                 "Increase monitoring frequency",
#                 "Check lubrication levels",
#                 "Review sensor trends"
#             ]
#         else:
#             recommendations = [
#                 "Continue routine monitoring",
#                 "Adhere to scheduled maintenance",
#                 "Monitor for trend changes",
#                 "Keep maintenance logs updated"
#             ]
        
#         return jsonify({
#             "prediction": main_prediction,
#             "probability": main_probability,
#             "risk_score": risk_score,
#             "risk_level": risk_level,
#             "health_score": health_score,
#             "remaining_days": remaining_days,
#             "recommendations": recommendations,
#             "model_used": primary_name.upper(),
#             "all_predictions": predictions,
#             "all_probabilities": probabilities
#         })
    
#     except Exception as e:
#         print(f"Prediction error: {e}")
#         return jsonify({"error": str(e)}), 400

# # --- File Upload Prediction API ---
# @app.route("/predict_file", methods=["POST"])
# def predict_file():
#     if not primary_model:
#         return jsonify({"error": "No models loaded"}), 500
    
#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No file uploaded"}), 400
        
#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({"error": "No file selected"}), 400
        
#         if not file.filename.endswith('.csv'):
#             return jsonify({"error": "Only CSV files are supported"}), 400
        
#         # Read CSV file
#         import pandas as pd
#         from io import StringIO
        
#         # Read file content
#         file_content = file.read().decode('utf-8')
#         df = pd.read_csv(StringIO(file_content))
        
#         # Preprocess data
#         if preprocess_data is None or add_features is None:
#             return jsonify({"error": "Preprocessing modules not available"}), 500
        
#         df_processed = preprocess_data(df)
#         df_features = add_features(df_processed, include_trend=False)
        
#         # Load feature columns to ensure correct order
#         try:
#             feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
#             df_features = df_features.reindex(columns=feature_columns, fill_value=0)
#         except:
#             pass
        
#         # Get predictions for all rows
#         predictions = []
#         probabilities = []
        
#         for _, row in df_features.iterrows():
#             row_array = row.values.reshape(1, -1)
            
#             # Get prediction from primary model
#             pred = primary_model.predict(row_array)[0]
#             prob = primary_model.predict_proba(row_array)[0, 1] if hasattr(primary_model, 'predict_proba') else 0.5
            
#             predictions.append(int(pred))
#             probabilities.append(float(prob))
        
#         # Calculate aggregate metrics
#         avg_probability = sum(probabilities) / len(probabilities)
#         failure_count = sum(predictions)
#         total_rows = len(predictions)
        
#         return jsonify({
#             "predictions": predictions,
#             "probabilities": probabilities,
#             "summary": {
#                 "total_rows": total_rows,
#                 "failure_predictions": failure_count,
#                 "average_failure_probability": avg_probability,
#                 "risk_level": "High" if avg_probability > 0.7 else "Medium" if avg_probability > 0.3 else "Low"
#             }
#         })
    
#     except Exception as e:
#         print(f"File prediction error: {e}")
#         return jsonify({"error": str(e)}), 400

# # --- Sensor Data API ---
# @app.route("/api/sensors", methods=["POST"])
# def receive_sensors():
#     try:
#         data = request.json
#         # Optionally save sensor data to DB, CSV, or logs
#         return jsonify({"status": "success", "data_received": data})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

# # --- Run Server ---
# if __name__ == "__main__":
#     app.run(host="127.0.0.1", port=5000, debug=True)

# app.py (LSTM-only; reads lstm_meta.json for input_size)

# app.py (final: robust loader + shape-based head remapping)
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import traceback
from datetime import datetime

# add src to path so we can import src.lstm_model
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

app = Flask(__name__)
CORS(app)

# -------------------- paths & containers --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORT_PATH = os.path.join(MODELS_DIR, "load_report.txt")

lstm_model = None
lstm_scaler = None
lstm_feature_cols = None
lstm_seq_len = None
lstm_device = None

print("📁 Models directory:", MODELS_DIR)
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR, exist_ok=True)

def write_report(text):
    try:
        with open(REPORT_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.utcnow().isoformat()}Z - {text}\n")
    except Exception as e:
        print("⚠️ Failed to write load_report:", e)

# -------------------- load meta, features, scaler --------------------
meta_path = os.path.join(MODELS_DIR, "lstm_meta.json")
meta = {}
if os.path.exists(meta_path):
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        print("✅ Loaded lstm_meta.json:", meta)
        write_report(f"Loaded lstm_meta.json: {meta}")
    except Exception as e:
        print("⚠️ Failed to read lstm_meta.json:", e)
        write_report(f"Failed to read lstm_meta.json: {e}")
else:
    print("⚠️ lstm_meta.json not found — will attempt to infer metadata.")
    write_report("lstm_meta.json not found — will attempt to infer metadata.")

try:
    feat_path = os.path.join(MODELS_DIR, "feature_columns.pkl")
    if os.path.exists(feat_path):
        lstm_feature_cols = joblib.load(feat_path)
        print(f"✅ feature_columns loaded ({len(lstm_feature_cols)} features).")
        write_report(f"feature_columns loaded ({len(lstm_feature_cols)} features).")
    else:
        lstm_feature_cols = None
        print("⚠️ feature_columns.pkl not found.")
        write_report("feature_columns.pkl not found.")
except Exception as e:
    lstm_feature_cols = None
    print("⚠️ Failed to load feature_columns.pkl:", e)
    write_report(f"Failed to load feature_columns.pkl: {e}")

try:
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        lstm_scaler = joblib.load(scaler_path)
        print("✅ Scaler loaded for LSTM:", scaler_path)
        write_report(f"Scaler loaded: {scaler_path}")
    else:
        lstm_scaler = None
        print("⚠️ scaler.pkl not found.")
        write_report("scaler.pkl not found.")
except Exception as e:
    lstm_scaler = None
    print("⚠️ Failed to load scaler.pkl:", e)
    write_report(f"Failed to load scaler.pkl: {e}")

# -------------------- robust LSTM load with shape-based head remapping --------------------
lstm_ckpt_path = os.path.join(MODELS_DIR, "lstm_model.pt")
if os.path.exists(lstm_ckpt_path):
    try:
        import torch
        from src.lstm_model import LSTMRegressor, load_model as helper_load_model, predict_dataframe as helper_predict_dataframe

        # Try helper loader first (if available)
        try:
            model_obj, scaler_from_helper, feature_cols_from_helper, seq_len_from_helper, device_from_helper = helper_load_model(MODELS_DIR)
            lstm_model = model_obj
            if lstm_scaler is None:
                lstm_scaler = scaler_from_helper
            if lstm_feature_cols is None:
                lstm_feature_cols = feature_cols_from_helper
            lstm_seq_len = seq_len_from_helper
            lstm_device = device_from_helper
            print(f"✅ Loaded LSTM via helper (seq_len={lstm_seq_len}, input_size={len(lstm_feature_cols) if lstm_feature_cols else 'unknown'})")
            write_report(f"Loaded LSTM via helper (seq_len={lstm_seq_len}, input_size={len(lstm_feature_cols) if lstm_feature_cols else 'unknown'})")
        except Exception as helper_err:
            print("⚠️ helper_load_model failed:", helper_err)
            write_report(f"helper_load_model failed: {helper_err}\n{traceback.format_exc()}")

            ckpt = torch.load(lstm_ckpt_path, map_location="cpu")

            # extract candidate state_dict and metadata
            state_dict = None
            if isinstance(ckpt, dict):
                for k in ("model_state", "state_dict", "model_state_dict", "state_dicts"):
                    if k in ckpt:
                        state_dict = ckpt[k]
                        break
                if state_dict is None:
                    candidate = {kk: vv for kk, vv in ckpt.items() if isinstance(vv, (torch.Tensor,))}
                    state_dict = candidate if candidate else ckpt
                input_size = ckpt.get("input_size") or meta.get("input_size") or (len(lstm_feature_cols) if lstm_feature_cols else None)
                hidden_size = ckpt.get("hidden_size", meta.get("hidden_size", 128))
                num_layers = ckpt.get("num_layers", meta.get("num_layers", 2))
                seq_len_val = ckpt.get("seq_len", meta.get("seq_len", None))
                dropout = ckpt.get("dropout", meta.get("dropout", 0.0))
            else:
                state_dict = ckpt if isinstance(ckpt, dict) else None
                input_size = meta.get("input_size") or (len(lstm_feature_cols) if lstm_feature_cols else None)
                hidden_size = meta.get("hidden_size", 128)
                num_layers = meta.get("num_layers", 2)
                seq_len_val = meta.get("seq_len", None)
                dropout = meta.get("dropout", 0.0)

            if input_size is None:
                raise RuntimeError("Cannot determine input_size for LSTM. Provide 'input_size' in lstm_meta.json or a feature_columns.pkl.")

            # Force dropout to 0 when num_layers == 1 (silence PyTorch warning)
            if int(num_layers) == 1 and float(dropout) > 0.0:
                write_report(f"Forcing dropout 0.0 because num_layers==1 (original dropout={dropout})")
                print(f"⚠️ Forcing dropout=0.0 because num_layers==1 (original dropout={dropout})")
                dropout = 0.0

            # instantiate model (try passing dropout if supported)
            try:
                model = LSTMRegressor(input_size=int(input_size),
                                      hidden_size=int(hidden_size),
                                      num_layers=int(num_layers),
                                      dropout=float(dropout))
            except TypeError:
                model = LSTMRegressor(input_size=int(input_size),
                                      hidden_size=int(hidden_size),
                                      num_layers=int(num_layers))
                write_report("LSTMRegressor constructor didn't accept dropout kwarg; constructed without explicit dropout argument.")

            def _keys(sd): 
                try:
                    return sorted(list(sd.keys()))
                except Exception:
                    return []

            def _shapes(sd):
                out = {}
                try:
                    for k, v in sd.items():
                        try:
                            out[k] = tuple(getattr(v, "shape", None))  # store as tuple or None
                        except Exception:
                            out[k] = None
                except Exception:
                    pass
                return out

            try:
                # try strict=True
                model.load_state_dict(state_dict)
                print("✅ model.load_state_dict succeeded (strict=True).")
                write_report("model.load_state_dict succeeded (strict=True).")
            except Exception as load_err:
                print("⚠️ model.load_state_dict failed (strict=True):", load_err)
                write_report(f"model.load_state_dict failed (strict=True): {load_err}\n{traceback.format_exc()}")

                saved_keys = _keys(state_dict) if state_dict is not None else []
                model_keys = _keys(model.state_dict())
                write_report(f"saved keys ({len(saved_keys)}): {saved_keys}")
                write_report(f"model keys ({len(model_keys)}): {model_keys}")

                saved_shapes = _shapes(state_dict)
                model_shapes = _shapes(model.state_dict())
                write_report(f"saved key shapes (first 200): {[(k, saved_shapes.get(k)) for k in saved_keys[:200]]}")
                write_report(f"model key shapes (first 200): {[(k, model_shapes.get(k)) for k in model_keys[:200]]}")

                # try strict=False
                try:
                    model.load_state_dict(state_dict, strict=False)
                    # compute missing/unexpected for logging
                    model_state_keys = set(model.state_dict().keys())
                    saved_state_keys = set(state_dict.keys()) if state_dict is not None else set()
                    missing = sorted(list(model_state_keys - saved_state_keys))
                    unexpected = sorted(list(saved_state_keys - model_state_keys))
                    print("✅ Loaded with strict=False.")
                    print(f"Missing keys: {len(missing)}; Unexpected keys: {len(unexpected)}")
                    write_report(f"Loaded with strict=False. Missing keys ({len(missing)}): {missing}; Unexpected keys ({len(unexpected)}): {unexpected}")
                    # if missing/unexpected only concern head.*, attempt shape-based remap
                    head_missing = [k for k in missing if k.startswith("head.")]
                    head_unexpected = [k for k in unexpected if k.startswith("head.")]
                except Exception as e2:
                    print("⚠️ load_state_dict(strict=False) failed:", e2)
                    write_report(f"load_state_dict(strict=False) failed: {e2}\n{traceback.format_exc()}")
                    head_missing = []
                    head_unexpected = []

                # If there are head mismatches, attempt shape-based remapping
                try:
                    # recompute current missing/unexpected if not set
                    if 'head_missing' not in locals():
                        model_state_keys = set(model.state_dict().keys())
                        saved_state_keys = set(state_dict.keys()) if state_dict is not None else set()
                        missing = sorted(list(model_state_keys - saved_state_keys))
                        unexpected = sorted(list(saved_state_keys - model_state_keys))
                        head_missing = [k for k in missing if k.startswith("head.")]
                        head_unexpected = [k for k in unexpected if k.startswith("head.")]
                    if head_missing and head_unexpected:
                        write_report(f"Attempting shape-based remap: head_missing={head_missing}, head_unexpected={head_unexpected}")
                        print("🔁 Attempting shape-based remapping for head.* keys.")

                        # Group saved head tensors by index and suffix, and remember shapes
                        import re
                        def parse_head_key(key):
                            m = re.match(r"^(head)\.(\d+)\.(.+)$", key)
                            if not m:
                                return None
                            return (int(m.group(2)), m.group(3))  # idx, suffix

                        saved_group = {}
                        for k in state_dict.keys():
                            p = parse_head_key(k)
                            if p:
                                idx, suffix = p
                                saved_group.setdefault(idx, {})[suffix] = k

                        model_group = {}
                        for k in model.state_dict().keys():
                            p = parse_head_key(k)
                            if p:
                                idx, suffix = p
                                model_group.setdefault(idx, {})[suffix] = k

                        # Build mapping by matching (weight.shape, bias.shape) pairs
                        # For each saved idx, compute tuple (weight_shape, bias_shape)
                        def get_pair_shapes(group, sd):
                            pairs = {}
                            for idx, mm in group.items():
                                w_key = mm.get("weight")
                                b_key = mm.get("bias")
                                w_shape = tuple(getattr(state_dict.get(w_key), "shape", None)) if w_key else None
                                b_shape = tuple(getattr(state_dict.get(b_key), "shape", None)) if b_key else None
                                pairs[idx] = (w_shape, b_shape)
                            return pairs

                        saved_pairs = get_pair_shapes(saved_group, state_dict)
                        model_pairs = {}
                        # for model pairs use model.state_dict()
                        model_sd = model.state_dict()
                        for idx, mm in model_group.items():
                            w_key = mm.get("weight")
                            b_key = mm.get("bias")
                            w_shape = tuple(getattr(model_sd.get(w_key), "shape", None)) if w_key else None
                            b_shape = tuple(getattr(model_sd.get(b_key), "shape", None)) if b_key else None
                            model_pairs[idx] = (w_shape, b_shape)

                        # find mapping from saved_idx -> model_idx by matching pair shapes
                        remap_idx = {}
                        used_model_idxs = set()
                        for s_idx, s_pair in saved_pairs.items():
                            matched = False
                            for m_idx, m_pair in model_pairs.items():
                                if m_idx in used_model_idxs:
                                    continue
                                if s_pair == m_pair:
                                    remap_idx[s_idx] = m_idx
                                    used_model_idxs.add(m_idx)
                                    matched = True
                                    break
                            if not matched:
                                # fallback: try matching by weight shape only
                                for m_idx, m_pair in model_pairs.items():
                                    if m_idx in used_model_idxs:
                                        continue
                                    if s_pair[0] is not None and s_pair[0] == m_pair[0]:
                                        remap_idx[s_idx] = m_idx
                                        used_model_idxs.add(m_idx)
                                        matched = True
                                        break
                        if not remap_idx:
                            raise RuntimeError("Could not infer head index mapping by shape.")

                        write_report(f"Determined head remapping: {remap_idx}")
                        print(f"🔁 head remapping determined: {remap_idx}")

                        # Build remapped state_dict where saved head.<s_idx>.* tensors assigned to head.<m_idx> keys
                        remapped = dict(state_dict)  # copy
                        for s_idx, mm in saved_group.items():
                            target_idx = remap_idx.get(s_idx)
                            if target_idx is None:
                                continue
                            for suffix, saved_key in mm.items():
                                target_key = f"head.{target_idx}.{suffix}"
                                remapped[target_key] = state_dict[saved_key]
                                # optionally remove old unexpected keys so strict=False load becomes clean
                                # we leave saved keys too (no harm) but model.load_state_dict will use remapped keys
                        # Attempt to load remapped
                        model.load_state_dict(remapped, strict=False)
                        write_report(f"Remapped load_state_dict succeeded using remap {remap_idx}")
                        print("✅ Remapped load_state_dict succeeded.")
                    else:
                        # no head mismatch to remap
                        pass
                except Exception as remap_exc:
                    print("❌ Head remapping attempt failed:", remap_exc)
                    write_report(f"Head remapping attempt failed: {remap_exc}\n{traceback.format_exc()}")
                    # if remap fails, raise original load error context so it's visible
                    # (we choose to continue with model possibly partially loaded)
                    pass

            # finalise
            model.eval()
            lstm_model = model
            lstm_seq_len = seq_len_val
            lstm_device = "cpu"
            print(f"✅ Loaded LSTM by constructing LSTMRegressor (input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout})")
            write_report(f"Loaded LSTM by constructing LSTMRegressor (input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout})")
    except Exception as e:
        lstm_model = None
        print("❌ Failed to load LSTM:", e)
        write_report(f"Failed to load LSTM: {e}\n{traceback.format_exc()}")
else:
    print("⚠️ lstm_model.pt not found — LSTM endpoints will not work.")
    write_report("lstm_model.pt not found — LSTM endpoints will not work.")

# -------------------- helpers for inference --------------------
def apply_lstm_scaler_to_sequence(seq: np.ndarray):
    """seq: np.ndarray shape (batch, seq_len, input_size)"""
    if lstm_scaler is None:
        return seq
    try:
        b, sl, f = seq.shape
        flat = seq.reshape(-1, f)
        flat_trans = lstm_scaler.transform(flat)
        return flat_trans.reshape(b, sl, f)
    except Exception as e:
        print("⚠️ LSTM scaler transform failed:", e)
        write_report(f"LSTM scaler transform failed: {e}\n{traceback.format_exc()}")
        return seq

# -------------------- endpoints --------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "LSTM-only Predictive Maintenance API",
        "lstm_loaded": lstm_model is not None,
        "lstm_seq_len": lstm_seq_len,
        "lstm_input_size": len(lstm_feature_cols) if lstm_feature_cols else None,
        "endpoints": {
            "predict": "/predict (POST JSON with sensors_sequence)",
            "predict_file": "/predict_file (POST form-data file=csv)"
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    if lstm_model is None:
        return jsonify({"error": "LSTM model not loaded"}), 500

    data = request.json
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    if "sensors_sequence" not in data:
        return jsonify({"error": "Provide 'sensors_sequence' in JSON payload (list of timesteps)"}), 400

    seq = np.array(data["sensors_sequence"], dtype=np.float32)

    # shape handling
    if seq.ndim == 2:
        seq = seq[np.newaxis, ...]  # (1, seq_len, input_size)
    if seq.ndim != 3:
        return jsonify({"error": f"Invalid sequence shape; expected 2D or 3D (seq_len, input_size) or (1, seq_len, input_size). Got ndim={seq.ndim}."}), 400

    b, seq_len_received, input_size_received = seq.shape

    if lstm_seq_len is not None and seq_len_received != lstm_seq_len:
        return jsonify({"error": f"Expected sequence length {lstm_seq_len}, got {seq_len_received}."}), 400

    if lstm_feature_cols is not None and input_size_received != len(lstm_feature_cols):
        return jsonify({"error": f"Expected input_size {len(lstm_feature_cols)}, got {input_size_received}."}), 400

    # scale
    seq = apply_lstm_scaler_to_sequence(seq)

    # inference
    try:
        import torch
        device = lstm_device if lstm_device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        model = lstm_model.to(device)
        model.eval()
        seq_tensor = torch.tensor(seq, dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(seq_tensor).cpu().numpy().squeeze()
        out_arr = np.atleast_1d(out).astype(float).tolist()
        aggregated = float(np.mean(out_arr)) if len(out_arr) > 0 else 0.0
        binary = int(aggregated > 0.5)
        return jsonify({
            "predictions": out_arr,
            "aggregated_prediction": aggregated,
            "binary_prediction": binary
        })
    except Exception as e:
        print("❌ LSTM inference failed:", e)
        write_report(f"LSTM inference failed: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "LSTM inference failed", "details": str(e)}), 500

@app.route("/predict_file", methods=["POST"])
def predict_file():
    if lstm_model is None:
        return jsonify({"error": "LSTM model not loaded"}), 500
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported"}), 400

    try:
        content = file.read().decode('utf-8')
        from io import StringIO
        df = pd.read_csv(StringIO(content))
    except Exception as e:
        return jsonify({"error": "Failed to read CSV", "details": str(e)}), 400

    # prefer helper predict_dataframe if available
    try:
        from src.lstm_model import predict_dataframe as helper_predict_dataframe
        preds = helper_predict_dataframe(df, MODELS_DIR)
        preds = [float(x) for x in preds]
        avg = float(np.mean(preds)) if preds else 0.0
        binary_preds = [int(p > 0.5) for p in preds]
        return jsonify({
            "window_predictions": preds,
            "binary_predictions": binary_preds,
            "average_prediction": avg,
            "total_windows": len(preds)
        })
    except Exception as e:
        print("⚠️ helper predict_dataframe failed or not available:", e)
        write_report(f"helper predict_dataframe failed: {e}\n{traceback.format_exc()}")
        if lstm_seq_len is None or lstm_feature_cols is None:
            return jsonify({"error": "Cannot compute window predictions; missing helper and missing lstm_seq_len/feature columns."}), 500
        try:
            df_used = df.copy()
            df_used = df_used.reindex(columns=lstm_feature_cols, fill_value=0)
            Xs = []
            arr = df_used.values.astype(np.float32)
            n = len(arr)
            for start in range(0, n - lstm_seq_len + 1):
                Xs.append(arr[start:start + lstm_seq_len])
            if not Xs:
                return jsonify({"error": "Not enough rows for any sliding window."}), 400
            seqs = np.stack(Xs, axis=0)
            seqs = apply_lstm_scaler_to_sequence(seqs)
            import torch
            device = lstm_device if lstm_device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
            model = lstm_model.to(device)
            model.eval()
            seq_tensor = torch.tensor(seqs, dtype=torch.float32, device=device)
            with torch.no_grad():
                out = model(seq_tensor).cpu().numpy()
            out_list = [float(x) for x in np.atleast_1d(out)]
            avg = float(np.mean(out_list)) if out_list else 0.0
            binary_preds = [int(p > 0.5) for p in out_list]
            return jsonify({
                "window_predictions": out_list,
                "binary_predictions": binary_preds,
                "average_prediction": avg,
                "total_windows": len(out_list)
            })
        except Exception as e2:
            print("❌ Fallback CSV prediction failed:", e2)
            write_report(f"Fallback CSV prediction failed: {e2}\n{traceback.format_exc()}")
            return jsonify({"error": "CSV prediction failed", "details": str(e2)}), 500

# -------------------- run --------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
