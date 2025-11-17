#!/usr/bin/env python3
"""
Simple working ML API for testing
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import joblib

app = Flask(__name__)
CORS(app)

# Load the LightGBM model
try:
    model_path = os.path.join(os.path.dirname(__file__), "models", "lgb_model.pkl")
    model = joblib.load(model_path)
    print(f"✅ LightGBM model loaded from {model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route("/")
def home():
    return jsonify({
        "message": "Predictive Maintenance API is running!",
        "model_loaded": model is not None,
        "endpoints": {
            "predict": "/predict (POST with JSON: {\"sensors\": {...}})"
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.json
        print(f"Received data: {data}")
        
        # Simple prediction - just return mock data for now
        # This will help us test the connection first
        return jsonify({
            "prediction": 0,
            "probability": 0.3,
            "risk_score": 0.3,
            "risk_level": "Low",
            "health_score": 70,
            "remaining_days": 25,
            "model_used": "LGB",
            "recommendations": [
                "Continue routine monitoring",
                "Adhere to scheduled maintenance",
                "Monitor for trend changes"
            ],
            "message": "Mock prediction - connection working!"
        })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    print("🚀 Starting Simple ML API...")
    app.run(host="127.0.0.1", port=5000, debug=True)

