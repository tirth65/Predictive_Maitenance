// import express from "express";
// import axios from "axios"; // to call Python ML API
// import Prediction from "../models/Prediction.js";

// const router = express.Router();

// // POST a new prediction (call Python ML service + save in DB)
// router.post("/", async (req, res) => {
//   try {
//     // Send input data to Python ML API (wrap in sensors object)
//     const mlRequest = { sensors: req.body };
//     const response = await axios.post("http://localhost:8000/predict", mlRequest);

//     // Extract prediction result
//     const result = response.data.prediction;

//     // Try to save input + result into MongoDB (optional)
//     try {
//       const prediction = new Prediction({
//         ...req.body,
//         result,
//       });
//       await prediction.save();
//       res.status(201).json(prediction);
//     } catch (dbError) {
//       // If database save fails, still return the prediction result
//       console.warn("⚠️ Database save failed, returning prediction without storage:", dbError.message);
//       res.status(200).json({
//         ...req.body,
//         result,
//         saved: false,
//         message: "Prediction completed but not saved to database"
//       });
//     }
//   } catch (error) {
//     console.error("❌ Prediction error:", error.message);
//     res.status(500).json({ error: "Prediction failed" });
//   }
// });

// // GET all predictions
// router.get("/", async (req, res) => {
//   try {
//     const predictions = await Prediction.find().sort({ createdAt: -1 });
//     res.json(predictions);
//   } catch (error) {
//     console.warn("⚠️ Database query failed:", error.message);
//     res.json({
//       message: "Database not available - no stored predictions",
//       predictions: []
//     });
//   }
// });

// export default router;

// backend/routes/predictionRoutes.js
import express from "express";
import axios from "axios";
import multer from "multer";
import FormData from "form-data";
import Prediction from "../models/prediction.js"; // your model
import SensorData from "../models/sensordata.js";

const router = express.Router();
const upload = multer(); // memory storage for file forwarding

// ML service URL (set in .env)
const ML_URL = process.env.ML_SERVICE_URL || process.env.VITE_ML_API_URL || "http://localhost:8000";

// POST /api/predict  -> expects { sensors: {temperature:..., vibration:..., ...}, machineId? }
router.post("/predict", async (req, res, next) => {
  try {
    const payload = req.body;
    if (!payload || !payload.sensors) {
      return res.status(400).json({ error: "Missing 'sensors' in request body" });
    }

    // Forward request to ML service
    const mlResp = await axios.post(`${ML_URL}/predict`, { sensors: payload.sensors }, { timeout: 15000 });
    const mlData = mlResp.data;

    // Save prediction to DB (optional)
    try {
      await Prediction.create({
        machineId: payload.machineId || payload.sensors.machineId || null,
        sensorSnapshot: payload.sensors,
        probability: mlData.probability ?? null,
        status: mlData.prediction === 1 || mlData.status === "Maintenance Required" ? "Maintenance Required" : "Healthy",
        timestamp: new Date(),
        modelVersion: mlData.modelVersion || "unknown"
      });
    } catch (saveErr) {
      console.warn("Warning: failed to save prediction to DB:", saveErr.message);
    }

    return res.json(mlData);
  } catch (err) {
    // pass through ML service error if available
    if (err.response && err.response.data) {
      return res.status(err.response.status || 500).json(err.response.data);
    }
    next(err);
  }
});

// POST /api/predict_file -> accepts multipart file and forwards to ML service
router.post("/predict_file", upload.single("file"), async (req, res, next) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No file uploaded" });

    const form = new FormData();
    form.append("file", req.file.buffer, { filename: req.file.originalname });

    const mlResp = await axios.post(`${ML_URL}/predict_file`, form, {
      headers: { ...form.getHeaders() },
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
      timeout: 60000
    });

    // Optionally save batch predictions - omitted here for brevity

    res.json(mlResp.data);
  } catch (err) {
    if (err.response && err.response.data) {
      return res.status(err.response.status || 500).json(err.response.data);
    }
    next(err);
  }
});

export default router;
