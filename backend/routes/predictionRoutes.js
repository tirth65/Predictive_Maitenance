// // // import express from "express";
// // // import axios from "axios"; // to call Python ML API
// // // import Prediction from "../models/Prediction.js";

// // // const router = express.Router();

// // // // POST a new prediction (call Python ML service + save in DB)
// // // router.post("/", async (req, res) => {
// // //   try {
// // //     // Send input data to Python ML API (wrap in sensors object)
// // //     const mlRequest = { sensors: req.body };
// // //     const response = await axios.post("http://localhost:8000/predict", mlRequest);

// // //     // Extract prediction result
// // //     const result = response.data.prediction;

// // //     // Try to save input + result into MongoDB (optional)
// // //     try {
// // //       const prediction = new Prediction({
// // //         ...req.body,
// // //         result,
// // //       });
// // //       await prediction.save();
// // //       res.status(201).json(prediction);
// // //     } catch (dbError) {
// // //       // If database save fails, still return the prediction result
// // //       console.warn("⚠️ Database save failed, returning prediction without storage:", dbError.message);
// // //       res.status(200).json({
// // //         ...req.body,
// // //         result,
// // //         saved: false,
// // //         message: "Prediction completed but not saved to database"
// // //       });
// // //     }
// // //   } catch (error) {
// // //     console.error("❌ Prediction error:", error.message);
// // //     res.status(500).json({ error: "Prediction failed" });
// // //   }
// // // });

// // // // GET all predictions
// // // router.get("/", async (req, res) => {
// // //   try {
// // //     const predictions = await Prediction.find().sort({ createdAt: -1 });
// // //     res.json(predictions);
// // //   } catch (error) {
// // //     console.warn("⚠️ Database query failed:", error.message);
// // //     res.json({
// // //       message: "Database not available - no stored predictions",
// // //       predictions: []
// // //     });
// // //   }
// // // });

// // // export default router;

// // // backend/routes/predictionRoutes.js
// // import express from "express";
// // import axios from "axios";
// // import multer from "multer";
// // import FormData from "form-data";
// // import Prediction from "../models/prediction.js"; // your model
// // // import SensorData from "../models/sensordata.js";

// // const router = express.Router();
// // const upload = multer(); // memory storage for file forwarding

// // // ML service URL (set in .env)
// // const ML_URL = process.env.ML_SERVICE_URL || process.env.VITE_ML_API_URL || "http://localhost:8000";

// // // POST /api/predict  -> expects { sensors: {temperature:..., vibration:..., ...}, machineId? }
// // router.post("/predict", async (req, res, next) => {
// //   try {
// //     const payload = req.body;
// //     if (!payload || !payload.sensors) {
// //       return res.status(400).json({ error: "Missing 'sensors' in request body" });
// //     }

// //     // Forward request to ML service
// //     const mlResp = await axios.post(`${ML_URL}/predict`, { sensors: payload.sensors }, { timeout: 15000 });
// //     const mlData = mlResp.data;

// //     // Save prediction to DB (optional)
// //     try {
// //       await Prediction.create({
// //         machineId: payload.machineId || payload.sensors.machineId || null,
// //         sensorSnapshot: payload.sensors,
// //         probability: mlData.probability ?? null,
// //         status: mlData.prediction === 1 || mlData.status === "Maintenance Required" ? "Maintenance Required" : "Healthy",
// //         timestamp: new Date(),
// //         modelVersion: mlData.modelVersion || "unknown"
// //       });
// //     } catch (saveErr) {
// //       console.warn("Warning: failed to save prediction to DB:", saveErr.message);
// //     }

// //     return res.json(mlData);
// //   } catch (err) {
// //     // pass through ML service error if available
// //     if (err.response && err.response.data) {
// //       return res.status(err.response.status || 500).json(err.response.data);
// //     }
// //     next(err);
// //   }
// // });

// // // POST /api/predict_file -> accepts multipart file and forwards to ML service
// // router.post("/predict_file", upload.single("file"), async (req, res, next) => {
// //   try {
// //     if (!req.file) return res.status(400).json({ error: "No file uploaded" });

// //     const form = new FormData();
// //     form.append("file", req.file.buffer, { filename: req.file.originalname });

// //     const mlResp = await axios.post(`${ML_URL}/predict_file`, form, {
// //       headers: { ...form.getHeaders() },
// //       maxContentLength: Infinity,
// //       maxBodyLength: Infinity,
// //       timeout: 60000
// //     });

// //     // Optionally save batch predictions - omitted here for brevity

// //     res.json(mlResp.data);
// //   } catch (err) {
// //     if (err.response && err.response.data) {
// //       return res.status(err.response.status || 500).json(err.response.data);
// //     }
// //     next(err);
// //   }
// // });

// // export default router;


// // backend/routes/predictionRoutes.js
// import express from "express";
// import axios from "axios";

// const predictionRoutes = express.Router();

// // ML server URL (adjust if different)
// const ML_BASE = process.env.ML_API_URL || "http://localhost:8000";

// let mlInfoCache = null;
// let mlInfoFetchedAt = 0;
// const ML_INFO_TTL_MS = 60 * 1000; // re-fetch ML root info once per minute

// async function getMlInfo() {
//   const now = Date.now();
//   if (mlInfoCache && now - mlInfoFetchedAt < ML_INFO_TTL_MS) return mlInfoCache;
//   try {
//     const resp = await axios.get(`${ML_BASE}/`);
//     mlInfoCache = resp.data || {};
//     mlInfoFetchedAt = now;
//     return mlInfoCache;
//   } catch (err) {
//     // If ML root not available, still allow fallback
//     mlInfoCache = {};
//     mlInfoFetchedAt = now;
//     return mlInfoCache;
//   }
// }

// // POST /api/predictions/predict
// router.post("/predict", async (req, res) => {
//   try {
//     const { sensors } = req.body || {};

//     if (!sensors || (typeof sensors !== "object")) {
//       return res.status(400).json({ error: "Request body must include 'sensors' object" });
//     }

//     // Example mapping order — adjust if your LSTM expects a different order
//     // Use the same order you trained your model on. This example uses:
//     // [temperature, vibration, pressure, rpm]
//     const sensorOrder = ["temperature", "vibration", "pressure", "rpm"];

//     // Build one numeric row array from sensors (fill missing with 0)
//     const row = sensorOrder.map((k) => {
//       const v = sensors[k];
//       return v === undefined || v === null ? 0 : Number(v);
//     });

//     // Get ML info (seq_len)
//     const info = await getMlInfo();
//     const seq_len = info.lstm_seq_len || info.seq_len || info.expected_features && null; // fallback attempt
//     // If ML doesn't expose seq_len, default to 32 (common) — adjust if you know exact seq_len
//     const final_seq_len = typeof seq_len === "number" ? seq_len : 32;

//     // Build sensors_sequence by repeating the single row final_seq_len times
//     const sensors_sequence = Array.from({ length: final_seq_len }, () => row);

//     // Send to ML server
//     const mlResp = await axios.post(`${ML_BASE}/predict`, { sensors_sequence }, { timeout: 20000 });

//     // Forward ML response
//     return res.json(mlResp.data);
//   } catch (err) {
//     console.error("/api/predictions/predict error:", err.message || err);
//     // If ML returned an HTTP error, forward message and status if available
//     if (err.response && err.response.data) {
//       const status = err.response.status || 500;
//       return res.status(status).json(err.response.data);
//     }
//     return res.status(500).json({ error: "Internal server error", details: err.message });
//   }
// });

// export default predictionRoutes ;

// backend/routes/predictionRoutes.js
import express from "express";
import axios from "axios";
import multer from "multer";
import Papa from "papaparse";

const router = express.Router();
const upload = multer(); // in-memory storage

const ML_BASE = process.env.ML_API_URL || "http://localhost:8000";
const ML_TIMEOUT_MS = Math.max(1000, Number(process.env.ML_TIMEOUT_MS || 3500));
const SENSOR_KEYS = ["temperature", "vibration", "pressure", "rpm"];
const SENSOR_ALIASES = {
  temperature: ["temperature", "temp", "tempc", "temperaturec", "temperature_degc", "temperaturedegc"],
  vibration: ["vibration", "vibrationlevel", "vibration_mm_s", "vibrationmms", "vibe"],
  pressure: ["pressure", "press", "pressurepsi", "psi", "pressurebar", "bar"],
  rpm: ["rpm", "speed", "rotationspeed", "rotationalspeed", "shaftspeed"],
};

const FILE_PREDICTION_CONCURRENCY = Math.max(
  1,
  Number(process.env.FILE_PREDICT_CONCURRENCY || process.env.FILE_PREDICT_POOL || 6)
);
const MAX_FILE_ROWS = Math.max(
  1,
  Number(process.env.MAX_FILE_PREDICTION_ROWS || process.env.FILE_PREDICT_MAX_ROWS || 750)
);
const FILE_FAST_MODE = (process.env.FILE_PREDICT_MODE || "heuristic").toLowerCase();
const FILE_FAST_THRESHOLD = Math.max(
  1,
  Number(process.env.FILE_PREDICT_FAST_THRESHOLD || process.env.FILE_PREDICT_FAST_ROWS || 80)
);
const FILE_FAST_CONCURRENCY = Math.max(
  1,
  Number(process.env.FILE_PREDICT_FAST_CONCURRENCY || process.env.FILE_PREDICT_FAST_POOL || FILE_PREDICTION_CONCURRENCY)
);

const RECOMMENDATIONS = {
  High: [
    "Immediate inspection required",
    "Reduce load and schedule downtime",
    "Verify lubrication and alignment",
  ],
  Medium: [
    "Monitor twice daily",
    "Plan maintenance within one week",
    "Check for abnormal vibration spikes",
  ],
  Low: [
    "Continue normal operation",
    "Review weekly sensor trends",
    "Keep maintenance schedule on track",
  ],
};

const HEURISTIC_RULES = {
  temperature: { start: 60.0, span: 15.0, weight: 0.25 },
  pressure: { start: 45.0, span: 10.0, weight: 0.2 },
  vibration: { start: 1.2, span: 1.5, weight: 0.45 },
  rpm: { start: 1500.0, span: 150.0, weight: 0.1 },
};

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function normalizeKey(key = "") {
  return key.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function sanitizeNumber(value) {
  if (value === "" || value === null || value === undefined) return undefined;
  const num = Number(value);
  return Number.isFinite(num) ? num : undefined;
}

function normalizeSensorsInput(input = {}) {
  const source = input.sensors && typeof input.sensors === "object" ? input.sensors : input;
  const normalizedEntries = {};

  Object.entries(source || {}).forEach(([key, value]) => {
    const normKey = normalizeKey(key);
    if (!normKey) return;
    normalizedEntries[normKey] = sanitizeNumber(value);
  });

  const sensors = {};
  SENSOR_KEYS.forEach((field) => {
    const aliases = SENSOR_ALIASES[field] || [field];
    let foundValue;
    for (const alias of aliases) {
      if (Object.prototype.hasOwnProperty.call(normalizedEntries, alias)) {
        foundValue = normalizedEntries[alias];
        break;
      }
    }
    sensors[field] = foundValue;
  });

  return sensors;
}

function buildRowFromSensors(sensors) {
  return SENSOR_KEYS.map((key) => {
    const v = sanitizeNumber(sensors?.[key]);
    return v === undefined ? 0 : v;
  });
}

function deriveInsights(probability, prediction) {
  const prob = typeof probability === "number" && !Number.isNaN(probability) ? probability : null;
  const pred = typeof prediction === "number" ? (prediction >= 0.5 ? 1 : 0) : 0;
  const maintenanceNeeded = pred === 1;

  let riskLevel;
  if (prob === null) {
    riskLevel = maintenanceNeeded ? "High" : "Low";
  } else if (prob > 0.66) {
    riskLevel = "High";
  } else if (prob > 0.33) {
    riskLevel = "Medium";
  } else {
    riskLevel = "Low";
  }

  const healthScore = prob === null ? null : Math.max(0, Math.min(100, Math.round((1 - prob) * 100)));
  const remainingDays = riskLevel === "High" ? 0 : riskLevel === "Medium" ? 7 : 30;
  const recommendations = RECOMMENDATIONS[riskLevel] || RECOMMENDATIONS.Low;

  return { maintenanceNeeded, riskLevel, healthScore, remainingDays, recommendations, riskScore: prob };
}

function normalizePredictionValue(value, probability) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    if (typeof probability === "number") {
      return probability >= 0.5 ? 1 : 0;
    }
    return 0;
  }
  const num = Number(value);
  if (Number.isNaN(num)) return 0;
  return num >= 0.5 ? 1 : 0;
}

function formatMlResponse(mlData, overrides = {}) {
  const probability =
    typeof mlData?.probability === "number"
      ? mlData.probability
      : typeof mlData?.avg_prediction === "number"
        ? mlData.avg_prediction
        : null;

  const prediction = typeof mlData?.prediction === "number"
    ? Number(mlData.prediction)
    : normalizePredictionValue(
        mlData?.binary_prediction ?? mlData?.avg_prediction,
        probability
      );

  const mlHealth = mlData?.health_score ?? mlData?.healthScore;
  const mlRisk = mlData?.risk_level ?? mlData?.riskLevel;
  const mlRemaining = mlData?.remaining_days ?? mlData?.remainingDays;
  const mlRecommendations = mlData?.recommendations;
  const mlMaintenance = mlData?.maintenance_needed ?? mlData?.maintenanceNeeded;
  const mlRiskScore = mlData?.risk_score ?? mlData?.riskScore;
  const modelKey = (mlData?.model_used || mlData?.modelUsed || "model").toLowerCase();

  const insights = mlHealth !== undefined || mlRisk !== undefined
    ? {
        healthScore: mlHealth,
        riskLevel: mlRisk,
        remainingDays: mlRemaining,
        maintenanceNeeded: mlMaintenance ?? (prediction === 1),
        riskScore: mlRiskScore ?? probability,
        recommendations: mlRecommendations || [],
      }
    : deriveInsights(probability, prediction);

  return {
    prediction,
    probability,
    avg_prediction: mlData?.avg_prediction ?? probability,
    binary_prediction: mlData?.binary_prediction ?? prediction,
    healthScore: insights.healthScore ?? mlHealth ?? 50,
    health_score: insights.healthScore ?? mlHealth ?? 50,
    riskLevel: insights.riskLevel ?? mlRisk ?? "Unknown",
    risk_level: insights.riskLevel ?? mlRisk ?? "Unknown",
    riskScore: insights.riskScore ?? probability,
    risk_score: insights.riskScore ?? probability,
    remainingDays: insights.remainingDays ?? mlRemaining ?? null,
    remaining_days: insights.remainingDays ?? mlRemaining ?? null,
    maintenanceNeeded: insights.maintenanceNeeded,
    maintenance_needed: insights.maintenanceNeeded,
    recommendations: insights.recommendations ?? mlRecommendations ?? [],
    modelUsed: mlData?.model_used || mlData?.modelUsed || "LSTM",
    model_used: mlData?.model_used || mlData?.modelUsed || "LSTM",
    allPredictions: mlData?.all_predictions || mlData?.allPredictions || { [modelKey]: prediction },
    all_predictions: mlData?.all_predictions || mlData?.allPredictions || { [modelKey]: prediction },
    allProbabilities: mlData?.all_probabilities || mlData?.allProbabilities || { [modelKey]: probability },
    all_probabilities: mlData?.all_probabilities || mlData?.allProbabilities || { [modelKey]: probability },
    raw: mlData,
    ...overrides,
  };
}

async function callMl(payload) {
  const resp = await axios.post(`${ML_BASE}/predict`, payload, { timeout: ML_TIMEOUT_MS });
  return resp.data || {};
}

async function predictFromSensors(sensors, { allowHeuristicFallback = false } = {}) {
  const row = buildRowFromSensors(sensors);
  const seqLen = Number(process.env.FORCE_SEQ_LEN) || 32;
  const sensors_sequence = Array.from({ length: seqLen }, () => row);
  const payload = { sensors, sensors_sequence };

  try {
    const mlData = await callMl(payload);
    return formatMlResponse(mlData, { sensors });
  } catch (err) {
    console.error("ML request failed:", err?.response?.data || err?.message || err);
    if (allowHeuristicFallback) {
      console.warn("Falling back to heuristic scoring for this row.");
      return buildHeuristicResponse(sensors);
    }
    throw err;
  }
}

function hasAtLeastOneValue(obj = {}) {
  return Object.values(obj).some((val) => val !== undefined);
}

function heuristicProbabilityFromSensors(sensors = {}) {
  const cleaned = {};
  SENSOR_KEYS.forEach((key) => {
    const val = sanitizeNumber(sensors[key]);
    cleaned[key] = Number.isFinite(val) ? val : undefined;
  });

  const excessRatio = (value, start, span) => {
    if (!Number.isFinite(value) || value <= start) return 0;
    return clamp((value - start) / span, 0, 1.5);
  };

  let score = 0;
  Object.entries(HEURISTIC_RULES).forEach(([key, rule]) => {
    score += rule.weight * excessRatio(cleaned[key], rule.start, rule.span);
  });

  const { temperature: temp, vibration: vib, pressure, rpm } = cleaned;

  if (Number.isFinite(temp) && Number.isFinite(vib)) {
    if (temp >= 80 && vib >= 3) score += 0.12;
    else if (temp >= 75 && vib >= 2.5) score += 0.1;
    else if (temp >= 70 && vib >= 2) score += 0.08;
    else if (vib >= 1.5 && temp >= 65) score += 0.05;
  }

  if (Number.isFinite(pressure)) {
    if (pressure >= 58) score += 0.1;
    else if (pressure >= 52) score += 0.06;
    else if (pressure >= 47) score += 0.03;
  }

  if (Number.isFinite(vib) && Number.isFinite(pressure) && vib >= 1.5 && pressure >= 47) {
    score += 0.04;
  }

  if (Number.isFinite(rpm)) {
    if (rpm >= 1650) score += 0.06;
    else if (rpm >= 1550) score += 0.03;
    else if (rpm >= 1500) score += 0.02;
  }

  score = clamp(score, 0, 1.35);
  if (score === 0) return 0.02;
  let probability = Math.pow(score, 0.7);
  if (score < 0.08) probability *= 0.6;
  return clamp(probability, 0, 0.99);
}

function buildHeuristicResponse(sensors) {
  const probability = heuristicProbabilityFromSensors(sensors);
  const prediction = probability >= 0.5 ? 1 : 0;
  const insights = deriveInsights(probability, prediction);
  return {
    prediction,
    probability,
    avg_prediction: probability,
    binary_prediction: prediction,
    ...insights,
    modelUsed: "Heuristic",
    model_used: "Heuristic",
    allPredictions: { heuristic: prediction },
    all_predictions: { heuristic: prediction },
    allProbabilities: { heuristic: probability },
    all_probabilities: { heuristic: probability },
    sensors,
    raw: { mode: "heuristic" },
  };
}

async function processRowsConcurrently(
  rows,
  {
    concurrency = FILE_PREDICTION_CONCURRENCY,
    predictor = predictFromSensors,
    useHeuristic = false,
  } = {}
) {
  const usableConcurrency = Math.max(1, Math.min(concurrency, rows.length));
  const perRowResults = new Array(rows.length);
  const perRowErrors = [];
  let cursor = 0;

  async function worker() {
    while (true) {
      const idx = cursor++;
      if (idx >= rows.length) break;

      try {
        const sensors = normalizeSensorsInput(rows[idx]);
        if (!hasAtLeastOneValue(sensors)) {
          perRowErrors.push({ index: idx, reason: "No recognizable sensor values." });
          continue;
        }
        const formatted = useHeuristic ? buildHeuristicResponse(sensors) : await predictor(sensors);
        perRowResults[idx] = formatted;
      } catch (err) {
        perRowErrors.push({
          index: idx,
          reason: err?.response?.data?.error || err?.message || "ML request failed",
        });
      }
    }
  }

  const workers = Array.from({ length: usableConcurrency }, () => worker());
  await Promise.all(workers);

  const completed = perRowResults.filter(Boolean);
  return { completed, errors: perRowErrors };
}

// POST /api/predictions/predict
router.post("/predict", async (req, res) => {
  try {
    const body = req.body || {};

    const sensors = normalizeSensorsInput(body);
    if (!hasAtLeastOneValue(sensors)) {
      return res.status(400).json({
        error: "Provide at least one sensor reading (temperature, vibration, pressure, rpm).",
      });
    }

    const formatted = await predictFromSensors(sensors);
    return res.json(formatted);
  } catch (err) {
    const downstream = err?.response?.data;
    const status = err?.response?.status || 500;
    console.error("/api/predictions/predict error:", {
      message: err?.message,
      status,
      downstream,
      stack: err?.stack,
    });
    if (downstream) {
      const payload =
        typeof downstream === "object" ? downstream : { error: downstream };
      return res.status(status).json(payload);
    }
    return res.status(status).json({
      error: err?.message || "Internal server error",
      details: err?.stack,
    });
  }
});

// POST /api/predictions/predict_file
router.post("/predict_file", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    const csvText = req.file.buffer.toString("utf-8");
    const parsed = Papa.parse(csvText, {
      header: true,
      skipEmptyLines: true,
      transformHeader: (header) => header?.trim(),
    });

    const parsedRows = (parsed.data || []).filter(
      (row) => row && Object.values(row).some((value) => value !== null && value !== "")
    );

    if (!parsedRows.length) {
      return res.status(400).json({ error: "CSV does not contain any data rows." });
    }

    const rows = parsedRows.slice(0, MAX_FILE_ROWS);
    const trimmed = parsedRows.length - rows.length;
    const forceHeuristic = FILE_FAST_MODE === "always";
    const shouldUseHeuristic =
      forceHeuristic ||
      (FILE_FAST_MODE === "auto" && rows.length >= FILE_FAST_THRESHOLD) ||
      FILE_FAST_MODE === "heuristic";

    const { completed: perRowResults, errors: rowErrors } = await processRowsConcurrently(rows, {
      concurrency: shouldUseHeuristic ? FILE_FAST_CONCURRENCY : FILE_PREDICTION_CONCURRENCY,
      predictor: (sensors) => predictFromSensors(sensors, { allowHeuristicFallback: true }),
      useHeuristic: shouldUseHeuristic,
    });

    if (!perRowResults.length) {
      return res.status(400).json({
        error: "No recognizable sensor columns found. Expected headers like temperature, vibration, pressure, rpm.",
      });
    }

    const predictions = perRowResults.map((r) => r.prediction);
    const probabilities = perRowResults.map((r) => r.probability);
    const failure_predictions = predictions.filter((p) => p === 1).length;
    const validProbs = probabilities.filter((p) => typeof p === "number" && !Number.isNaN(p));
    const avg_probability =
      validProbs.length > 0
        ? validProbs.reduce((sum, val) => sum + val, 0) / validProbs.length
        : null;

    const insights = deriveInsights(avg_probability, failure_predictions > perRowResults.length / 2 ? 1 : 0);

    const summary = {
      total_rows: perRowResults.length,
      failure_predictions,
      average_failure_probability: avg_probability,
      risk_level: insights.riskLevel,
      health_score: insights.healthScore,
      remaining_days: insights.remainingDays,
    };

    return res.json({
      summary,
      predictions,
      probabilities,
      rows: perRowResults,
      meta: {
        total_rows_in_file: parsedRows.length,
        processed_rows: perRowResults.length,
        trimmed_rows: trimmed,
        concurrency: shouldUseHeuristic ? FILE_FAST_CONCURRENCY : FILE_PREDICTION_CONCURRENCY,
        mode: shouldUseHeuristic ? "heuristic" : "ml",
        row_errors: rowErrors,
      },
    });
  } catch (err) {
    const downstream = err?.response?.data;
    const status = err?.response?.status || 500;
    console.error("/api/predictions/predict_file error:", {
      message: err?.message,
      status,
      downstream,
      stack: err?.stack,
    });
    if (downstream) {
      const payload =
        typeof downstream === "object" ? downstream : { error: downstream };
      return res.status(status).json(payload);
    }
    return res.status(status).json({
      error: err?.message || "Internal server error",
      details: err?.stack,
    });
  }
});

export default router;
