// frontend/src/services/API.js
import axios from "axios";
import { config } from "../../env.config.js"; // adjust path if needed

// ---------------------------------------------------------
// BACKEND & ML API base URL
// backend: Node.js (port 5000)
// ml-model: Flask/ML server (proxied by backend)
// ---------------------------------------------------------

const BACKEND_URL =
  config?.BACKEND_URL ??
  import.meta.env.VITE_BACKEND_URL ??
  "http://localhost:5000";

export const API = axios.create({
  baseURL: `${BACKEND_URL}/api`,
  timeout: 30000,
});

// ---------------------------------------------------------
//  PREDICTIONS (LIVE FROM ML SERVER THROUGH BACKEND)
// ---------------------------------------------------------

export const predictRow = async (sensors) => {
  // sensors = { temperature: 75, pressure: 50, vibration: 3.2, rpm: 1500 }
  const { data } = await API.post("/predictions/predict", { sensors });
  return data; // { prediction, probability, modelVersion }
};

export const predictFile = async (file) => {
  const form = new FormData();
  form.append("file", file);

  const { data } = await API.post("/predictions/predict_file", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return data; // { count, predictions[], probabilities[] }
};

// ---------------------------------------------------------
//  SENSOR + LOG ENDPOINTS (IF BACKEND STORES DATA)
// ---------------------------------------------------------

export const fetchSensors = () =>
  API.get("/sensors").then((res) => res.data);

export const fetchLogs = () =>
  API.get("/logs").then((res) => res.data);

export const postSensor = (payload) =>
  API.post("/sensors", payload).then((res) => res.data);

// ---------------------------------------------------------
//  NOTE:
//  ⚠️ We removed static exports:
//     - sensorData
//     - equipmentData
//     - healthDistribution
//  Any component that used these MUST fetch real backend data
// ---------------------------------------------------------
