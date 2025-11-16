  // // // import axios from "axios";
  // // // import { config } from "../../env.config.js";

  // // // // Pick base URLs from environment variables (set in .env for Vite)
  // // // const BACKEND_URL = config.BACKEND_URL;
  // // // const ML_API_URL  = config.ML_API_URL;

  // // // // Create axios instance for backend (sensors, logs, etc.)
  // // // export const API = axios.create({
  // // // 	baseURL: `${BACKEND_URL}/api`,
  // // // 	timeout: 10000,
  // // // });

  // // // // Separate instance for ML API (Flask)
  // // // export const ML_API = axios.create({
  // // // 	baseURL: ML_API_URL,
  // // // 	timeout: 15000,
  // // // });

  // // // // ----------------- API calls -----------------
  // // // export const fetchSensors = () => API.get("/sensors").then(r => r.data);
  // // // export const fetchLogs = () => API.get("/logs").then(r => r.data);
  // // // export const postSensor = (payload) => API.post("/sensors", payload).then(r => r.data);

  // // // // Predictions
  // // // export const predictRow = async (sensors) => {
  // // // 	// sensors can be an object keyed by feature names, or an ordered array
  // // // 	const { data } = await ML_API.post("/predict", { sensors });
  // // // 	return data;
  // // // };

  // // // export const predictFile = async (file) => {
  // // // 	const form = new FormData();
  // // // 	form.append("file", file);
  // // // 	const { data } = await ML_API.post("/predict_file", form, {
  // // // 		headers: { "Content-Type": "multipart/form-data" },
  // // // 	});
  // // // 	return data;
  // // // };


  // // // // Sample data - move this to a separate data service
  // // // export const sensorData = [
  // // //   { time: '00:00', pressure: 45, temperature: 75, vibration: 2.1 },
  // // //   { time: '04:00', pressure: 48, temperature: 78, vibration: 2.3 },
  // // //   { time: '08:00', pressure: 52, temperature: 82, vibration: 2.8 },
  // // //   { time: '12:00', pressure: 49, temperature: 79, vibration: 2.4 },
  // // //   { time: '16:00', pressure: 46, temperature: 76, vibration: 2.2 },
  // // //   { time: '20:00', pressure: 44, temperature: 74, vibration: 2.0 },
  // // // ];

  // // // export const equipmentData = [
  // // //   { id: 'EQ001', name: 'Motor A1', status: 'Good', health: 85, daysLeft: 45, risk: 'Low' },
  // // //   { id: 'EQ002', name: 'Pump B2', status: 'Warning', health: 60, daysLeft: 15, risk: 'Medium' },
  // // //   { id: 'EQ003', name: 'Compressor C3', status: 'Critical', health: 25, daysLeft: 3, risk: 'High' },
  // // //   { id: 'EQ004', name: 'Generator D4', status: 'Good', health: 90, daysLeft: 60, risk: 'Low' },
  // // // ];

  // // // export const healthDistribution = [
  // // //   { name: 'Good', value: 65, color: '#10B981' },
  // // //   { name: 'Warning', value: 25, color: '#F59E0B' },
  // // //   { name: 'Critical', value: 10, color: '#EF4444' },
  // // // ];

  // // // frontend/src/services/api.js

  // // // in frontend/src/services/api.js
  // // export const predictRow = (sensors) => handleResponse(API.post("/predictions/predict", { sensors }));
  // // export const predictFile = (file) => {
  // //   const form = new FormData();
  // //   form.append("file", file);
  // //   return handleResponse(API.post("/predictions/predict_file", form, {
  // //     headers: { "Content-Type": "multipart/form-data" },
  // //   }));
  // // };
  // // frontend/src/services/API.js
  // import axios from "axios";
  // import { config } from "../../env.config.js"; // path depending on your structure

  // const BACKEND_URL = config?.BACKEND_URL ?? import.meta.env.VITE_BACKEND_URL ?? "http://localhost:5000";

  // export const API = axios.create({
  //   baseURL: `${BACKEND_URL}/api`,
  //   timeout: 15000,
  // });

  // // Calls backend which forwards to ML
  // export const predictRow = async (sensors) => {
  //   // sensors is expected to be an object: { temperature: 75, vibration: 2.5, ... }
  //   const { data } = await API.post("/predictions/predict", { sensors });
  //   return data;
  // };

  // export const predictFile = async (file) => {
  //   const form = new FormData();
  //   form.append("file", file);
  //   const { data } = await API.post("/predictions/predict_file", form, {
  //     headers: { "Content-Type": "multipart/form-data" },
  //   });
  //   return data;
  // };

  // // Optional backend helpers (fetch logs, sensors, etc.)
  // export const fetchSensors = () => API.get("/sensors").then(r => r.data);
  // export const fetchLogs = () => API.get("/logs").then(r => r.data);
  // export const postSensor = (payload) => API.post("/sensors", payload).then(r => r.data);

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
  timeout: 15000,
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
