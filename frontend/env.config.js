// Environment configuration for frontend
export const config = {
  BACKEND_URL: import.meta.env.VITE_BACKEND_URL || "http://localhost:5000",
  ML_API_URL: import.meta.env.VITE_ML_API_URL || "http://localhost:8000"
};
