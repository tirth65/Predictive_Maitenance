import React, { useEffect, useState } from "react";
import { predictRow } from "../services/API"; // adjust path if needed

// sensors prop should be an object { temperature, pressure, vibration, rpm, ... }
const SensorChart = ({ sensors }) => {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let mounted = true;
    const run = async () => {
      if (!sensors || Object.keys(sensors).length === 0) {
        setPrediction(null);
        return;
      }
      setLoading(true);
      setError(null);
      try {
        const res = await predictRow(sensors);
        if (!mounted) return;
        setPrediction(res);
      } catch (err) {
        if (!mounted) return;
        setError(err?.response?.data?.error || err.message || "Prediction failed");
      } finally {
        if (mounted) setLoading(false);
      }
    };
    run();
    return () => { mounted = false; };
  }, [sensors]);

  if (!sensors) return <div className="p-4 text-sm text-gray-500">No sensor input provided</div>;

  return (
    <div className="bg-white rounded-lg p-4 border">
      <h4 className="font-semibold mb-2">Live Prediction</h4>

      <div className="text-xs text-gray-600 mb-2">Input:</div>
      <pre className="text-sm bg-gray-50 p-2 rounded">{JSON.stringify(sensors, null, 2)}</pre>

      {loading && <div className="text-sm text-gray-500 mt-2">Running prediction...</div>}
      {error && <div className="text-sm text-red-600 mt-2">Error: {error}</div>}

      {prediction && (
        <div className="mt-3 space-y-2">
          <div><strong>Model:</strong> {prediction.modelVersion ?? "unknown"}</div>
          <div><strong>Prediction:</strong> {prediction.prediction === 1 ? "Maintenance required" : "Healthy"}</div>
          {typeof prediction.probability === "number" && (
            <div><strong>Probability of failure:</strong> {(prediction.probability * 100).toFixed(1)}%</div>
          )}
        </div>
      )}
    </div>
  );
};

export default SensorChart;