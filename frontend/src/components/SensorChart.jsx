// // import React from 'react';
// // import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from 'recharts';
// // import { sensorData } from '../services/API';

// // const SensorChart = () => (
// //   <div className="bg-white p-6 rounded-lg border border-gray-200">
// //     <h3 className="text-lg font-semibold mb-4">Live Sensor Trends</h3>
// //     <ResponsiveContainer width="100%" height={250}>
// //       <LineChart data={sensorData}>
// //         <CartesianGrid strokeDasharray="3 3" />
// //         <XAxis dataKey="time" />
// //         <YAxis />
// //         <Line type="monotone" dataKey="pressure" stroke="#3B82F6" name="Pressure" />
// //         <Line type="monotone" dataKey="temperature" stroke="#EF4444" name="Temperature" />
// //         <Line type="monotone" dataKey="vibration" stroke="#10B981" name="Vibration" />
// //       </LineChart>
// //     </ResponsiveContainer>
// //     <div className="flex justify-center space-x-6 mt-2 text-sm">
// //       <span className="flex items-center"><div className="w-3 h-3 bg-blue-500 rounded mr-2"></div>Pressure</span>
// //       <span className="flex items-center"><div className="w-3 h-3 bg-red-500 rounded mr-2"></div>Temperature</span>
// //       <span className="flex items-center"><div className="w-3 h-3 bg-green-500 rounded mr-2"></div>Vibration</span>
// //     </div>
// //   </div>
// // );

// // export default SensorChart;


// // frontend/src/components/SensorChart.jsx
// import React, { useEffect, useState } from "react";
// import { predictRow } from "../services/API"; // adjust relative path

// // sensors prop should be an object { temperature, pressure, vibration, rpm, ... }
// const SensorChart = ({ sensors }) => {
//   const [prediction, setPrediction] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState(null);

//   useEffect(() => {
//     let mounted = true;
//     const run = async () => {
//       if (!sensors || Object.keys(sensors).length === 0) {
//         setPrediction(null);
//         return;
//       }
//       setLoading(true);
//       setError(null);
//       try {
//         const res = await predictRow(sensors);
//         if (!mounted) return;
//         setPrediction(res);
//       } catch (err) {
//         if (!mounted) return;
//         setError(err?.response?.data?.error || err.message || "Prediction failed");
//       } finally {
//         if (mounted) setLoading(false);
//       }
//     };
//     run();
//     return () => { mounted = false; };
//   }, [sensors]);

//   if (!sensors) return <div className="p-4 text-sm text-gray-500">No sensor input provided</div>;

//   return (
//     <div className="bg-white rounded-lg p-4 border">
//       <h4 className="font-semibold mb-2">Live Prediction</h4>

//       <div className="text-xs text-gray-600 mb-2">Input:</div>
//       <pre className="text-sm bg-gray-50 p-2 rounded">{JSON.stringify(sensors, null, 2)}</pre>

//       {loading && <div className="text-sm text-gray-500 mt-2">Running prediction...</div>}
//       {error && <div className="text-sm text-red-600 mt-2">Error: {error}</div>}

//       {prediction && (
//         <div className="mt-3 space-y-2">
//           <div><strong>Model:</strong> {prediction.modelVersion ?? "unknown"}</div>
//           <div><strong>Prediction:</strong> {prediction.prediction === 1 ? "Maintenance required" : "Healthy"}</div>
//           {typeof prediction.probability === "number" && (
//             <div><strong>Probability of failure:</strong> {(prediction.probability * 100).toFixed(1)}%</div>
//           )}
//         </div>
//       )}
//     </div>
//   );
// };

// export default SensorChart;
// frontend/src/components/SensorChart.jsx
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
