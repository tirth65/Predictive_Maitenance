// // import React from 'react';
// // import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
// // import { Activity, AlertTriangle, CheckCircle, Cpu } from 'lucide-react';
// // import StatusCard from '../components/StateCard';
// // import SensorChart from '../components/SensorChart';
// // import SensorTable from '../components/SensorTable';
// // import { healthDistribution } from '../services/API';

// // const Dashboard = () => (
// //   <div className="p-6 bg-gray-50 min-h-screen">
// //     <div className="mb-6">
// //       <h2 className="text-2xl font-bold text-gray-900 mb-2">Dashboard</h2>
// //       <p className="text-gray-600">Overview of equipment health and maintenance status</p>
// //     </div>

// //     {/* Status Cards */}
// //     <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
// //       <StatusCard title="Total Equipment" value="24" icon={Cpu} color="blue" />
// //       <StatusCard title="Healthy" value="18" icon={CheckCircle} color="green" />
// //       <StatusCard title="Warnings" value="4" icon={AlertTriangle} color="yellow" />
// //       <StatusCard title="Critical" value="2" icon={AlertTriangle} color="red" />
// //     </div>

// //     {/* Charts */}
// //     <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
// //       <div className="bg-white p-6 rounded-lg border border-gray-200">
// //         <h3 className="text-lg font-semibold mb-4">Equipment Health Distribution</h3>
// //         <ResponsiveContainer width="100%" height={250}>
// //           <PieChart>
// //             <Pie
// //               data={healthDistribution}
// //               cx="50%"
// //               cy="50%"
// //               outerRadius={80}
// //               dataKey="value"
// //             >
// //               {healthDistribution.map((entry, index) => (
// //                 <Cell key={`cell-${index}`} fill={entry.color} />
// //               ))}
// //             </Pie>
// //           </PieChart>
// //         </ResponsiveContainer>
// //       </div>

// //       <SensorChart />
// //     </div>

// //     {/* Equipment Overview */}
// //     <SensorTable />
// //   </div>
// // );

// // export default Dashboard;

// // frontend/src/pages/Dashboard.jsx
// import React, { useEffect, useState } from "react";
// import { fetchSensors } from "../services/API"; // adjust path if needed

// const computeDistribution = (items) => {
//   // items: array of equipment objects or prediction summaries.
//   // We try to infer 'status' or map from probability/health fields.
//   const counts = { Good: 0, Warning: 0, Critical: 0 };

//   items.forEach((it) => {
//     // try multiple possible shapes from backend
//     const status =
//       it.status ||
//       (typeof it.health === "number" ? (it.health >= 80 ? "Good" : it.health >= 50 ? "Warning" : "Critical") : null) ||
//       (typeof it.probability === "number" ? (it.probability > 0.66 ? "Critical" : it.probability > 0.33 ? "Warning" : "Good") : "Good");

//     if (status === "Good") counts.Good++;
//     else if (status === "Warning") counts.Warning++;
//     else counts.Critical++;
//   });

//   const total = Math.max(1, items.length);
//   return [
//     { name: "Good", value: Math.round((counts.Good / total) * 100) },
//     { name: "Warning", value: Math.round((counts.Warning / total) * 100) },
//     { name: "Critical", value: Math.round((counts.Critical / total) * 100) },
//   ];
// };

// const Dashboard = () => {
//   const [items, setItems] = useState([]);
//   const [dist, setDist] = useState([]);
//   const [loading, setLoading] = useState(true);
//   const [err, setErr] = useState("");

//   useEffect(() => {
//     let mounted = true;
//     const load = async () => {
//       setLoading(true);
//       setErr("");
//       try {
//         const data = await fetchSensors(); // should return array of equipment/prediction summaries
//         if (!mounted) return;
//         setItems(Array.isArray(data) ? data : []);
//         setDist(computeDistribution(Array.isArray(data) ? data : []));
//       } catch (e) {
//         if (!mounted) return;
//         setErr(e?.response?.data?.error || e.message || "Failed to load data");
//       } finally {
//         if (mounted) setLoading(false);
//       }
//     };
//     load();
//     return () => { mounted = false; };
//   }, []);

//   return (
//     <div className="p-6 bg-gray-50 min-h-screen">
//       <div className="mb-6">
//         <h2 className="text-2xl font-bold text-gray-900 mb-2">Dashboard</h2>
//         <p className="text-gray-600">Live overview from model predictions</p>
//       </div>

//       {loading && <div className="text-gray-500">Loading dashboard...</div>}
//       {err && <div className="text-red-600">Error: {err}</div>}

//       {!loading && !err && (
//         <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
//           {/* quick cards */}
//           <div className="bg-white rounded-lg p-6 border border-gray-200">
//             <p className="text-sm text-gray-600">Total Equipment</p>
//             <p className="text-2xl font-semibold">{items.length}</p>
//           </div>

//           <div className="bg-white rounded-lg p-6 border border-gray-200">
//             <p className="text-sm text-gray-600">At-risk (Critical)</p>
//             <p className="text-2xl font-semibold">
//               {items.filter(i => ((i.status === "Critical") || (typeof i.probability === "number" && i.probability > 0.66))).length}
//             </p>
//           </div>

//           <div className="bg-white rounded-lg p-6 border border-gray-200">
//             <p className="text-sm text-gray-600">Warnings</p>
//             <p className="text-2xl font-semibold">
//               {items.filter(i => (i.status === "Warning" || (typeof i.probability === "number" && i.probability > 0.33 && i.probability <= 0.66))).length}
//             </p>
//           </div>

//           {/* distribution display */}
//           <div className="md:col-span-3 bg-white rounded-lg p-6 border border-gray-200">
//             <h3 className="text-lg font-semibold mb-3">Health Distribution</h3>
//             <div className="flex gap-4">
//               {dist.map(d => (
//                 <div key={d.name} className="flex-1 text-center p-4">
//                   <div className="text-sm text-gray-600">{d.name}</div>
//                   <div className="text-2xl font-bold">{d.value}%</div>
//                 </div>
//               ))}
//             </div>
//           </div>
//         </div>
//       )}
//     </div>
//   );
// };

// export default Dashboard;

// frontend/src/pages/Dashboard.jsx
import React, { useEffect, useState } from "react";
import { fetchSensors } from "../services/API"; // adjust path if needed

const computeDistribution = (items) => {
  const counts = { Good: 0, Warning: 0, Critical: 0 };
  items.forEach((it) => {
    const status =
      it.status ||
      (typeof it.health === "number" ? (it.health >= 80 ? "Good" : it.health >= 50 ? "Warning" : "Critical") : null) ||
      (typeof it.probability === "number" ? (it.probability > 0.66 ? "Critical" : it.probability > 0.33 ? "Warning" : "Good") : "Good");

    if (status === "Good") counts.Good++;
    else if (status === "Warning") counts.Warning++;
    else counts.Critical++;
  });

  const total = Math.max(1, items.length);
  return [
    { name: "Good", value: Math.round((counts.Good / total) * 100) },
    { name: "Warning", value: Math.round((counts.Warning / total) * 100) },
    { name: "Critical", value: Math.round((counts.Critical / total) * 100) },
  ];
};

const Dashboard = () => {
  const [items, setItems] = useState([]);
  const [dist, setDist] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      setLoading(true);
      setErr("");
      try {
        const data = await fetchSensors();
        if (!mounted) return;
        setItems(Array.isArray(data) ? data : []);
        setDist(computeDistribution(Array.isArray(data) ? data : []));
      } catch (e) {
        if (!mounted) return;
        setErr(e?.response?.data?.error || e.message || "Failed to load data");
      } finally {
        if (mounted) setLoading(false);
      }
    };
    load();
    return () => { mounted = false; };
  }, []);

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Dashboard</h2>
        <p className="text-gray-600">Live overview from model predictions</p>
      </div>

      {loading && <div className="text-gray-500">Loading dashboard...</div>}
      {err && <div className="text-red-600">Error: {err}</div>}

      {!loading && !err && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white rounded-lg p-6 border border-gray-200">
            <p className="text-sm text-gray-600">Total Equipment</p>
            <p className="text-2xl font-semibold">{items.length}</p>
          </div>

          <div className="bg-white rounded-lg p-6 border border-gray-200">
            <p className="text-sm text-gray-600">At-risk (Critical)</p>
            <p className="text-2xl font-semibold">
              {items.filter(i => ((i.status === "Critical") || (typeof i.probability === "number" && i.probability > 0.66))).length}
            </p>
          </div>

          <div className="bg-white rounded-lg p-6 border border-gray-200">
            <p className="text-sm text-gray-600">Warnings</p>
            <p className="text-2xl font-semibold">
              {items.filter(i => (i.status === "Warning" || (typeof i.probability === "number" && i.probability > 0.33 && i.probability <= 0.66))).length}
            </p>
          </div>

          <div className="md:col-span-3 bg-white rounded-lg p-6 border border-gray-200">
            <h3 className="text-lg font-semibold mb-3">Health Distribution</h3>
            <div className="flex gap-4">
              {dist.map(d => (
                <div key={d.name} className="flex-1 text-center p-4">
                  <div className="text-sm text-gray-600">{d.name}</div>
                  <div className="text-2xl font-bold">{d.value}%</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
