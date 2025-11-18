// import React from 'react';
// import { equipmentData } from '../services/API';

// const Equipment = () => (
//   <div className="p-6 bg-gray-50 min-h-screen">
//     <div className="mb-6">
//       <h2 className="text-2xl font-bold text-gray-900 mb-2">Equipment</h2>
//       <p className="text-gray-600">Manage and monitor all industrial equipment</p>
//     </div>

//     <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
//       {equipmentData.map((equipment) => (
//         <div key={equipment.id} className="bg-white rounded-lg border border-gray-200 p-6">
//           <div className="flex items-center justify-between mb-4">
//             <h3 className="text-lg font-semibold text-gray-900">{equipment.name}</h3>
//             <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
//               equipment.status === 'Good' ? 'bg-green-100 text-green-800' :
//               equipment.status === 'Warning' ? 'bg-yellow-100 text-yellow-800' :
//               'bg-red-100 text-red-800'
//             }`}>
//               {equipment.status}
//             </span>
//           </div>
//           <div className="space-y-3">
//             <div>
//               <p className="text-sm text-gray-600">Equipment ID</p>
//               <p className="font-medium">{equipment.id}</p>
//             </div>
//             <div>
//               <p className="text-sm text-gray-600">Health Score</p>
//               <div className="flex items-center">
//                 <span className="mr-2 font-medium">{equipment.health}%</span>
//                 <div className="flex-1 bg-gray-200 rounded-full h-2">
//                   <div 
//                     className={`h-2 rounded-full ${equipment.health >= 80 ? 'bg-green-500' : equipment.health >= 50 ? 'bg-yellow-500' : 'bg-red-500'}`}
//                     style={{ width: `${equipment.health}%` }}
//                   />
//                 </div>
//               </div>
//             </div>
//             <div>
//               <p className="text-sm text-gray-600">Estimated Remaining Life</p>
//               <p className="font-medium">{equipment.daysLeft} days</p>
//             </div>
//             <div>
//               <p className="text-sm text-gray-600">Risk Level</p>
//               <p className={`font-medium ${
//                 equipment.risk === 'Low' ? 'text-green-600' :
//                 equipment.risk === 'Medium' ? 'text-yellow-600' :
//                 'text-red-600'
//               }`}>{equipment.risk}</p>
//             </div>
//           </div>
//           <button className="mt-4 w-full bg-blue-500 text-white py-2 px-4 rounded-lg hover:bg-blue-600 transition-colors">
//             View Details
//           </button>
//         </div>
//       ))}
//     </div>
//   </div>
// );

// export default Equipment;

// frontend/src/pages/Equipment.jsx
import React, { useEffect, useState } from "react";
import { fetchEquipment } from "../services/API";

const Equipment = () => {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      setLoading(true);
      setErr("");
      try {
        const data = await fetchEquipment();
        if (!mounted) return;
        setItems(Array.isArray(data) ? data : []);
      } catch (e) {
        if (!mounted) return;
        setErr(
          e?.response?.data?.error || e.message || "Failed to load equipment"
        );
      } finally {
        if (mounted) setLoading(false);
      }
    };
    load();
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <div className="page-shell">
      <section className="glass-panel">
        <p className="eyebrow">Asset Library</p>
        <div className="flex flex-wrap items-end gap-3 justify-between">
          <div>
            <h2 className="page-heading">Field Equipment</h2>
            <p className="page-description">
              Monitor all live endpoints, health signals, and risk stratification
              for your predictive maintenance fleet.
            </p>
          </div>
          <span className="chip chip-neutral">
            {items.length} tracked assets
          </span>
        </div>
      </section>

      {loading && (
        <section className="glass-panel">
          <p className="text-sm text-slate-300 uppercase tracking-widest">
            Syncing equipment registry…
          </p>
        </section>
      )}

      {err && (
        <section className="glass-panel">
          <p className="text-red-300">{err}</p>
        </section>
      )}

      <div className="equipment-grid">
        {items.map((equipment) => {
          const id =
            equipment.machineId ||
            equipment.id ||
            equipment._id ||
            equipment.assetId ||
            "unknown";
          const lastPrediction = equipment.lastPrediction || {};
          const lastSensors = equipment.lastSensors || {};
          const probability =
            typeof equipment.probability === "number"
              ? equipment.probability
              : typeof lastPrediction.probability === "number"
                ? lastPrediction.probability
                : 0;
          const status =
            equipment.status ||
            lastPrediction.riskLevel ||
            (probability > 0.66 ? "Critical" : probability > 0.33 ? "Warning" : "Good");
          const health =
            equipment.health ??
            lastPrediction.healthScore ??
            Math.round((1 - probability) * 100);
          const daysLeft =
            equipment.daysLeft ??
            lastPrediction.remainingDays ??
            equipment.remainingDays ??
            "-";
          const risk =
            equipment.risk ??
            lastPrediction.riskLevel ??
            (status === "Good" ? "Low" : status === "Warning" ? "Medium" : "High");

          const chipTone =
            status === "Good"
              ? "chip-live"
              : status === "Warning"
                ? "chip-warning"
                : "";

          const name = equipment.name || lastPrediction.equipmentName || equipment.equipment || id;

          const barColor =
            health >= 80
              ? "linear-gradient(90deg,#34d399,#10b981)"
              : health >= 50
                ? "linear-gradient(90deg,#facc15,#f97316)"
                : "linear-gradient(90deg,#f43f5e,#fb7185)";

          return (
            <article key={id} className="holo-card equipment-card">
              <div className="equipment-meta">
                <span>Unit {id}</span>
                <span>{risk} risk</span>
              </div>
              <h3>{name}</h3>
              <div className="flex flex-wrap gap-2 mb-4">
                <span className={`chip ${chipTone}`}>{status}</span>
                {typeof daysLeft === "number" && (
                  <span className="chip chip-neutral">{daysLeft} days left</span>
                )}
              </div>

              <div className="space-y-4 text-sm text-slate-300">
                <div>
                  <p className="stat-label">Health</p>
                  <p className="text-2xl font-semibold">{health}%</p>
                  <div className="equipment-health mt-2">
                    <span
                      style={{ width: `${health}%`, background: barColor }}
                    ></span>
                  </div>
                </div>
                <div className="flex justify-between text-xs uppercase tracking-widest text-slate-400">
                  <span>Lifecycle</span>
                  <span>{typeof daysLeft === "number" ? `${daysLeft} days` : daysLeft}</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  <span className="chip chip-neutral">
                    RPM {lastSensors.rpm ?? equipment.rpm ?? "—"}
                  </span>
                  <span className="chip chip-neutral">
                    Temp {lastSensors.temperature ?? equipment.temperature ?? "—"}
                  </span>
                </div>
              </div>

              <button type="button" className="btn-ghost mt-5">
                View System Sheet
              </button>
            </article>
          );
        })}
      </div>
    </div>
  );
};

export default Equipment;
