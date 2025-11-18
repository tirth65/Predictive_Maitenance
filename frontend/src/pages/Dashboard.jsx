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
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { fetchEquipment } from "../services/API";

const resolveStatus = (item) => {
  // Handle equipment model structure
  if (item.lastPrediction?.riskLevel) {
    const risk = item.lastPrediction.riskLevel;
    if (risk === "High") return "Critical";
    if (risk === "Medium") return "Warning";
    return "Good";
  }
  if (item.lastPrediction?.healthScore !== undefined) {
    const health = item.lastPrediction.healthScore;
    if (health >= 80) return "Good";
    if (health >= 50) return "Warning";
    return "Critical";
  }
  if (item.status) return item.status;
  if (typeof item.health === "number") {
    if (item.health >= 80) return "Good";
    if (item.health >= 50) return "Warning";
    return "Critical";
  }
  if (typeof item.probability === "number") {
    if (item.probability > 0.66) return "Critical";
    if (item.probability > 0.33) return "Warning";
    return "Good";
  }
  return "Good";
};

const deriveHealth = (item) => {
  // Handle equipment model structure
  if (item.lastPrediction?.healthScore !== undefined) {
    return Math.round(item.lastPrediction.healthScore);
  }
  if (typeof item.health === "number") return Math.round(item.health);
  const prob =
    typeof item.probability === "number" ? Math.min(Math.max(item.probability, 0), 1) : 0;
  return Math.round((1 - prob) * 100);
};

const computeDistribution = (items) => {
  const counts = { Good: 0, Warning: 0, Critical: 0 };
  items.forEach((it) => {
    const status = resolveStatus(it);
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
  const [refreshing, setRefreshing] = useState(false);
  const [err, setErr] = useState("");
  const [filter, setFilter] = useState("all");
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);

  const refreshData = useCallback(async (isInitial = false) => {
    if (isInitial) {
      setLoading(true);
    } else {
      setRefreshing(true);
    }
    setErr("");
    try {
      const data = await fetchEquipment();
      const safeItems = Array.isArray(data) ? data : [];
      setItems(safeItems);
      setDist(computeDistribution(safeItems));
      setLastUpdated(new Date());
    } catch (e) {
      setErr(e?.response?.data?.error || e.message || "Failed to load equipment data");
    } finally {
      if (isInitial) {
        setLoading(false);
      }
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    refreshData(true);
  }, [refreshData]);

  useEffect(() => {
    if (!autoRefresh) return undefined;
    const id = setInterval(() => {
      refreshData();
    }, 15000);
    return () => clearInterval(id);
  }, [autoRefresh, refreshData]);

  const metrics = useMemo(() => {
    if (!items.length) {
      return {
        critical: 0,
        warning: 0,
        good: 0,
        avgHealth: 0,
        stability: 100,
      };
    }

    const critical = items.filter((i) => resolveStatus(i) === "Critical").length;
    const warning = items.filter((i) => resolveStatus(i) === "Warning").length;
    const good = items.length - critical - warning;

    const avgHealth = Math.round(
      items.reduce((acc, cur) => {
        return acc + deriveHealth(cur);
      }, 0) / items.length
    );

    const stability = Math.max(
      0,
      Math.min(100, 100 - (critical * 8 + warning * 4))
    );

    return { critical, warning, good, avgHealth, stability };
  }, [items]);

  const filteredItems = useMemo(() => {
    if (filter === "all") return items;
    return items.filter((item) => resolveStatus(item).toLowerCase() === filter);
  }, [items, filter]);

  const topAssets = filteredItems.slice(0, 4);
  const lastUpdatedLabel = lastUpdated
    ? lastUpdated.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
    : "—";

  const ready = !loading && !err;
  const filterOptions = [
    { key: "all", label: "All Assets" },
    { key: "good", label: "Healthy" },
    { key: "warning", label: "Warnings" },
    { key: "critical", label: "Critical" },
  ];
  const activeFilterLabel =
    filterOptions.find((option) => option.key === filter)?.label || "All Assets";

  const getAssetMeta = (equipment) => {
    const id =
      equipment.machineId ||
      equipment.id ||
      equipment._id ||
      equipment.serial ||
      "unknown";
    const name = equipment.name || equipment.equipmentName || equipment.equipment || `Motor ${id}`;
    return { id, name };
  };

  return (
    <div className="page-shell">
      <section className="glass-panel hero-panel">
        <div>
          <p className="eyebrow">Control Deck</p>
          <h2 className="page-heading">Operational Overview</h2>
          <p className="page-description">
            Real-time synthesis of equipment health, risk posture, and model
            outputs flowing from the predictive maintenance stack.
          </p>
          <div className="flex flex-wrap gap-2 mt-4">
            <span className="chip chip-live">Live Feed</span>
            <span className="chip chip-warning">Model v2.3</span>
            <span className="chip chip-neutral">Edge Sync Enabled</span>
          </div>
        </div>
        <div className="holo-card">
          <p className="stat-label">Stability Index</p>
          <p className="stat-value">{metrics.stability}%</p>
          <p className="text-sm text-slate-400 mt-3">
            Weighted signal based on current warning and critical counts.
          </p>
          <div className="mt-4 h-2 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full"
              style={{
                width: `${metrics.stability}%`,
                background:
                  "linear-gradient(90deg, rgba(16,185,129,0.9), rgba(59,130,246,0.9))",
              }}
            ></div>
          </div>
        </div>
      </section>

      {loading && (
        <section className="glass-panel">
          <p className="text-sm text-slate-300 tracking-wide uppercase">
            Streaming live data…
          </p>
        </section>
      )}

      {err && (
        <section className="glass-panel">
          <p className="text-red-300">{err}</p>
        </section>
      )}

      {ready && (
        <>
          <section className="glass-panel control-panel">
            <div className="control-meta">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-slate-500">
                  Last Sync
                </p>
                <p className="text-base text-slate-200">{lastUpdatedLabel}</p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-slate-500">
                  Records
                </p>
                <p className="text-base text-slate-200">{items.length}</p>
              </div>
              {refreshing && (
                <p className="text-xs text-slate-400 uppercase tracking-[0.3em]">
                  Updating feed…
                </p>
              )}
            </div>

            <div className="control-actions">
              <div className="filter-group">
                {filterOptions.map((option) => (
                  <button
                    type="button"
                    key={option.key}
                    className={`filter-pill ${
                      filter === option.key ? "is-active" : ""
                    }`}
                    onClick={() => setFilter(option.key)}
                  >
                    {option.label}
                  </button>
                ))}
              </div>

              <div className="flex items-center gap-3 flex-wrap">
                <label className={`toggle ${autoRefresh ? "is-on" : ""}`}>
                  <input
                    type="checkbox"
                    checked={autoRefresh}
                    onChange={() => setAutoRefresh((prev) => !prev)}
                  />
                  <div className="toggle-switch" aria-hidden="true"></div>
                  Auto refresh
                </label>
                <button
                  type="button"
                  className="btn-ghost btn-compact"
                  onClick={() => refreshData()}
                  disabled={refreshing}
                >
                  {refreshing ? "Refreshing…" : "Manual refresh"}
                </button>
              </div>
            </div>
          </section>

          <section className="stat-grid">
            <article className="holo-card">
              <p className="stat-label">Total Assets</p>
              <p className="stat-value">{items.length}</p>
              <p className="text-xs text-slate-400 mt-2">Active telemetry nodes</p>
            </article>
            <article className="holo-card">
              <p className="stat-label">Critical Alerts</p>
              <p className="stat-value text-rose-300">{metrics.critical}</p>
              <p className="text-xs text-slate-400 mt-2">Require immediate action</p>
              {metrics.critical > 0 && (
                <div className="mt-3 space-y-1">
                  {items
                    .filter((i) => resolveStatus(i) === "Critical")
                    .slice(0, 3)
                    .map((item) => {
                      const id = item.machineId || item.id || item._id || "Unknown";
                      return (
                        <p key={id} className="text-xs text-rose-300 font-medium">
                          • Motor {id}
                        </p>
                      );
                    })}
                  {metrics.critical > 3 && (
                    <p className="text-xs text-slate-500">
                      +{metrics.critical - 3} more
                    </p>
                  )}
                </div>
              )}
            </article>
            <article className="holo-card">
              <p className="stat-label">Warning Signals</p>
              <p className="stat-value text-amber-200">{metrics.warning}</p>
              <p className="text-xs text-slate-400 mt-2">Monitor closely</p>
            </article>
            <article className="holo-card">
              <p className="stat-label">Average Health</p>
              <p className="stat-value text-emerald-200">{metrics.avgHealth}%</p>
              <p className="text-xs text-slate-400 mt-2">Across connected assets</p>
            </article>
          </section>

          <section className="glass-panel dist-panel">
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3>Health Distribution</h3>
                {dist.map((d) => (
                  <div className="dist-row" key={d.name}>
                    <span className="dist-name">{d.name}</span>
                    <div className="dist-bar">
                      <span
                        className="dist-bar-fill"
                        style={{
                          width: `${d.value}%`,
                          background:
                            d.name === "Critical"
                              ? "linear-gradient(90deg, #f43f5e, #fb7185)"
                              : d.name === "Warning"
                                ? "linear-gradient(90deg, #facc15, #f97316)"
                                : "linear-gradient(90deg, #34d399, #22d3ee)",
                        }}
                      ></span>
                    </div>
                    <span className="text-sm font-semibold">{d.value}%</span>
                  </div>
                ))}
              </div>
              <div className="grid gap-4">
                <article className="holo-card">
                  <p className="eyebrow">Signal Health</p>
                  <p className="text-base text-slate-300">
                    {metrics.good} units are trending in the safe zone. Edge
                    agents are pushing updates every 3 minutes.
                  </p>
                </article>
                <article className="holo-card">
                  <p className="eyebrow">Next Actions</p>
                  <ul className="text-sm text-slate-300 space-y-2">
                    <li>• Validate critical alerts &amp; confirm crew readiness.</li>
                    <li>• Align maintenance slots for warning tier equipment.</li>
                    <li>• Export daily snapshots for compliance board.</li>
                  </ul>
                </article>
              </div>
            </div>
          </section>

          <section className="glass-panel">
            <div className="panel-heading">
              <div>
                <p className="panel-label">Focus Board</p>
                <p className="text-xl font-semibold">Key Assets</p>
              </div>
              <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
                Filter: {activeFilterLabel}
              </p>
            </div>

            {topAssets.length ? (
              <div className="asset-grid">
                {topAssets.map((asset, index) => {
                  const { id, name } = getAssetMeta(asset);
                  const status = resolveStatus(asset);
                  const health = deriveHealth(asset);
                  const probability =
                    asset.lastPrediction?.probability !== undefined
                      ? Math.round(asset.lastPrediction.probability * 100)
                      : typeof asset.probability === "number"
                        ? Math.round(asset.probability * 100)
                        : null;
                  const statusTone =
                    status === "Good"
                      ? "chip chip-live"
                      : status === "Warning"
                        ? "chip chip-warning"
                        : "chip chip-critical";
                  const gradeColor =
                    status === "Good"
                      ? "linear-gradient(90deg,#34d399,#10b981)"
                      : status === "Warning"
                        ? "linear-gradient(90deg,#facc15,#f97316)"
                        : "linear-gradient(90deg,#f43f5e,#fb7185)";
                  return (
                    <div key={`${id}-${index}`} className="asset-tile">
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-500">
                            Unit {id}
                          </p>
                          <h4>{name}</h4>
                        </div>
                        <span className={statusTone}>{status}</span>
                      </div>
                      <div className="asset-stats">
                        <div className="asset-stat">
                          <small>Health</small>
                          <span>{health}%</span>
                        </div>
                        <div className="asset-stat">
                          <small>Failure</small>
                          <span>{probability !== null ? `${probability}%` : "—"}</span>
                        </div>
                      </div>
                      <div className="equipment-health">
                        <span
                          style={{
                            width: `${health}%`,
                            background: gradeColor,
                          }}
                        ></span>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="text-sm text-slate-400">
                No assets match this filter.
              </p>
            )}
          </section>
        </>
      )}
    </div>
  );
};

export default Dashboard;
