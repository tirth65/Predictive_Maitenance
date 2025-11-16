import React, { useEffect, useState } from "react";
import { fetchSensors } from "../services/API"; // ensure path matches

const SensorTable = () => {
  const [equipment, setEquipment] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      setLoading(true);
      setError("");
      try {
        const data = await fetchSensors();
        if (!mounted) return;
        setEquipment(Array.isArray(data) ? data : []);
      } catch (err) {
        if (!mounted) return;
        setError(err?.response?.data?.error || err.message || "Failed to load equipment");
      } finally {
        if (mounted) setLoading(false);
      }
    };
    load();
    return () => { mounted = false; };
  }, []);

  if (loading) {
    return (
      <div className="bg-white rounded-lg p-6 border border-gray-200 text-center">
        <div className="text-gray-500">Loading equipment...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg p-6 border border-red-200 text-center">
        <div className="text-red-600">Error: {error}</div>
      </div>
    );
  }

  if (!equipment.length) {
    return (
      <div className="bg-white rounded-lg p-6 border border-gray-200 text-center">
        <div className="text-gray-500">No equipment data available</div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold">Equipment Status</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Equipment</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Health Score</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Days Remaining</th>
            </tr>
          </thead>
          <tbody>
            {equipment.map((equipmentItem) => {
              const id = equipmentItem.id || equipmentItem._id || equipmentItem.machineId || "unknown";
              const name = equipmentItem.name || equipmentItem.equipment || id;
              const status = equipmentItem.status || (equipmentItem.maintenanceNeeded ? "Warning" : "Good");
              const health = equipmentItem.health ?? equipmentItem.healthScore ?? Math.round((1 - (equipmentItem.probability ?? 0)) * 100);
              const daysLeft = equipmentItem.daysLeft ?? equipmentItem.remainingDays ?? "-";

              return (
                <tr key={id} className="border-b border-gray-200">
                  <td className="px-6 py-4">
                    <div className="font-medium text-gray-900">{name}</div>
                    <div className="text-sm text-gray-500">{id}</div>
                  </td>
                  <td className="px-6 py-4">
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                      status === 'Good' ? 'bg-green-100 text-green-800' :
                      status === 'Warning' ? 'bg-yellow-100 text-yellow-800' :
                      'bg-red-100 text-red-800'
                    }`}>
                      {status}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center">
                      <span className="mr-2">{health}%</span>
                      <div className="w-16 bg-gray-200 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${health >= 80 ? 'bg-green-500' : health >= 50 ? 'bg-yellow-500' : 'bg-red-500'}`}
                          style={{ width: `${health}%` }}
                        />
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-gray-900">{daysLeft} {typeof daysLeft === "number" ? "days" : ""}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default SensorTable;