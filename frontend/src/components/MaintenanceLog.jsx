import React from "react";
import { Upload } from "lucide-react";

const MaintenanceLog = () => {
  const logs = [
    {
      date: "2024-09-23",
      equipment: "Motor A1",
      activity: "Routine Inspection",
      status: "Completed",
      technician: "John Doe",
    },
    {
      date: "2024-09-22",
      equipment: "Pump B2",
      activity: "Oil Change",
      status: "Completed",
      technician: "Jane Smith",
    },
    {
      date: "2024-09-21",
      equipment: "Compressor C3",
      activity: "Emergency Repair",
      status: "In Progress",
      technician: "Mike Johnson",
    },
    {
      date: "2024-09-20",
      equipment: "Generator D4",
      activity: "Filter Replacement",
      status: "Scheduled",
      technician: "Sarah Wilson",
    },
  ];

  const statusClass = (status) => {
    if (status === "Completed") return "status-pill status-completed";
    if (status === "In Progress") return "status-pill status-progress";
    return "status-pill status-scheduled";
  };

  return (
    <section className="glass-panel">
      <div className="panel-heading">
        <div>
          <p className="panel-label">Recent Activity</p>
          <p className="text-xl font-semibold">Maintenance Timeline</p>
        </div>
        <button
          type="button"
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl border border-slate-500 text-xs uppercase tracking-widest text-slate-200 hover:border-sky-400 transition"
        >
          <Upload className="h-4 w-4" />
          Import
        </button>
      </div>

      <div className="overflow-x-auto mt-4">
        <table className="log-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Equipment</th>
              <th>Activity</th>
              <th>Status</th>
              <th>Technician</th>
            </tr>
          </thead>
          <tbody>
            {logs.map((log, index) => (
              <tr key={index}>
                <td>{log.date}</td>
                <td>{log.equipment}</td>
                <td>{log.activity}</td>
                <td>
                  <span className={statusClass(log.status)}>{log.status}</span>
                </td>
                <td>{log.technician}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
};

export default MaintenanceLog;