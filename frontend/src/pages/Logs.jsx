import React from "react";
import MaintenanceLog from "../components/MaintenanceLog";

const Logs = () => (
  <div className="page-shell">
    <section className="glass-panel">
      <p className="eyebrow">Mission Log</p>
      <h2 className="page-heading">Maintenance History</h2>
      <p className="page-description">
        Trace every intervention, technician note, and upcoming schedule in a
        single stream. Export-ready and synced with the predictive pipeline.
      </p>
    </section>
    <MaintenanceLog />
  </div>
);

export default Logs;