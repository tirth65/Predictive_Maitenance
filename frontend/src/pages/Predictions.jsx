import React from "react";
import PredictionTable from "../components/PredictionTable";

const Predictions = () => (
  <div className="page-shell">
    <section className="glass-panel">
      <p className="eyebrow">Inference Studio</p>
      <h2 className="page-heading">Generate Predictions</h2>
      <p className="page-description">
        Upload CSV telemetry or stream in manual parameters to query the ML
        stack. Outputs blend probability, risk sentiment, and recommended
        maintenance actions.
      </p>
    </section>
    <PredictionTable />
  </div>
);

export default Predictions;