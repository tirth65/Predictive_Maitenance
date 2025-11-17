import React from "react";

const RiskGauge = ({ value = 0, label }) => {
  const safeValue = Math.max(0, Math.min(100, Number(value) || 0));

  const getGradient = () => {
    if (safeValue >= 80) return "linear-gradient(135deg,#34d399,#22d3ee)";
    if (safeValue >= 50) return "linear-gradient(135deg,#facc15,#f97316)";
    return "linear-gradient(135deg,#f43f5e,#fb7185)";
  };

  return (
    <div className="gauge-shell">
      <div className="gauge">
        <div
          className="gauge-ring"
          style={{
            background: `conic-gradient(${getGradient()} ${safeValue}%, rgba(30,41,59,0.7) ${safeValue}% 100%)`,
          }}
        ></div>
        <div className="gauge-core">
          <span>{safeValue}%</span>
          <p>{label}</p>
        </div>
      </div>
    </div>
  );
};

export default RiskGauge;