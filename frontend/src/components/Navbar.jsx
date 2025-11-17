import React from "react";
import { NavLink } from "react-router-dom";

const Navbar = () => {
  const tabs = [
    { name: "Dashboard", path: "/dashboard" },
    { name: "Equipment", path: "/equipment" },
    { name: "Predictions", path: "/predictions" },
    { name: "Logs", path: "/logs" },
  ];

  return (
    <header className="nav-shell glass-panel">
      <div className="nav-brand">
        <div className="brand-icon">PM</div>
        <div>
          <p className="eyebrow">Predictive Core</p>
          <p className="text-lg font-semibold tracking-tight leading-snug">
            Maintenance OS
          </p>
        </div>
      </div>

      <nav className="nav-tabs">
        {tabs.map((tab) => (
          <NavLink
            key={tab.name}
            to={tab.path}
            className={({ isActive }) =>
              `nav-pill ${isActive ? "is-active" : ""}`
            }
          >
            {tab.name}
          </NavLink>
        ))}
      </nav>

      <div className="nav-status">
        <span className="pulse-dot" aria-hidden="true"></span>
        <span>Systems nominal</span>
      </div>
    </header>
  );
};

export default Navbar;
