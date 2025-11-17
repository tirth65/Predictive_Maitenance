import { Routes, Route, Navigate } from "react-router-dom";
import Navbar from "./components/Navbar.jsx";
import Dashboard from "./pages/Dashboard";
import Equipment from "./pages/Equipment"; 
// import Sensors from "./pages/Sensors";
import Predictions from "./pages/Predictions";
import Logs from "./pages/Logs";
// import NotFound from "./pages/NotFound.jsx";

export default function App() {
  return (
    <div className="app-surface">
      <div className="nebula-blur nebula-one" aria-hidden="true"></div>
      <div className="nebula-blur nebula-two" aria-hidden="true"></div>
      <div className="grid-overlay" aria-hidden="true"></div>
      <Navbar />
      <main className="app-main">
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          {/* <Route path="/sensors" element={<Sensors />} /> */}
          <Route path="/equipment" element={<Equipment />} />
          <Route path="/predictions" element={<Predictions />} />
          <Route path="/logs" element={<Logs />} />
          {/* <Route path="*" element={<NotFound />} /> */}
        </Routes>
      </main>
    </div>
  );
}