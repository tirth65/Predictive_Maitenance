// import express from "express";
// import cors from "cors";
// import sensorRoutes from "./routes/sensorRoutes.js";
// import predictionRoutes from "./routes/predictionRoutes.js";
// import maintenanceRoutes from "./routes/maintenanceRoutes.js";

// const app = express();

// // Middleware
// app.use(cors());
// app.use(express.json());

// // Routes
// app.use("/api/sensors", sensorRoutes);
// app.use("/api/predictions", predictionRoutes);
// app.use("/api/maintenance", maintenanceRoutes);

// // Health check route
// app.get("/", (req, res) => {
//   res.send("Predictive Maintenance API is running...");
// });

// export default app;

// backend/app.js
import express from "express";
import cors from "cors";
import sensorRoutes from "./routes/sensorRoutes.js";
import predictionRoutes from "./routes/predictionRoutes.js";
import maintenanceRoutes from "./routes/maintenanceRoutes.js";

const app = express();

// Middleware
// In prod, set a more restrictive origin list; using cors() without options allows all origins (dev)
app.use(cors());
app.use(express.json({ limit: "1mb" })); // avoid unexpectedly large payloads

// Routes
app.use("/api/sensors", sensorRoutes);
app.use("/api/predictions", predictionRoutes);
app.use("/api/maintenance", maintenanceRoutes);

// Health check route
app.get("/", (req, res) => {
  res.json({ status: "ok", service: "Predictive Maintenance API" });
});

// Basic error handler (so errors are returned as JSON)
app.use((err, req, res, next) => {
  // eslint-disable-next-line no-console
  console.error("Unhandled error:", err);
  const status = err.status || 500;
  res.status(status).json({ error: err.message || "Internal Server Error" });
});

export default app;
