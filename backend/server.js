// import mongoose from "mongoose";
// import dotenv from "dotenv";
// import app from "./app.js";

// dotenv.config();

// const PORT = process.env.PORT || 5000;
// const MONGO_URI = process.env.MONGO_URI || "mongodb://localhost:27017/predictive_maintenance";

// // MongoDB connection (optional)
// mongoose
//   .connect(MONGO_URI)
//   .then(() => {
//     console.log("✅ MongoDB connected");
//     app.listen(PORT, () => {
//       console.log(`🚀 Server running on http://localhost:${PORT}`);
//     });
//   })
//   .catch((err) => {
//     console.warn("⚠️ MongoDB connection failed - running without database storage");
//     console.warn("   Error:", err.message);
//     console.warn("   Predictions will work but won't be saved to database");
//     app.listen(PORT, () => {
//       console.log(`🚀 Server running on http://localhost:${PORT}`);
//     });
//   });

// backend/server.js
import dotenv from "dotenv";
dotenv.config(); // load env as early as possible

import mongoose from "mongoose";
import app from "./app.js";

const PORT = process.env.PORT || 5000;
const MONGO_URI =
  process.env.MONGO_URI || "mongodb://localhost:27017/predictive_maintenance";
const ENABLE_DB_FLAG = (process.env.ENABLE_DB ?? "true").toLowerCase();
const ENABLE_DB = ENABLE_DB_FLAG !== "false" && ENABLE_DB_FLAG !== "0";

let server;
let dbConnected = false;

async function maybeConnectDb() {
  if (!ENABLE_DB) {
    console.log("ℹ️ MongoDB connection disabled (set ENABLE_DB=true to force connection).");
    return;
  }
  try {
    await mongoose.connect(MONGO_URI);
    dbConnected = true;
    console.log("✅ MongoDB connected");
  } catch (err) {
    console.warn("⚠️ MongoDB connection failed. Starting server without database.");
    console.warn("   Error:", err.message);
  }
}

// Connect to MongoDB and start server
async function start() {
  await maybeConnectDb();

  server = app.listen(PORT, () => {
    console.log(`🚀 Server running on http://localhost:${PORT}`);
  });

  // Graceful shutdown hooks
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

async function shutdown() {
  console.log("Shutting down server...");
  if (server) server.close();
  if (dbConnected) {
    try {
      await mongoose.disconnect();
      console.log("MongoDB disconnected");
    } catch (e) {
      // ignore
    }
  }
  process.exit(0);
}

start();
