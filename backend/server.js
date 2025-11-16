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

let server;

// Connect to MongoDB and start server
async function start() {
  try {
    await mongoose.connect(MONGO_URI);
    console.log("✅ MongoDB connected");
  } catch (err) {
    console.warn("⚠️ MongoDB connection failed. Starting server anyway.");
    console.warn("   Error:", err.message);
    // If DB is essential, you might prefer to `process.exit(1)` here instead.
  }

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
  try {
    await mongoose.disconnect();
    console.log("MongoDB disconnected");
  } catch (e) {
    // ignore
  }
  process.exit(0);
}

start();
