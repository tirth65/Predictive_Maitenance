// // import mongoose from "mongoose";

// // const sensorDataSchema = new mongoose.Schema({
// //   machineId: { type: String, required: true },
// //   temperature: Number,
// //   vibration: Number,
// //   pressure: Number,
// //   timestamp: { type: Date, default: Date.now }
// // });

// // export default mongoose.model("SensorData", sensorDataSchema);

// // backend/models/sensordata.js
// import mongoose from "mongoose";

// const sensorDataSchema = new mongoose.Schema(
//   {
//     machineId: { type: String, required: true, index: true },
//     temperature: { type: Number, required: false },
//     vibration: { type: Number, required: false },
//     pressure: { type: Number, required: false },
//     timestamp: { type: Date, default: Date.now, index: true },
//     raw: { type: mongoose.Schema.Types.Mixed } // in case you have more fields
//   },
//   {
//     timestamps: true,
//     toJSON: { virtuals: true },
//     toObject: { virtuals: true }
//   }
// );

// // Index for efficient lookups by machine and time window
// sensorDataSchema.index({ machineId: 1, timestamp: -1 });

// export default mongoose.model("SensorData", sensorDataSchema);


// backend/models/sensordata.js
import mongoose from "mongoose";

const sensorDataSchema = new mongoose.Schema(
  {
    machineId: { type: String, required: true, index: true },
    temperature: { type: Number, required: false },
    vibration: { type: Number, required: false },
    pressure: { type: Number, required: false },
    timestamp: { type: Date, default: Date.now, index: true },
    raw: { type: mongoose.Schema.Types.Mixed }
  },
  {
    timestamps: true,
    toJSON: { virtuals: true },
    toObject: { virtuals: true }
  }
);

sensorDataSchema.index({ machineId: 1, timestamp: -1 });

const SensorData = mongoose.models.SensorData || mongoose.model("SensorData", sensorDataSchema);
export default SensorData;
