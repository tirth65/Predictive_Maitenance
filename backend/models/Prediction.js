// // import mongoose from "mongoose";

// // const predictionSchema = new mongoose.Schema({
// //   timestamp: { type: Date, default: Date.now },
// //   status: { type: String, enum: ["Healthy", "Maintenance Required"] },
// //   probability: Number,
// //   sensorSnapshot: Object,
// // });

// // export default mongoose.model("Prediction", predictionSchema);

// // backend/models/prediction.js
// import mongoose from "mongoose";

// const { Schema } = mongoose;

// const predictionSchema = new Schema(
//   {
//     timestamp: { type: Date, default: Date.now, index: true },
//     status: {
//       type: String,
//       enum: ["Healthy", "Maintenance Required"],
//       default: "Healthy",
//       required: true
//     },
//     probability: {
//       type: Number,
//       min: 0,
//       max: 1,
//       required: true
//     },
//     // store raw snapshot of sensor values; Mixed allows any object/array
//     sensorSnapshot: { type: Schema.Types.Mixed, required: true },
//     machineId: { type: String, required: false, index: true } // optional but useful to store
//   },
//   {
//     timestamps: true,
//     toJSON: { virtuals: true },
//     toObject: { virtuals: true }
//   }
// );

// // Add compound index for queries like "recent predictions for machineId"
// predictionSchema.index({ machineId: 1, timestamp: -1 });

// export default mongoose.model("Prediction", predictionSchema);

// backend/models/prediction.js
import mongoose from "mongoose";

const { Schema } = mongoose;

const predictionSchema = new Schema(
  {
    timestamp: { type: Date, default: Date.now, index: true },
    status: {
      type: String,
      enum: ["Healthy", "Maintenance Required"],
      default: "Healthy",
      required: true
    },
    probability: {
      type: Number,
      min: 0,
      max: 1,
      required: true
    },
    sensorSnapshot: { type: Schema.Types.Mixed, required: true },
    machineId: { type: String, required: false, index: true },
    modelVersion: { type: String, default: "unknown" }
  },
  {
    timestamps: true,
    toJSON: { virtuals: true },
    toObject: { virtuals: true }
  }
);

predictionSchema.index({ machineId: 1, timestamp: -1 });

const Prediction = mongoose.models.Prediction || mongoose.model("Prediction", predictionSchema);
export default Prediction;
