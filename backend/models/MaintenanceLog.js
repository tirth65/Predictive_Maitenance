// // import mongoose from "mongoose";

// // const maintenanceLogSchema = new mongoose.Schema({
// //   machineId: { type: String, required: true },
// //   actionTaken: { type: String, required: true },
// //   technician: String,
// //   date: { type: Date, default: Date.now }
// // });

// // export default mongoose.model("MaintenanceLog", maintenanceLogSchema);

// // backend/models/maintenancelogs.js
// import mongoose from "mongoose";

// const maintenanceLogSchema = new mongoose.Schema(
//   {
//     machineId: { type: String, required: true, index: true },
//     actionTaken: { type: String, required: true },
//     technician: { type: String, default: "unknown" },
//     date: { type: Date, default: Date.now }
//   },
//   {
//     timestamps: true, // adds createdAt & updatedAt
//     toJSON: { virtuals: true },
//     toObject: { virtuals: true }
//   }
// );

// // Optional index on date if you will query by date ranges
// maintenanceLogSchema.index({ date: -1 });

// export default mongoose.model("MaintenanceLog", maintenanceLogSchema);

// backend/models/maintenancelogs.js
import mongoose from "mongoose";

const maintenanceLogSchema = new mongoose.Schema(
  {
    machineId: { type: String, required: true, index: true },
    actionTaken: { type: String, required: true },
    technician: { type: String, default: "unknown" },
    date: { type: Date, default: Date.now }
  },
  {
    timestamps: true,
    toJSON: { virtuals: true },
    toObject: { virtuals: true }
  }
);

maintenanceLogSchema.index({ date: -1 });

const MaintenanceLog = mongoose.models.MaintenanceLog || mongoose.model("MaintenanceLog", maintenanceLogSchema);
export default MaintenanceLog;
