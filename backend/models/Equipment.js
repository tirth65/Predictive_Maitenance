import mongoose from "mongoose";

const equipmentStateSchema = new mongoose.Schema(
  {
    machineId: { type: String, required: true, unique: true, index: true },
    name: { type: String },
    lastPrediction: {
      prediction: Number,
      probability: Number,
      healthScore: Number,
      riskLevel: String,
      remainingDays: Number,
      maintenanceNeeded: Boolean,
      riskScore: Number,
      modelUsed: String,
      recommendations: [String],
      updatedAt: { type: Date, default: Date.now },
    },
    lastSensors: {
      temperature: Number,
      vibration: Number,
      pressure: Number,
      rpm: Number,
    },
    lastSeenAt: { type: Date, default: Date.now },
  },
  {
    timestamps: true,
  }
);

equipmentStateSchema.index({ machineId: 1 });

const EquipmentState =
  mongoose.models.EquipmentState ||
  mongoose.model("EquipmentState", equipmentStateSchema);

export default EquipmentState;


