import express from "express";
import EquipmentState from "../models/Equipment.js";

const router = express.Router();

router.get("/", async (req, res) => {
  try {
    const equipment = await EquipmentState.find().sort({ updatedAt: -1 });
    res.json(equipment);
  } catch (err) {
    res.status(500).json({ error: err.message || "Failed to load equipment" });
  }
});

export default router;


