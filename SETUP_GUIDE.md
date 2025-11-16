# Predictive Maintenance System - Setup Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ with pip
- Node.js 16+ with npm
- MongoDB (optional, for backend storage)

### MongoDB Setup (Optional)
The system can run with or without MongoDB:

**Option 1: With MongoDB (recommended for production)**
1. Install MongoDB Community Server from https://www.mongodb.com/try/download/community
2. Start MongoDB service: `net start MongoDB`
3. Or use the provided script: `start_mongodb.bat`

**Option 2: Without MongoDB (for testing)**
- The system will work without MongoDB
- Predictions will be generated but not saved to database
- All ML functionality remains available

### 1. Install Dependencies

#### ML Service Dependencies
```bash
cd ml-model
pip install -r requirements.txt
```

#### Backend Dependencies
```bash
cd backend
npm install
```

#### Frontend Dependencies
```bash
cd frontend
npm install
```

### 2. Start Services

#### Option A: Use Batch Files (Windows)
1. Double-click `start_ml_service.bat` (starts ML API on port 8000)
2. Double-click `start_backend.bat` (starts Backend on port 5000)
3. Double-click `start_frontend.bat` (starts Frontend on port 5173)

#### Option B: Manual Start
```bash
# Terminal 1: Start ML Service
cd ml-model
python app.py

# Terminal 2: Start Backend
cd backend
npm start

# Terminal 3: Start Frontend
cd frontend
npm run dev
```

### 3. Test Connection
```bash
python test_connection.py
```

## 🔧 Configuration

### Environment Variables
The system uses configuration files instead of .env files:

- `frontend/env.config.js` - Frontend configuration
- `backend/config.env.js` - Backend configuration

### Default Ports
- Frontend: http://localhost:5173
- Backend: http://localhost:5000
- ML API: http://localhost:8000

## 📊 Using the System

### 1. Access Frontend
Open http://localhost:5173 in your browser

### 2. Make Predictions
- **Upload CSV**: Upload a CSV file with sensor data
- **Manual Input**: Enter sensor values manually:
  - Temperature (°C)
  - Vibration (mm/s)
  - Pressure (Pa)
  - RPM
  - Flow Rate
  - Operational Hours

### 3. View Results
The system will show:
- Health Score
- Risk Level (Low/Medium/High)
- Maintenance Recommendations
- Days Remaining (if available)

## 🐛 Troubleshooting

### Common Issues

1. **"Cannot connect to ML API"**
   - Make sure ML service is running on port 8000
   - Check if Python dependencies are installed

2. **"Cannot connect to Backend API"**
   - Make sure Backend is running on port 5000
   - Check if Node.js dependencies are installed

3. **"Model not loaded"**
   - Ensure model files exist in `ml-model/models/`
   - Check if `gb_model.pkl` and `feature_columns.pkl` are present

4. **Prediction errors**
   - Verify input data format matches expected structure
   - Check console logs for detailed error messages

### Data Format
The system expects sensor data with these fields:
- `temperature`: Temperature in Celsius
- `vibration`: Vibration in mm/s
- `pressure`: Pressure in Pa
- `rpm`: Rotations per minute
- `flow_rate`: Flow rate
- `operational_hours`: Total operational hours

## 📁 Project Structure
```
Predictive_Maitenance/
├── frontend/          # React frontend
├── backend/           # Node.js backend
├── ml-model/          # Python ML service
├── test_connection.py # Connection test script
└── start_*.bat        # Startup scripts
```

## 🔄 Data Flow
```
Frontend (React) → Backend (Node.js:5000) → ML API (Flask:8000) → Trained Model
```

## 📝 Notes
- The system uses Gradient Boosting model by default
- All trained models are available in `ml-model/models/`
- The system automatically maps frontend field names to training data column names
- MongoDB is optional - predictions work without database storage
