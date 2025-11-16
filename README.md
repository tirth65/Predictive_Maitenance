# Predictive Maintenance - Setup

## Environment

Create `Predictive_Maitenance/frontend/.env` with:
```
VITE_BACKEND_URL=http://localhost:5000
VITE_ML_API_URL=http://localhost:8000
```

## Run services
- Python ML API (Flask):
```
cd Predictive_Maitenance/ml-model
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
- Node backend:
```
cd Predictive_Maitenance/backend
npm install
node server.js
```
- Frontend:
```
cd Predictive_Maitenance/frontend
npm install
npm run dev
```

The predictions page supports:
- CSV upload to `/predict_file`
- Guided parameter entry to `/predict`
- Optional free-form text (mocked)