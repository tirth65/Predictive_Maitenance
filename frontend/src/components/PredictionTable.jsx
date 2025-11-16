import React, { useState } from 'react';
import { TrendingUp, CheckCircle } from 'lucide-react';
import RiskGauge from './RiskGauge';
import { predictRow, predictFile } from '../services/API';

const PredictionTable = () => {
  const [logInput, setLogInput] = useState('');
  const [file, setFile] = useState(null);
  const [formValues, setFormValues] = useState({
    temperature: '',
    pressure: '',
    vibration: '',
    rpm: ''
  });
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState('');
  const [debug, setDebug] = useState({ request: null, response: null });

  const handleFileChange = (e) => {
    setFile(e.target.files[0] || null);
  };

  const handleFormChange = (e) => {
    const { name, value } = e.target;
    setFormValues((prev) => ({ ...prev, [name]: value }));
  };

  const toUiFromModel = ({ prediction, probability, remaining_days }) => {
    const needsMaintenance = Number(prediction) === 1;
    const prob = typeof probability === 'number' ? probability : undefined;
    const healthScore = prob !== undefined ? Math.round((1 - prob) * 100) : undefined;
    const riskLevel = prob !== undefined
      ? (prob > 0.66 ? 'High' : prob > 0.33 ? 'Medium' : 'Low')
      : (needsMaintenance ? 'High' : 'Low');

    const recs = needsMaintenance
      ? ['Schedule inspection', 'Lubricate moving parts', 'Check for abnormal vibration']
      : ['Continue monitoring', 'Adhere to routine maintenance', 'Review sensor trends weekly'];

    return {
      maintenanceNeeded: needsMaintenance,
      healthScore,
      remainingDays: typeof remaining_days === 'number' ? remaining_days : null,
      riskLevel,
      recommendations: recs
    };
  };

  const handlePredictFromForm = async () => {
    setError('');
    setLoading(true);
    try {
      const sensors = Object.fromEntries(
        Object.entries(formValues).map(([k, v]) => [k, v === '' ? 0 : Number(v)])
      );
      const data = await predictRow(sensors);
      setPrediction(toUiFromModel(data));
      setDebug({ request: { sensors }, response: data });
    } catch (err) {
      setError(err?.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  const handlePredictFromFile = async () => {
    if (!file) return;
    setError('');
    setLoading(true);
    try {
      const data = await predictFile(file);
      const probs = data.probabilities || [];
      const preds = data.predictions || [];
      let prob = undefined;
      if (probs.length) prob = probs.reduce((a, b) => a + b, 0) / probs.length;
      const majorityNeeds = preds.length ? preds.filter(p => Number(p) === 1).length > preds.length / 2 : undefined;
      const ui = toUiFromModel({
        prediction: majorityNeeds === undefined ? 0 : (majorityNeeds ? 1 : 0),
        probability: prob,
      });
      setPrediction(ui);
      setDebug({ request: { file: file.name }, response: data });
    } catch (err) {
      setError(err?.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  const handlePredictFromText = () => {
    if (logInput.trim()) {
      setPrediction({
        maintenanceNeeded: false,
        healthScore: undefined,
        remainingDays: null,
        riskLevel: 'Low',
        recommendations: ['Paste structured parameters or upload a CSV for real results']
      });
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Input Section */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h3 className="text-lg font-semibold mb-4">Equipment Log Input</h3>
        <div className="space-y-6">
          {/* Step 1: CSV Upload */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Upload Log File (CSV)</label>
            <input type="file" accept=".csv" onChange={handleFileChange} className="w-full" />
            <button
              onClick={handlePredictFromFile}
              disabled={!file || loading}
              className="mt-2 w-full bg-indigo-500 disabled:opacity-60 text-white py-2 px-4 rounded-lg hover:bg-indigo-600"
            >
              Predict from File
            </button>
          </div>

          {/* Step 2: Guided Form */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Or enter parameters</label>
            <div className="grid grid-cols-2 gap-3">
              {Object.keys(formValues).map((key) => (
                <input
                  key={key}
                  name={key}
                  type="number"
                  value={formValues[key]}
                  onChange={handleFormChange}
                  placeholder={key.replace('_', ' ')}
                  className="p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              ))}
            </div>
            <button
              onClick={handlePredictFromForm}
              disabled={loading}
              className="mt-2 w-full bg-blue-500 disabled:opacity-60 text-white py-2 px-4 rounded-lg hover:bg-blue-600"
            >
              Predict from Parameters
            </button>
          </div>

          {/* Optional: Free-form text (legacy) */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Free-form log (optional)</label>
            <textarea
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              rows="6"
              placeholder="Paste raw log text here..."
              value={logInput}
              onChange={(e) => setLogInput(e.target.value)}
            />
            <button
              onClick={handlePredictFromText}
              className="mt-2 w-full bg-gray-600 text-white py-2 px-4 rounded-lg hover:bg-gray-700"
            >
              Quick Mock Prediction
            </button>
          </div>

          {error && (
            <div className="text-red-600 text-sm">{error}</div>
          )}
        </div>
      </div>

      {/* Results Section */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h3 className="text-lg font-semibold mb-4">Prediction Results</h3>
        {prediction ? (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <RiskGauge value={prediction.healthScore ?? 50} label="Health Score" />
              {prediction.remainingDays !== null && (
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">{prediction.remainingDays}</div>
                    <div className="text-gray-600 text-sm">Days Remaining</div>
                  </div>
                </div>
              )}
            </div>
            <div>
              <p className="text-sm font-medium text-gray-700">Risk Level</p>
              <span className={`inline-flex px-3 py-1 text-sm font-semibold rounded-full mt-1 ${
                prediction.riskLevel === 'Low' ? 'bg-green-100 text-green-800' :
                prediction.riskLevel === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                'bg-red-100 text-red-800'
              }`}>
                {prediction.riskLevel} Risk — {prediction.maintenanceNeeded ? 'Needs Maintenance' : 'No Immediate Maintenance'}
              </span>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-700 mb-2">Recommendations</p>
              <ul className="space-y-1">
                {prediction.recommendations.map((rec, index) => (
                  <li key={index} className="text-sm text-gray-600 flex items-center">
                    <CheckCircle className="mr-2 h-4 w-4 text-green-500" />
                    {rec}
                  </li>
                ))}
              </ul>
            </div>

            {/* Debug: raw API data */}
            {(debug.request || debug.response) && (
              <details className="mt-4">
                <summary className="cursor-pointer text-sm text-gray-600">Show debug (raw request/response)</summary>
                <pre className="mt-2 bg-gray-50 p-3 rounded text-xs overflow-auto">{JSON.stringify(debug, null, 2)}</pre>
              </details>
            )}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <TrendingUp className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            <p>Upload a CSV or enter parameters to generate predictions</p>
          </div>
        )}
        {loading && <div className="text-sm text-gray-500 mt-2">Running prediction...</div>}
      </div>
    </div>
  );
};

export default PredictionTable;