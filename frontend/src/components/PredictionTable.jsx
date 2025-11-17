import React, { useState } from 'react';
import { TrendingUp, CheckCircle } from 'lucide-react';
import RiskGauge from './RiskGauge';
import { predictRow, predictFile } from '../services/API';

const PredictionTable = () => {
  // NOTE: free-form mock state removed to ensure every result originates from the backend model
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

  const formatError = (err) => {
    if (err?.response?.data?.error) return err.response.data.error;
    if (err?.code === 'ECONNABORTED') {
      return 'Prediction request timed out. Please try again after restarting the backend and ML services.';
    }
    if (err?.message === 'Network Error') {
      return 'Cannot reach the backend API. Ensure the Node backend is running on http://localhost:5000 (or update BACKEND_URL).';
    }
    return err?.message || 'Unexpected error while predicting.';
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0] || null);
  };

  const handleFormChange = (e) => {
    const { name, value } = e.target;
    setFormValues((prev) => ({ ...prev, [name]: value }));
  };

  const toUiFromModel = (data) => {
    // Handle enhanced API response format
    if (data.risk_level) {
      return {
        maintenanceNeeded: Number(data.prediction) === 1,
        healthScore: data.health_score,
        remainingDays: data.remaining_days,
        riskLevel: data.risk_level,
        riskScore: data.risk_score,
        probability: data.probability,
        modelUsed: data.model_used,
        recommendations: data.recommendations || [],
        allPredictions: data.all_predictions,
        allProbabilities: data.all_probabilities
      };
    }
    
    // Fallback for legacy format
    const needsMaintenance = Number(data.prediction) === 1;
    const prob = typeof data.probability === 'number' ? data.probability : undefined;
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
      remainingDays: typeof data.remaining_days === 'number' ? data.remaining_days : null,
      riskLevel,
      riskScore: prob,
      probability: prob,
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
      setError(formatError(err));
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
      
      // Handle enhanced file prediction response
      if (data.summary) {
        const summary = data.summary;
        const ui = toUiFromModel({
          prediction: summary.failure_predictions > summary.total_rows / 2 ? 1 : 0,
          probability: summary.average_failure_probability,
          risk_level: summary.risk_level,
          health_score: Math.round((1 - summary.average_failure_probability) * 100),
          remaining_days: summary.risk_level === 'High' ? 0 : 
                         summary.risk_level === 'Medium' ? 7 : 30,
          model_used: 'LGB', // Assuming LightGBM for file predictions
          recommendations: summary.risk_level === 'High' ? [
            'Immediate attention required',
            'Review all equipment in dataset',
            'Schedule comprehensive inspection'
          ] : summary.risk_level === 'Medium' ? [
            'Monitor equipment closely',
            'Schedule maintenance within 1-2 weeks',
            'Review sensor data trends'
          ] : [
            'Continue routine monitoring',
            'Maintain regular maintenance schedule',
            'Keep monitoring sensor data'
          ]
        });
        setPrediction(ui);
      } else {
        // Fallback for legacy format
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
      }
      
      setDebug({ request: { file: file.name }, response: data });
    } catch (err) {
      setError(formatError(err));
    } finally {
      setLoading(false);
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

          {/* Free-form mock prediction removed intentionally: we only show real model outputs */}

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
            {/* Model Information */}
            {prediction.modelUsed && (
              <div className="bg-blue-50 p-3 rounded-lg">
                <p className="text-sm text-blue-800">
                  <span className="font-medium">Model:</span> {prediction.modelUsed}
                  {prediction.modelUsed === 'LGB' && <span className="ml-2 text-xs bg-blue-200 px-2 py-1 rounded">LightGBM</span>}
                </p>
              </div>
            )}

            {/* Health and Risk Metrics */}
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

            {/* Risk Assessment */}
            <div className="space-y-3">
              <div>
                <p className="text-sm font-medium text-gray-700">Risk Assessment</p>
                <div className="flex items-center gap-3 mt-2">
                  <span className={`inline-flex px-3 py-1 text-sm font-semibold rounded-full ${
                    prediction.riskLevel === 'Low' ? 'bg-green-100 text-green-800' :
                    prediction.riskLevel === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {prediction.riskLevel} Risk
                  </span>
                  <span className={`text-sm font-medium ${
                    prediction.maintenanceNeeded ? 'text-red-600' : 'text-green-600'
                  }`}>
                    {prediction.maintenanceNeeded ? '⚠️ Needs Maintenance' : '✅ No Immediate Maintenance'}
                  </span>
                </div>
              </div>

              {/* Risk Score Bar */}
              {prediction.riskScore !== undefined && (
                <div>
                  <div className="flex justify-between text-sm text-gray-600 mb-1">
                    <span>Risk Score</span>
                    <span>{Math.round(prediction.riskScore * 100)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full transition-all duration-300 ${
                        prediction.riskLevel === 'Low' ? 'bg-green-500' :
                        prediction.riskLevel === 'Medium' ? 'bg-yellow-500' :
                        'bg-red-500'
                      }`}
                      style={{ width: `${prediction.riskScore * 100}%` }}
                    ></div>
                  </div>
                </div>
              )}

              {/* Probability Display */}
              {prediction.probability !== undefined && (
                <div className="bg-gray-50 p-3 rounded-lg">
                  <p className="text-sm text-gray-600">
                    <span className="font-medium">Failure Probability:</span> {Math.round(prediction.probability * 100)}%
                  </p>
                </div>
              )}
            </div>

            {/* Recommendations */}
            <div>
              <p className="text-sm font-medium text-gray-700 mb-2">Recommendations</p>
              <ul className="space-y-1">
                {prediction.recommendations.map((rec, index) => (
                  <li key={index} className="text-sm text-gray-600 flex items-center">
                    <CheckCircle className="mr-2 h-4 w-4 text-green-500 flex-shrink-0" />
                    {rec}
                  </li>
                ))}
              </ul>
            </div>

            {/* Model Comparison (if available) */}
            {prediction.allPredictions && Object.keys(prediction.allPredictions).length > 1 && (
              <details className="mt-4">
                <summary className="cursor-pointer text-sm text-gray-600 font-medium">Model Comparison</summary>
                <div className="mt-2 space-y-2">
                  {Object.entries(prediction.allPredictions).map(([model, pred]) => (
                    <div key={model} className="flex justify-between items-center bg-gray-50 p-2 rounded text-xs">
                      <span className="font-medium">{model.toUpperCase()}</span>
                      <div className="flex items-center gap-2">
                        <span className={`px-2 py-1 rounded ${
                          pred === 1 ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                        }`}>
                          {pred === 1 ? 'Failure' : 'Normal'}
                        </span>
                        {prediction.allProbabilities?.[model] && (
                          <span className="text-gray-600">
                            {Math.round(prediction.allProbabilities[model] * 100)}%
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </details>
            )}

            {/* Debug: raw API data */}
            {(debug.request || debug.response) && (
              <details className="mt-4">
                <summary className="cursor-pointer text-sm text-gray-600">Show debug (raw request/response)</summary>
                <pre className="mt-2 bg-gray-50 p-3 rounded text-xs overflow-auto max-h-40">{JSON.stringify(debug, null, 2)}</pre>
              </details>
            )}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <TrendingUp className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            <p>Upload a CSV or enter parameters to generate predictions</p>
          </div>
        )}
        {loading && <div className="text-sm text-gray-500 mt-2">Running prediction with LightGBM...</div>}
      </div>
    </div>
  );
};

export default PredictionTable;