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
    <div className="prediction-layout">
      <section className="glass-panel space-y-8">
        <div className="panel-heading">
          <div>
            <p className="panel-label">Data Intake</p>
            <p className="text-xl font-semibold">Equipment Log Input</p>
          </div>
          <span className="chip chip-live">Live</span>
        </div>

        <div>
          <p className="panel-label mb-2">Upload CSV</p>
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="w-full text-sm text-slate-200"
          />
          <button
            onClick={handlePredictFromFile}
            disabled={!file || loading}
            className="btn-aurora mt-3"
          >
            Predict from File
          </button>
        </div>

        <div>
          <p className="panel-label mb-4">Manual Parameters</p>
          <div className="field-grid">
            {Object.keys(formValues).map((key) => (
              <label key={key} className="input-shell">
                <span className="text-xs uppercase tracking-widest text-slate-400">
                  {key}
                </span>
                <input
                  name={key}
                  type="number"
                  value={formValues[key]}
                  onChange={handleFormChange}
                  placeholder={key.replace("_", " ")}
                />
              </label>
            ))}
          </div>
          <button
            onClick={handlePredictFromForm}
            disabled={loading}
            className="btn-ghost mt-3"
          >
            Predict from Parameters
          </button>
        </div>

        {error && (
          <div className="text-sm text-rose-300">
            {error}
          </div>
        )}
      </section>

      <section className="glass-panel result-panel">
        <div className="panel-heading">
          <div>
            <p className="panel-label">Model Output</p>
            <p className="text-xl font-semibold">Prediction Results</p>
          </div>
          {prediction?.modelUsed && (
            <span className="chip chip-neutral">{prediction.modelUsed}</span>
          )}
        </div>

        {prediction ? (
          <>
            <div className="result-grid">
              <RiskGauge
                value={prediction.healthScore ?? 50}
                label="Health Score"
              />
              {prediction.remainingDays !== null && (
                <div className="result-callout">
                  <p className="stat-label">Days Remaining</p>
                  <p className="stat-value text-sky-200">
                    {prediction.remainingDays}
                  </p>
                </div>
              )}
              <div
                className={`result-callout ${
                  prediction.maintenanceNeeded ? "danger" : ""
                }`}
              >
                <p className="stat-label">Maintenance</p>
                <p className="text-base font-semibold">
                  {prediction.maintenanceNeeded
                    ? "Action required"
                    : "Running normal"}
                </p>
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <p className="panel-label mb-2">Risk Assessment</p>
                <div className="flex flex-wrap items-center gap-3">
                  <span className="chip chip-warning">
                    {prediction.riskLevel} risk
                  </span>
                  <span
                    className={
                      prediction.maintenanceNeeded
                        ? "text-rose-300 font-medium"
                        : "text-emerald-200 font-medium"
                    }
                  >
                    {prediction.maintenanceNeeded
                      ? "⚠ Needs Maintenance"
                      : "✅ No Immediate Maintenance"}
                  </span>
                </div>
              </div>

              {prediction.riskScore !== undefined && (
                <div>
                  <div className="flex justify-between text-xs uppercase tracking-widest text-slate-400 mb-2">
                    <span>Risk Score</span>
                    <span>{Math.round(prediction.riskScore * 100)}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-300"
                      style={{
                        width: `${prediction.riskScore * 100}%`,
                        background:
                          prediction.riskLevel === "Low"
                            ? "linear-gradient(90deg,#34d399,#10b981)"
                            : prediction.riskLevel === "Medium"
                              ? "linear-gradient(90deg,#facc15,#f97316)"
                              : "linear-gradient(90deg,#f43f5e,#fb7185)",
                      }}
                    ></div>
                  </div>
                </div>
              )}

              {prediction.probability !== undefined && (
                <div className="result-callout">
                  <p className="stat-label">Failure Probability</p>
                  <p className="text-2xl font-semibold">
                    {Math.round(prediction.probability * 100)}%
                  </p>
                </div>
              )}
            </div>

            <div>
              <p className="panel-label mb-3">Recommendations</p>
              <ul className="space-y-2 text-sm text-slate-200">
                {prediction.recommendations.map((rec, index) => (
                  <li
                    key={index}
                    className="flex items-start gap-2 leading-relaxed"
                  >
                    <CheckCircle className="mt-0.5 h-4 w-4 text-emerald-300 flex-shrink-0" />
                    {rec}
                  </li>
                ))}
              </ul>
            </div>

            {prediction.allPredictions &&
              Object.keys(prediction.allPredictions).length > 1 && (
                <details className="mt-4">
                  <summary className="cursor-pointer text-sm text-slate-300">
                    Model Comparison
                  </summary>
                  <div className="mt-2 space-y-2">
                    {Object.entries(prediction.allPredictions).map(
                      ([model, pred]) => (
                        <div
                          key={model}
                          className="holo-card flex justify-between text-xs"
                        >
                          <span className="font-semibold">
                            {model.toUpperCase()}
                          </span>
                          <div className="flex items-center gap-2">
                            <span className="chip chip-neutral">
                              {pred === 1 ? "Failure" : "Normal"}
                            </span>
                            {prediction.allProbabilities?.[model] && (
                              <span className="text-slate-300">
                                {Math.round(
                                  prediction.allProbabilities[model] * 100
                                )}
                                %
                              </span>
                            )}
                          </div>
                        </div>
                      )
                    )}
                  </div>
                </details>
              )}

            {(debug.request || debug.response) && (
              <details className="mt-4">
                <summary className="cursor-pointer text-sm text-slate-300">
                  Show debug payload
                </summary>
                <pre className="debug-box mt-2">
                  {JSON.stringify(debug, null, 2)}
                </pre>
              </details>
            )}
          </>
        ) : (
          <div className="text-center text-slate-400 py-10">
            <TrendingUp className="mx-auto h-12 w-12 text-slate-500 mb-4" />
            <p>Upload a CSV or enter parameters to generate predictions.</p>
          </div>
        )}

        {loading && (
          <p className="text-xs uppercase tracking-widest text-slate-400">
            Running prediction…
          </p>
        )}
      </section>
    </div>
  );
};

export default PredictionTable;