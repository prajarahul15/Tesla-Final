import React, { useEffect, useState, useMemo } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

/**
 * Agent-Based Vehicle Forecast Component
 * =========================================
 * 
 * This component displays pre-computed forecasts from the autonomous agent.
 * Key features:
 * - Auto-generated forecasts (no "Run Forecast" button needed)
 * - Shows BOTH univariate and multivariate in single table
 * - Forecasts production & deliveries
 * - ASP/Elasticity for simulation only (not forecasting)
 */

const VehicleForecastAgentBased = () => {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [forecastData, setForecastData] = useState(null);
  const [agentStatus, setAgentStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Simulation parameters (ASP & Elasticity)
  const [priceChange, setPriceChange] = useState(0);
  const [elasticity, setElasticity] = useState(-1.0);
  const [aspPerUnit, setAspPerUnit] = useState('');
  const [simulationResult, setSimulationResult] = useState(null);

  // Load models on mount
  useEffect(() => {
    const loadModels = async () => {
      try {
        const res = await axios.get(`${API}/vehicles/models`);
        const list = res.data?.models || [];
        setModels(list);
        if (list.length > 0) setSelectedModel(list[0].model_key);
      } catch (e) {
        setError('Failed to load vehicle models');
      }
    };
    loadModels();
  }, []);

  // Load agent status
  useEffect(() => {
    const loadAgentStatus = async () => {
      try {
        const res = await axios.get(`${API}/vehicles/forecast/agent-status`);
        if (res.data?.success) {
          setAgentStatus(res.data.agent_status);
        }
      } catch (e) {
        console.warn('Failed to load agent status:', e);
      }
    };
    loadAgentStatus();
    // Refresh status every 30 seconds
    const interval = setInterval(loadAgentStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  // Load forecast when model changes
  useEffect(() => {
    if (!selectedModel) return;
    loadForecast();
  }, [selectedModel]);

  const loadForecast = async () => {
    if (!selectedModel) return;
    
    setLoading(true);
    setError(null);
    setForecastData(null);
    setSimulationResult(null);
    
    try {
      const res = await axios.get(`${API}/vehicles/forecast/cached/${selectedModel}`);
      if (res.data?.success) {
        setForecastData(res.data);
      }
    } catch (e) {
      if (e.response?.status === 404) {
        setError('Forecast not yet available. Agent may still be processing. Please wait...');
      } else {
        setError(e?.response?.data?.detail || 'Failed to load forecast');
      }
    } finally {
      setLoading(false);
    }
  };

  const runSimulation = async () => {
    if (!selectedModel) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const payload = {
        model_key: selectedModel,
        price_change: priceChange / 100, // Convert percentage to decimal
        elasticity: elasticity,
        asp_per_unit: aspPerUnit ? parseFloat(aspPerUnit) : null
      };
      
      const res = await axios.post(`${API}/vehicles/forecast/simulate`, payload);
      if (res.data?.success) {
        setSimulationResult(res.data);
      }
    } catch (e) {
      setError(e?.response?.data?.detail || 'Simulation failed');
    } finally {
      setLoading(false);
    }
  };

  const refreshForecasts = async () => {
    setLoading(true);
    setError(null);
    
    try {
      await axios.post(`${API}/vehicles/forecast/refresh`);
      setTimeout(() => {
        loadForecast();
      }, 5000); // Wait 5 seconds then reload
    } catch (e) {
      setError(e?.response?.data?.detail || 'Refresh failed');
    } finally {
      setLoading(false);
    }
  };

  // Prepare base forecast table data
  const baseForecastTableData = useMemo(() => {
    if (!forecastData) return [];
    
    // Extract forecasts by target
    const univariate = forecastData.univariate?.forecasts_by_target || {};
    const multivariate = forecastData.multivariate?.forecasts_by_target || {};
    
    // Get available targets
    const availableTargets = forecastData.univariate?.available_targets || ['deliveries'];
    
    const rows = [];
    const maxLength = Math.max(
      ...(univariate.deliveries?.forecasts?.length ? [univariate.deliveries.forecasts.length] : [0]),
      ...(multivariate.deliveries?.forecasts?.length ? [multivariate.deliveries.forecasts.length] : [0])
    );
    
    for (let i = 0; i < maxLength; i++) {
      const row = {
        month_ahead: i + 1,
      };
      
      // Add forecasts for each target
      availableTargets.forEach(target => {
        const uni = univariate[target]?.forecasts?.[i] || {};
        const multi = multivariate[target]?.forecasts?.[i] || {};
        
        row.date = uni.date || multi.date || '';
        row[`univariate_${target}`] = uni.forecast || 0;
        row[`multivariate_${target}`] = multi.forecast || 0;
        
        if (uni.revenue) row[`univariate_${target}_revenue`] = uni.revenue;
        if (multi.revenue) row[`multivariate_${target}_revenue`] = multi.revenue;
      });
      
      rows.push(row);
    }
    
    return { rows, availableTargets };
  }, [forecastData]);

  // Prepare simulation forecast table data
  const simulationTableData = useMemo(() => {
    if (!simulationResult) return [];
    
    // Extract forecasts by target
    const univariate = simulationResult.univariate?.forecasts_by_target || {};
    const multivariate = simulationResult.multivariate?.forecasts_by_target || {};
    
    // Get available targets
    const availableTargets = simulationResult.univariate?.available_targets || ['deliveries'];
    
    const rows = [];
    const maxLength = Math.max(
      ...(univariate.deliveries?.forecasts?.length ? [univariate.deliveries.forecasts.length] : [0]),
      ...(multivariate.deliveries?.forecasts?.length ? [multivariate.deliveries.forecasts.length] : [0])
    );
    
    for (let i = 0; i < maxLength; i++) {
      const row = {
        month_ahead: i + 1,
      };
      
      // Add forecasts for each target
      availableTargets.forEach(target => {
        const uni = univariate[target]?.forecasts?.[i] || {};
        const multi = multivariate[target]?.forecasts?.[i] || {};
        
        row.date = uni.date || multi.date || '';
        row[`univariate_${target}`] = uni.forecast || 0;
        row[`multivariate_${target}`] = multi.forecast || 0;
        
        if (uni.revenue) row[`univariate_${target}_revenue`] = uni.revenue;
        if (multi.revenue) row[`multivariate_${target}_revenue`] = multi.revenue;
      });
      
      rows.push(row);
    }
    
    return { rows, availableTargets };
  }, [simulationResult]);

  // Calculate model performance metrics
  const performanceMetrics = useMemo(() => {
    if (!forecastData) return null;
    
    const univariate = forecastData.univariate?.forecasts_by_target || {};
    const multivariate = forecastData.multivariate?.forecasts_by_target || {};
    
    // Calculate average metrics across all targets
    const uni_targets = Object.values(univariate);
    const multi_targets = Object.values(multivariate);
    
    const uni_mae = uni_targets.length > 0 
      ? uni_targets.reduce((sum, t) => sum + (t.model_metrics?.mae || 0), 0) / uni_targets.length 
      : 0;
    const uni_mape = uni_targets.length > 0
      ? uni_targets.reduce((sum, t) => sum + (t.model_metrics?.mape || 0), 0) / uni_targets.length
      : 0;
    const multi_mae = multi_targets.length > 0
      ? multi_targets.reduce((sum, t) => sum + (t.model_metrics?.mae || 0), 0) / multi_targets.length
      : 0;
    const multi_mape = multi_targets.length > 0
      ? multi_targets.reduce((sum, t) => sum + (t.model_metrics?.mape || 0), 0) / multi_targets.length
      : 0;
    
    return {
      univariate: { mae: uni_mae, mape: uni_mape },
      multivariate: { mae: multi_mae, mape: multi_mape },
      by_target: {
        univariate,
        multivariate
      }
    };
  }, [forecastData]);

  const formatNumber = (num) => {
    return Math.round(num).toLocaleString();
  };

  const formatCurrency = (num) => {
    if (!num) return '-';
    return `$${(num / 1000000).toFixed(1)}M`;
  };

  // Get selected model display name
  const selectedModelName = useMemo(() => {
    const model = models.find(m => m.model_key === selectedModel);
    return model ? model.display_name : selectedModel;
  }, [models, selectedModel]);

  // Reusable forecast table component
  const ForecastTable = ({ title, data, isSimulation = false }) => {
    if (!data || !data.rows || data.rows.length === 0) return null;

    return (
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">
              {isSimulation ? '🎯 Simulated Forecast' : '📊 Base Forecast'}
            </h3>
            <div className="text-sm text-gray-600 mt-1">
              Model: <span className="font-semibold text-gray-900">{selectedModelName}</span>
            </div>
          </div>
          <div className="text-sm text-gray-600">
            {data.rows.length} months ahead | Targets: {data.availableTargets.join(', ')}
          </div>
        </div>
        
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Date
                </th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Month #
                </th>
                {data.availableTargets.map(target => (
                  <React.Fragment key={target}>
                    <th className="px-6 py-3 text-right text-xs font-medium text-blue-600 uppercase tracking-wider">
                      Univariate<br/>{target.charAt(0).toUpperCase() + target.slice(1)}
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-purple-600 uppercase tracking-wider">
                      Multivariate<br/>{target.charAt(0).toUpperCase() + target.slice(1)}
                    </th>
                  </React.Fragment>
                ))}
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {data.rows.map((row, idx) => (
                <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {row.date ? new Date(row.date).toLocaleDateString('en-GB', {year:'numeric', month:'short'}) : '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-center text-gray-600">
                    +{row.month_ahead}
                  </td>
                  {data.availableTargets.map(target => (
                    <React.Fragment key={target}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-blue-700 font-semibold">
                        {formatNumber(row[`univariate_${target}`] || 0)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-purple-700 font-semibold">
                        {formatNumber(row[`multivariate_${target}`] || 0)}
                      </td>
                    </React.Fragment>
                  ))}
                </tr>
              ))}
            </tbody>
            <tfoot className="bg-gray-100">
              <tr>
                <td colSpan={2} className="px-6 py-4 text-sm font-bold text-gray-900">
                  Total (12-month)
                </td>
                {data.availableTargets.map(target => (
                  <React.Fragment key={target}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-right font-bold text-blue-700">
                      {formatNumber(data.rows.reduce((sum, row) => sum + (row[`univariate_${target}`] || 0), 0))}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-right font-bold text-purple-700">
                      {formatNumber(data.rows.reduce((sum, row) => sum + (row[`multivariate_${target}`] || 0), 0))}
                    </td>
                  </React.Fragment>
                ))}
              </tr>
            </tfoot>
          </table>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white p-6 rounded-lg shadow">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold">🤖 Agent-Based Vehicle Forecast</h2>
            <p className="text-purple-100 mt-1">
              Autonomous AI agent pre-generates forecasts for all models
            </p>
          </div>
          <button
            onClick={refreshForecasts}
            disabled={loading || agentStatus?.is_running}
            className="bg-white text-purple-600 px-4 py-2 rounded-lg hover:bg-purple-50 disabled:opacity-50 disabled:cursor-not-allowed font-semibold"
          >
            🔄 Refresh Forecasts
          </button>
        </div>
        
        {/* Agent Status */}
        {agentStatus && (
          <div className="mt-4 bg-white/10 rounded-lg p-3">
            <div className="grid grid-cols-4 gap-4 text-sm">
              <div>
                <div className="text-purple-200">Status</div>
                <div className="font-semibold">
                  {agentStatus.is_running ? '🔄 Running' : '✅ Ready'}
                </div>
              </div>
              <div>
                <div className="text-purple-200">Cached Models</div>
                <div className="font-semibold">{agentStatus.cache_size || 0}</div>
              </div>
              <div>
                <div className="text-purple-200">Last Update</div>
                <div className="font-semibold">
                  {agentStatus.last_update 
                    ? new Date(agentStatus.last_update).toLocaleTimeString() 
                    : 'Never'}
                </div>
              </div>
              <div>
                <div className="text-purple-200">Cache Status</div>
                <div className="font-semibold">
                  {agentStatus.is_stale ? '⚠️ Stale' : '✅ Fresh'}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Model Selection */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Vehicle Model
            </label>
            <select
              value={selectedModel}
              onChange={e => setSelectedModel(e.target.value)}
              className="w-full p-3 border rounded-lg text-lg"
            >
              {models.map(m => (
                <option key={m.model_key} value={m.model_key}>
                  {m.display_name}
                </option>
              ))}
            </select>
          </div>
          
          {forecastData && (
            <div className="text-sm text-gray-600">
              <div>📅 Forecast Age: {forecastData.age_hours?.toFixed(1)} hours</div>
              <div>🎯 Horizon: {forecastData.metadata?.horizon_months} months</div>
            </div>
          )}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}

      {/* Loading State */}
      {loading && !forecastData && (
        <div className="bg-blue-50 border border-blue-200 text-blue-700 px-4 py-3 rounded flex items-center gap-2">
          <div className="animate-spin">⏳</div>
          <span>Loading forecast data...</span>
        </div>
      )}

      {/* Model Performance */}
      {performanceMetrics && (
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Performance (Test Period)</h3>
          <div className="grid grid-cols-2 gap-6">
            {/* Univariate */}
            <div>
              <div className="text-sm font-medium text-blue-600 mb-2">📊 Univariate Model</div>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-blue-50 p-3 rounded">
                  <div className="text-xs text-blue-800">MAE (test)</div>
                  <div className="text-2xl font-bold text-blue-700">
                    {formatNumber(performanceMetrics.univariate.mae)}
                  </div>
                </div>
                <div className="bg-green-50 p-3 rounded">
                  <div className="text-xs text-green-800">MAPE (test)</div>
                  <div className="text-2xl font-bold text-green-700">
                    {(performanceMetrics.univariate.mape * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
            
            {/* Multivariate */}
            <div>
              <div className="text-sm font-medium text-purple-600 mb-2">📈 Multivariate Model</div>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-purple-50 p-3 rounded">
                  <div className="text-xs text-purple-800">MAE (test)</div>
                  <div className="text-2xl font-bold text-purple-700">
                    {formatNumber(performanceMetrics.multivariate.mae)}
                  </div>
                </div>
                <div className="bg-green-50 p-3 rounded">
                  <div className="text-xs text-green-800">MAPE (test)</div>
                  <div className="text-2xl font-bold text-green-700">
                    {(performanceMetrics.multivariate.mape * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Simulation Controls */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="mb-4">
          <h3 className="text-lg font-semibold text-gray-900">
            🎛️ Forecast Simulation (ASP & Elasticity)
          </h3>
          <p className="text-sm text-gray-600 mt-1">
            Apply price and elasticity adjustments to see how they impact the base forecast
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Price Change (%): <span className="font-bold text-indigo-600">{priceChange}%</span>
            </label>
            <input
              type="range"
              min="-50"
              max="50"
              step="1"
              value={priceChange}
              onChange={e => setPriceChange(parseFloat(e.target.value) || 0)}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
              style={{
                background: `linear-gradient(to right, #4f46e5 0%, #4f46e5 ${(priceChange + 50) / 100 * 100}%, #e5e7eb ${(priceChange + 50) / 100 * 100}%, #e5e7eb 100%)`
              }}
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>-50%</span>
              <span>0%</span>
              <span>+50%</span>
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Elasticity: <span className="font-bold text-indigo-600">{elasticity}</span>
            </label>
            <input
              type="range"
              min="-5"
              max="0"
              step="0.1"
              value={elasticity}
              onChange={e => setElasticity(parseFloat(e.target.value) || -1.0)}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
              style={{
                background: `linear-gradient(to right, #4f46e5 0%, #4f46e5 ${(Math.abs(elasticity) / 5) * 100}%, #e5e7eb ${(Math.abs(elasticity) / 5) * 100}%, #e5e7eb 100%)`
              }}
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>-5.0</span>
              <span>-2.5</span>
              <span>0.0</span>
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              ASP per Unit ($)
            </label>
            <input
              type="number"
              value={aspPerUnit}
              onChange={e => setAspPerUnit(e.target.value)}
              className="w-full p-2 border rounded"
              placeholder="e.g., 52000"
            />
          </div>
          
          <div className="flex items-end">
            <button
              onClick={runSimulation}
              disabled={loading || !selectedModel}
              className="w-full bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Running...' : 'Run Simulation'}
            </button>
          </div>
        </div>
        
        <div className="mt-3 text-sm text-gray-600">
          <p>💡 <strong>Tip:</strong> Simulation applies price/elasticity adjustments to the base forecast without regenerating the ML model.</p>
        </div>
      </div>

      {/* Base Forecast Table */}
      <ForecastTable 
        data={baseForecastTableData} 
        isSimulation={false}
      />

      {/* Simulation Results Table */}
      {simulationResult && (
        <ForecastTable 
          data={simulationTableData} 
          isSimulation={true}
        />
      )}
    </div>
  );
};

export default VehicleForecastAgentBased;

