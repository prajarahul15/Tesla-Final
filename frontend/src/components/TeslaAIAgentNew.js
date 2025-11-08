import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { formatCompactNumber } from '../lib/utils';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const TeslaAIAgentNew = ({ initialScenario = 'base', year }) => {
  const [initialized, setInitialized] = useState(false);
  const [scenario, setScenario] = useState(initialScenario);
  const [sliders, setSliders] = useState({ asp_change: 0, cost_change: 0, delivery_change: 0 });
  const [simulation, setSimulation] = useState(null);
  const [vehicleModels, setVehicleModels] = useState({});
  const [excelModelKeys, setExcelModelKeys] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [generating, setGenerating] = useState(false);
  const getCacheKey = () => `tesla-ai-agent:new:${year || 'none'}:${scenario}`;
  const suppressNextInitRef = useRef(false);

  useEffect(() => {
    try {
      let isReload = false;
      const navEntries = typeof performance !== 'undefined' && performance.getEntriesByType ? performance.getEntriesByType('navigation') : [];
      if (navEntries && navEntries[0] && navEntries[0].type === 'reload') {
        isReload = true;
      } else if (performance && performance.navigation && performance.navigation.type === 1) {
        isReload = true;
      }
      if (isReload) {
        sessionStorage.removeItem(getCacheKey());
      }
    } catch {}

    try {
      const cached = sessionStorage.getItem(getCacheKey());
      if (cached) {
        const state = JSON.parse(cached);
        if (state && typeof state === 'object') {
          if (state.scenario) setScenario(state.scenario);
          if (state.sliders) setSliders(state.sliders);
          if (state.simulation) setSimulation(state.simulation);
          setInitialized(true);
          suppressNextInitRef.current = true;
        }
      }
    } catch {}
    fetchVehicleModels();
    if (!sessionStorage.getItem(getCacheKey())) {
      initializeAgent();
    }
  }, []);

  useEffect(() => { setScenario(initialScenario); }, [initialScenario]);

  useEffect(() => {
    if (initialized) {
      if (suppressNextInitRef.current) {
        suppressNextInitRef.current = false;
        return;
      }
      initializeAgent();
    }
  }, [scenario]);

  // Load Excel model keys for selected year to ensure all models (8) are displayed
  useEffect(() => {
    const loadExcelKeys = async () => {
      if (!year) { setExcelModelKeys([]); return; }
      try {
        const res = await axios.get(`${API}/vehicles/summary/${year}`);
        const keys = (res.data?.summary?.models || []).map(m => m.model_key);
        setExcelModelKeys(keys);
      } catch {
        setExcelModelKeys([]);
      }
    };
    loadExcelKeys();
  }, [year]);

  // Auto-sync impact analysis to the selected year and scenario
  useEffect(() => {
    if (!initialized) return;
    simulateChanges(sliders);
  }, [year, scenario]);

  const initializeAgent = async () => {
    try {
      setLoading(true);
      const endpoint = year
        ? `${API}/ai/tesla-agent/initialize-year?scenario=${scenario}&year=${year}`
        : `${API}/ai/tesla-agent/initialize?scenario=${scenario}`;
      const response = await axios.post(endpoint);
      if (response.data.success) {
        setInitialized(true);
        setSliders({ asp_change: 0, cost_change: 0, delivery_change: 0 });
        setSimulation(null);
      }
    } catch (err) {
      setError('Failed to initialize Tesla AI Agent');
      console.error('Initialization error:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchVehicleModels = async () => {
    try {
      const response = await axios.get(`${API}/ai/tesla-agent/vehicle-models`);
      if (response.data.success) {
        setVehicleModels(response.data.vehicle_models);
      }
    } catch (err) { console.error('Vehicle models error:', err); }
  };

  const handleSliderChange = async (sliderName, value) => {
    const newSliders = { ...sliders, [sliderName]: value };
    setSliders(newSliders);
    try {
      const cached = JSON.parse(sessionStorage.getItem(getCacheKey()) || '{}');
      sessionStorage.setItem(getCacheKey(), JSON.stringify({ scenario, sliders: newSliders, simulation: cached.simulation || null }));
    } catch {}
  };

  const handleGenerateClick = () => { simulateChanges(sliders); };

  const simulateChanges = async (currentSliders = sliders) => {
    try {
      setLoading(true);
      setGenerating(true);
      let response;
      if (year) {
        response = await axios.post(`${API}/ai/tesla-agent/simulate-year`, { scenario, year, changes: currentSliders });
      } else {
        response = await axios.post(`${API}/ai/tesla-agent/simulate`, { scenario, changes: currentSliders });
      }
      if (response.data.success) {
        setSimulation(response.data.simulation);
        try {
          sessionStorage.setItem(getCacheKey(), JSON.stringify({ scenario, sliders: currentSliders, simulation: response.data.simulation }));
        } catch {}
      }
    } catch (err) {
      setError('Failed to simulate changes');
      console.error('Simulation error:', err);
    } finally {
      setLoading(false);
      setGenerating(false);
    }
  };

  const resetSliders = () => {
    const zeros = { asp_change: 0, cost_change: 0, delivery_change: 0 };
    setSliders(zeros);
    setSimulation(null);
    try {
      sessionStorage.setItem(getCacheKey(), JSON.stringify({ scenario, sliders: zeros, simulation: null }));
    } catch {}
  };

  const formatCurrency = (value) => `$${formatCompactNumber(value)}`;
  const formatNumber = (value) => formatCompactNumber(value);

  // Format dollar amounts in text to billions (e.g., $1,562,313,453.75 -> $1.56B)
  const formatCurrencyInText = (text) => {
    if (!text || typeof text !== 'string') return text;
    
    // Match dollar amounts with various formats:
    // - $1,562,313,453.75 (with commas and decimals)
    // - $1562313453.75 (no commas, with decimals)
    // - $1,562,313,453 (with commas, no decimals)
    // Pattern: $ followed by 1-3 digits, then optional (comma + 3 digits) groups, then optional decimal part
    return text.replace(/\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+)/g, (match, numberStr) => {
      // Remove commas and parse the number
      const numValue = parseFloat(numberStr.replace(/,/g, ''));
      if (isNaN(numValue) || numValue < 1000000000) {
        // If less than 1 billion, return original format
        return match;
      }
      // Format to billions using the existing formatCompactNumber function
      return `$${formatCompactNumber(numValue)}`;
    });
  };

  const getInsightIcon = (type) => {
    const icons = { pricing_strategy:'💰', competitive_pressure:'🏁', production_scaling:'🏭', demand_concern:'📉', cost_inflation:'📈', operational_efficiency:'⚙️', financial_impact:'💼', error:'🤖', info:'ℹ️' };
    return icons[type] || '🤖';
  };
  const getRiskColor = (level) => { switch(level){case 'high':return 'border-red-200 bg-red-50 text-red-800';case 'medium':return 'border-yellow-200 bg-yellow-50 text-yellow-800';case 'low':return 'border-green-200 bg-green-50 text-green-800';default:return 'border-gray-200 bg-gray-50 text-gray-800';} };

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-red-600 to-orange-600 text-white p-6 rounded-lg shadow-lg">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold mb-2">🚗 Tesla AI Agent Updated</h2>
            <p className="text-red-100">Interactive model analysis with real-time insights</p>
          </div>
          <div className="flex items-center space-x-4">
            <select
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
              className="px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white"
            >
              <option value="best">Best Case</option>
              <option value="base">Base Case</option>
              <option value="worst">Worst Case</option>
            </select>
            <button onClick={handleGenerateClick} disabled={!initialized || loading || generating}
              className={`px-4 py-2 rounded-lg transition-colors flex items-center gap-2 ${(!initialized || loading || generating) ? 'bg-white/10 text-white/60 cursor-not-allowed' : 'bg-white/20 hover:bg-white/30 text-white'}`}
              aria-busy={generating} title={generating ? 'Generating insights…' : 'Generate insights'}>
              {generating ? (<><span className="inline-block h-4 w-4 border-b-2 border-white rounded-full animate-spin"></span>Generating…</>) : (<>⚡ Generate Insight</>)}
            </button>
            <button onClick={resetSliders} className="bg-white/20 hover:bg-white/30 text-white px-4 py-2 rounded-lg transition-colors">🔄 Reset</button>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-sm border">
        <div className="p-6 border-b">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center"><span className="mr-2">🎛️</span>Interactive Controls</h3>
          <p className="text-sm text-gray-600 mt-1">Adjust key parameters and see real-time impact</p>
        </div>
        <div className="p-6 space-y-6">
          <div>
            <div className="flex justify-between items-center mb-2"><label className="text-sm font-medium text-gray-700">Average Selling Price (ASP) Change</label><span className="text-lg font-bold text-blue-600">{sliders.asp_change > 0 ? '+' : ''}{sliders.asp_change}%</span></div>
            <input type="range" min="-30" max="30" step="1" value={sliders.asp_change} onChange={(e) => handleSliderChange('asp_change', parseFloat(e.target.value))} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider-blue" />
            <div className="relative flex justify-between text-xs text-gray-500 mt-1">
              <span>-30%</span>
              <span style={{position: 'absolute', left: '50%', transform: 'translateX(-50%)'}}>0%</span>
              <span>+30%</span>
            </div>
          </div>
          <div>
            <div className="flex justify-between items-center mb-2"><label className="text-sm font-medium text-gray-700">Cost Change</label><span className="text-lg font-bold text-red-600">{sliders.cost_change > 0 ? '+' : ''}{sliders.cost_change}%</span></div>
            <input type="range" min="-20" max="40" step="1" value={sliders.cost_change} onChange={(e) => handleSliderChange('cost_change', parseFloat(e.target.value))} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider-red" />
            <div className="relative flex justify-between text-xs text-gray-500 mt-1">
              <span>-20%</span>
              <span style={{position: 'absolute', left: '33.33%', transform: 'translateX(-50%)'}}>0%</span>
              <span>+40%</span>
            </div>
          </div>
          <div>
            <div className="flex justify-between items-center mb-2"><label className="text-sm font-medium text-gray-700">Delivery Volume Change</label><span className="text-lg font-bold text-green-600">{sliders.delivery_change > 0 ? '+' : ''}{sliders.delivery_change}%</span></div>
            <input type="range" min="-50" max="100" step="5" value={sliders.delivery_change} onChange={(e) => handleSliderChange('delivery_change', parseFloat(e.target.value))} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider-green" />
            <div className="relative flex justify-between text-xs text-gray-500 mt-1">
              <span>-50%</span>
              <span style={{position: 'absolute', left: '33.33%', transform: 'translateX(-50%)'}}>0%</span>
              <span>+100%</span>
            </div>
          </div>
        </div>
      </div>

      {simulation && (
        <div className="bg-white rounded-lg shadow-sm border">
          <div className="p-6 border-b"><h3 className="text-lg font-semibold text-gray-900 flex items-center"><span className="mr-2">📊</span>Impact Analysis</h3></div>
          <div className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <div className="bg-blue-50 p-4 rounded-lg border border-blue-200"><h4 className="font-semibold text-blue-900 mb-2">Revenue Impact</h4><div className="text-2xl font-bold text-blue-600">{formatCurrency(simulation.impact_analysis.revenue_impact.absolute_change)}</div><p className="text-sm text-blue-700 mt-1">{simulation.impact_analysis.revenue_impact.percentage_change > 0 ? '+' : ''}{simulation.impact_analysis.revenue_impact.percentage_change.toFixed(1)}% change</p></div>
              <div className="bg-green-50 p-4 rounded-lg border border-green-200"><h4 className="font-semibold text-green-900 mb-2">Margin Impact</h4><div className="text-2xl font-bold text-green-600">{simulation.impact_analysis.margin_impact.margin_change > 0 ? '+' : ''}{simulation.impact_analysis.margin_impact.margin_change.toFixed(1)}pp</div><p className="text-sm text-green-700 mt-1">New margin: {(simulation.impact_analysis.margin_impact.new_margin * 100).toFixed(1)}%</p></div>
              <div className="bg-purple-50 p-4 rounded-lg border border-purple-200"><h4 className="font-semibold text-purple-900 mb-2">Delivery Change</h4><div className="text-2xl font-bold text-purple-600">{simulation.impact_analysis.total_delivery_change > 0 ? '+' : ''}{formatNumber(simulation.impact_analysis.total_delivery_change)}</div><p className="text-sm text-purple-700 mt-1">Total units</p></div>
            </div>
            {/* Updated Vehicle Deliveries */}
            <div className="mb-8">
              <h4 className="font-semibold text-gray-900 mb-4">Updated Vehicle Deliveries</h4>
              {(() => {
                const deliveries = (simulation?.new_values?.deliveries) || {};
                const metaKeys = Object.keys(vehicleModels || {});
                const unionSet = new Set([...(excelModelKeys || []), ...metaKeys, ...Object.keys(deliveries || {})]);
                const ordered = Array.from(unionSet);
                return (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {ordered.map((model) => (
                      <div key={model} className="bg-gray-50 p-3 rounded-lg">
                        <div className="font-medium text-gray-900">{vehicleModels[model]?.name || model.replace('_', ' ')}</div>
                        <div className="text-lg font-bold text-gray-700">{formatNumber(deliveries[model] ?? 0)}</div>
                        <div className="text-xs text-gray-600">{vehicleModels[model]?.segment || 'Vehicle'}</div>
                      </div>
                    ))}
                  </div>
                );
              })()}
            </div>
          </div>
        </div>
      )}

      <div className="bg-white rounded-lg shadow-sm border"><div className="p-6 border-b"><h3 className="text-lg font-semibold text-gray-900 flex items-center"><span className="mr-2">🤖</span>AI Insights & Recommendations</h3></div><div className="p-6">{simulation && simulation.ai_insights && simulation.ai_insights.length > 0 ? (<div className="space-y-4">{simulation.ai_insights.map((insight, idx) => (<div key={idx} className={`p-4 rounded-lg border ${getRiskColor(insight.risk_level)}`}><div className="flex items-start space-x-3"><span className="text-2xl">{getInsightIcon(insight.type)}</span><div className="flex-1"><div className="flex items-center justify-between mb-2"><h4 className="font-semibold">{insight.title}</h4><span className={`px-2 py-1 text-xs rounded-full ${getRiskColor(insight.risk_level)}`}>{insight.risk_level} risk</span></div><p className="text-sm mb-3 whitespace-pre-wrap">{formatCurrencyInText(insight.description)}</p>{insight.recommendation && (<div className="bg-white/50 p-3 rounded border"><div className="text-xs font-medium mb-1">💡 Recommendation:</div><div className="text-sm whitespace-pre-wrap">{formatCurrencyInText(insight.recommendation)}</div></div>)}</div></div></div>))}</div>) : (<div className="text-sm text-gray-600">No insights yet. Adjust the sliders above and the agent will generate insights here.</div>)}</div></div>

      {loading && (<div className="bg-white rounded-lg shadow-sm border p-6"><div className="flex items-center justify-center py-8"><div className="animate-spin rounded-full h-8 w-8 border-b-2 border-red-600"></div><span className="ml-3 text-gray-600">Analyzing changes...</span></div></div>)}
      {error && (<div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg">{error}</div>)}

      <style jsx>{`
        .slider-blue::-webkit-slider-thumb { appearance: none; height: 20px; width: 20px; border-radius: 50%; background: #3B82F6; cursor: pointer; }
        .slider-red::-webkit-slider-thumb { appearance: none; height: 20px; width: 20px; border-radius: 50%; background: #EF4444; cursor: pointer; }
        .slider-green::-webkit-slider-thumb { appearance: none; height: 20px; width: 20px; border-radius: 50%; background: #10B981; cursor: pointer; }
      `}</style>
    </div>
  );
};

export default TeslaAIAgentNew;