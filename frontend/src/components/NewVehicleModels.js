import React, { useState, useEffect, Suspense, lazy } from 'react';
import axios from 'axios';
import { formatCompactNumber } from '../lib/utils';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const TeslaAIAgent = lazy(() => import('./TeslaAIAgentNew'));

const NewVehicleModels = () => {
  const [years, setYears] = useState([]);
  const [selectedYear, setSelectedYear] = useState(null);
  const [currentSummary, setCurrentSummary] = useState(null);
  const [prevSummary, setPrevSummary] = useState(null);
  const [modelMeta, setModelMeta] = useState({});
  const [selectedScenario, setSelectedScenario] = useState('base');
  const [scenarioInputs, setScenarioInputs] = useState(null);
  const [scenarioYearDeliveries, setScenarioYearDeliveries] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadYears = async () => {
    try {
      const res = await axios.get(`${API}/vehicles/years`);
      const ysRaw = res.data?.years || [];
      const ys = ysRaw.map((v) => Number(v)).filter((v) => Number.isFinite(v));
      if (!ys.length) {
        setYears([]);
        setSelectedYear(null);
        return;
      }
      const current = Math.max(...ys);
      const prevCandidates = ys.filter((y) => y < current);
      const prev = prevCandidates.length ? Math.max(...prevCandidates) : null;
      const twoYears = prev !== null ? [current, prev] : [current];
      setYears(twoYears);
      setSelectedYear(current);
    } catch (e) {
      setError('Failed to load years');
    }
  };

  const loadSummary = async (year) => {
    if (!year) return;
    setLoading(true);
    try {
      const cur = await axios.get(`${API}/vehicles/summary/${year}`);
      setCurrentSummary(cur.data?.summary || null);
      const lastYear = years.length > 0 ? Math.max(...years.filter(y => y < year)) : null;
      if (lastYear) {
        const prev = await axios.get(`${API}/vehicles/summary/${lastYear}`);
        setPrevSummary(prev.data?.summary || null);
      } else {
        setPrevSummary(null);
      }
    } catch (e) {
      setError('Failed to load summary');
    } finally {
      setLoading(false);
    }
  };

  const loadScenarioInputs = async (scenario) => {
    try {
      const resp = await axios.post(`${API}/tesla/enhanced-model/${scenario}`);
      const model = resp.data?.model;
      const first = model?.income_statements?.[0];
      const drivers = first?.drivers || null;
      setScenarioInputs(drivers);
    } catch {
      setScenarioInputs(null);
    }
  };

  const loadScenarioYearDeliveries = async (scenario, year) => {
    try {
      const resp = await axios.get(`${API}/tesla/vehicle-analysis/${scenario}`);
      const trends = resp.data?.vehicle_analysis?.vehicle_trends || {};
      const yearData = trends[year] || {};
      const map = {};
      Object.keys(yearData).forEach((k) => {
        const v = yearData[k];
        if (v && typeof v === 'object' && 'deliveries' in v) {
          map[k] = v.deliveries;
        }
      });
      setScenarioYearDeliveries(map);
    } catch {
      setScenarioYearDeliveries({});
    }
  };

  useEffect(() => { loadYears(); }, []);

  useEffect(() => { 
    const loadMeta = async () => {
      try {
        const metaResp = await axios.get(`${API}/ai/tesla-agent/vehicle-models`);
        if (metaResp.data?.success) setModelMeta(metaResp.data.vehicle_models || {});
      } catch {}
    };
    if (!Object.keys(modelMeta).length) loadMeta();
  }, []);

  useEffect(() => { if (selectedYear) loadSummary(selectedYear); }, [selectedYear]);
  useEffect(() => { if (selectedYear) { loadSummary(selectedYear); loadScenarioInputs(selectedScenario); loadScenarioYearDeliveries(selectedScenario, selectedYear); } }, [selectedScenario]);

  const currentYear = selectedYear;
  const lastYear = years.length > 0 ? Math.max(...years.filter(y => y < currentYear)) : null;

  const getModelRow = (modelKey) => {
    const cur = currentSummary?.models?.find(m => m.model_key === modelKey);
    const prev = prevSummary?.models?.find(m => m.model_key === modelKey);
    // Prefer scenario-specific deliveries for the selected year if available
    const scenarioDeliv = scenarioYearDeliveries[modelKey];
    const deliveries = (scenarioDeliv !== undefined) ? scenarioDeliv : (cur ? cur.deliveries : 0);
    const baseForGrowth = prev ? prev.deliveries : null;
    const growth = (baseForGrowth && baseForGrowth > 0) ? (deliveries / baseForGrowth) - 1 : null;
    const name = (modelMeta[modelKey]?.name) || modelKey.replace('_',' ').replace(/\b\w/g, l=>l.toUpperCase());
    const segment = modelMeta[modelKey]?.segment || 'Vehicle';
    return { modelKey, name, segment, deliveries, growth };
  };

  const modelKeys = Array.from(new Set([
    ...(currentSummary?.models?.map(m => m.model_key) || []),
    ...(prevSummary?.models?.map(m => m.model_key) || []),
    ...Object.keys(scenarioYearDeliveries || {}),
  ]));

  const formatNumber = (value) => formatCompactNumber(value);
  const formatGrowth = (value) => (value === null || isNaN(value) ? 'N/A' : `${(value*100).toFixed(1)}%`);
  const pct = (v) => `${(Number(v || 0)*100).toFixed(1)}%`;

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-blue-600 to-emerald-600 text-white p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-2">New Vehicle Models Analysis</h2>
        <p className="text-blue-100">Excel-backed model view with scenario UI parity</p>
      </div>

      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Select Year</label>
              <select
                value={selectedYear || ''}
                onChange={(e) => setSelectedYear(parseInt(e.target.value))}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {years.map((y) => (
                  <option key={y} value={y}>{y}</option>
                ))}
              </select>
            </div>
            {currentYear && (
              <div className="text-sm text-gray-600">Current year: <span className="font-medium">{currentYear}</span>{lastYear ? <> · Last year: <span className="font-medium">{lastYear}</span></> : null}</div>
            )}
          </div>

          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-600">Models: <span className="font-medium">{modelKeys.length}</span></span>
            <select
              value={selectedScenario}
              onChange={(e) => setSelectedScenario(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="best">Best Case</option>
              <option value="base">Base Case</option>
              <option value="worst">Worst Case</option>
            </select>
          </div>
        </div>

        {loading ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-2 text-gray-600">Loading data...</p>
          </div>
        ) : currentSummary ? (
          <div className="space-y-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-blue-50 p-4 rounded border">
                <div className="text-sm text-blue-800">Total Sold Units (Deliveries)</div>
                <div className="text-2xl font-bold text-blue-700">{formatCompactNumber(currentSummary.totals.total_deliveries)}</div>
              </div>
              <div className="bg-green-50 p-4 rounded border">
                <div className="text-sm text-green-800">Total Produced Units</div>
                <div className="text-2xl font-bold text-green-700">{formatCompactNumber(currentSummary.totals.total_production)}</div>
              </div>

              <div className="bg-indigo-50 p-4 rounded border md:col-span-2">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold text-indigo-900">Scenario Inputs — {selectedScenario.charAt(0).toUpperCase()+selectedScenario.slice(1)}</h4>
                </div>
                {scenarioInputs ? (
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm text-indigo-900">
                    <div><span className="text-indigo-700">ASP Multiplier:</span> <span className="font-semibold">{Number(scenarioInputs.asp_multiplier||1).toFixed(2)}x</span></div>
                    <div><span className="text-indigo-700">Energy Growth:</span> <span className="font-semibold">{pct(scenarioInputs.energy_growth_rate)}</span></div>
                    <div><span className="text-indigo-700">Services Growth:</span> <span className="font-semibold">{pct(scenarioInputs.services_growth_rate)}</span></div>
                    <div><span className="text-indigo-700">Auto Margin +:</span> <span className="font-semibold">{pct(scenarioInputs.automotive_margin_improvement)}</span></div>
                    <div><span className="text-indigo-700">DSO Target:</span> <span className="font-semibold">{Number(scenarioInputs.dso_target||0).toFixed(1)} days</span></div>
                    <div><span className="text-indigo-700">DIO Target:</span> <span className="font-semibold">{Number(scenarioInputs.dio_target||0).toFixed(1)} days</span></div>
                    <div><span className="text-indigo-700">CapEx Rate:</span> <span className="font-semibold">{pct(scenarioInputs.capex_rate)}</span></div>
                    <div><span className="text-indigo-700">R&D Eff.:</span> <span className="font-semibold">{pct(scenarioInputs.rd_efficiency)}</span></div>
                    <div><span className="text-indigo-700">SG&A Eff.:</span> <span className="font-semibold">{pct(scenarioInputs.sga_efficiency)}</span></div>
                    <div><span className="text-indigo-700">Risk-free:</span> <span className="font-semibold">{pct(scenarioInputs.risk_free_rate)}</span></div>
                    <div><span className="text-indigo-700">Beta:</span> <span className="font-semibold">{Number(scenarioInputs.beta||0).toFixed(2)}</span></div>
                    <div><span className="text-indigo-700">MRP:</span> <span className="font-semibold">{pct(scenarioInputs.market_risk_premium)}</span></div>
                    <div><span className="text-indigo-700">Tax Rate:</span> <span className="font-semibold">{pct(scenarioInputs.tax_rate)}</span></div>
                    <div><span className="text-indigo-700">Cost of Debt:</span> <span className="font-semibold">{pct(scenarioInputs.cost_of_debt)}</span></div>
                  </div>
                ) : (
                  <div className="text-sm text-indigo-700">No inputs available.</div>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {modelKeys.map((mk) => {
                const row = getModelRow(mk);
                return (
                  <div key={mk} className="bg-gray-50 p-4 rounded-lg border">
                    <div className="flex justify-between items-start mb-3">
                      <h4 className="font-semibold text-gray-900">{row.name}</h4>
                      <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">{row.segment}</span>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">{currentYear} Deliveries:</span>
                        <span className="text-sm font-medium">{formatNumber(row.deliveries)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Growth vs {lastYear || (currentYear-1)}:</span>
                        <span className="text-sm font-medium text-gray-700">{formatGrowth(row.growth)}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="mt-10">
              <h3 className="text-lg font-semibold text-gray-900 mb-3">Interactive Tesla AI Agent</h3>
              <Suspense fallback={<div className="text-sm text-gray-600">Loading AI Agent…</div>}>
                <TeslaAIAgent year={selectedYear} initialScenario={selectedScenario} />
              </Suspense>
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-600">No data found for the selected year.</div>
        )}
      </div>
    </div>
  );
};

export default NewVehicleModels;
