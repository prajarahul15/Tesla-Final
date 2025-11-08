import React, { useState, useEffect } from 'react';
import axios from 'axios';
import FloatingChatButton from './FloatingChatButton';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const DataOverview = () => {
  const [years, setYears] = useState([]);
  const [selectedYear, setSelectedYear] = useState(null);
  const [summary, setSummary] = useState(null);
  const [econColumns, setEconColumns] = useState([]);
  const [vehicleMeta, setVehicleMeta] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchYears();
    fetchEconColumns();
    fetchVehicleMeta();
  }, []);

  useEffect(() => {
    if (selectedYear) fetchSummary(selectedYear);
  }, [selectedYear]);

  const fetchYears = async () => {
    try {
      const res = await axios.get(`${API}/vehicles/years`);
      const ys = res.data?.years || res.data?.data?.years || [];
      setYears(ys);
      if (ys && ys.length) {
        const current = Math.max(...ys);
        setSelectedYear(current);
      }
    } catch (err) {
      setError('Failed to fetch available years');
      setLoading(false);
      console.error('Years fetch error:', err);
    }
  };

  const fetchSummary = async (year) => {
    try {
      setLoading(true);
      const res = await axios.get(`${API}/vehicles/summary/${year}`);
      setSummary(res.data?.summary || res.data?.data?.summary || null);
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch vehicle summary');
      setLoading(false);
      console.error('Summary fetch error:', err);
    }
  };

  const fetchVehicleMeta = async () => {
    try {
      const resp = await axios.get(`${API}/ai/tesla-agent/vehicle-models`);
      if (resp.data?.success) {
        setVehicleMeta(resp.data.vehicle_models || {});
      }
    } catch (err) {
      setVehicleMeta({});
      console.warn('Vehicle meta fetch error:', err);
    }
  };

  const fetchEconColumns = async () => {
    try {
      const res = await axios.get(`${API}/vehicles/econ-columns`);
      const cols = res.data?.econ_columns || res.data?.data?.econ_columns || [];
      setEconColumns(cols);
    } catch (err) {
      // optional; keep page working without econ cols
      setEconColumns([]);
      console.warn('Econ columns fetch error:', err);
    }
  };

  const formatNumber = (v) => {
    if (v === null || v === undefined) return '0';
    try {
      return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(Number(v));
    } catch {
      return String(v);
    }
  };

  const formatUnits = (v) => {
    if (v === null || v === undefined) return '0';
    const num = Number(v);
    const hasDecimals = Math.abs(num % 1) > 1e-9;
    try {
      return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: hasDecimals ? 2 : 0,
        maximumFractionDigits: hasDecimals ? 2 : 0,
      }).format(num);
    } catch {
      return hasDecimals ? String(num.toFixed(2)) : String(Math.round(num));
    }
  };

  const formatCurrency = (v) => {
    try {
      return new Intl.NumberFormat('en-US', {
      style: 'currency',
        currency: 'USD',
        maximumFractionDigits: 0,
      }).format(Number(v || 0));
    } catch {
      return `$${formatNumber(v || 0)}`;
    }
  };

  // Format large currency values in billions for better readability
  const formatCurrencyBillions = (v) => {
    try {
      const value = Number(v || 0);
      const billions = value / 1000000000;
      if (billions >= 1) {
        return `$${billions.toFixed(1)}B`;
      } else if (value >= 1000000) {
        const millions = value / 1000000;
        return `$${millions.toFixed(0)}M`;
      } else if (value >= 1000) {
        const thousands = value / 1000;
        return `$${thousands.toFixed(0)}K`;
      } else {
        return `$${value.toFixed(0)}`;
      }
    } catch {
      return `$0.0B`;
    }
  };

  // Format numbers in billions for revenue, cost, profit
  const formatNumberBillions = (v) => {
    try {
      const value = Number(v || 0);
      const billions = value / 1000000000;
      if (billions >= 1) {
        return `${billions.toFixed(1)}B`;
      } else if (value >= 1000000) {
        const millions = value / 1000000;
        return `${millions.toFixed(0)}M`;
      } else if (value >= 1000) {
        const thousands = value / 1000;
        return `${thousands.toFixed(0)}K`;
      } else {
        return value.toFixed(0);
      }
    } catch {
      return '0.0B';
    }
  };

  // Derived: build per-model financials once summary and meta are present
  const buildModelRows = () => {
    const models = summary?.models || [];
    // Average ASP fallback for unknown keys
    const aspValues = Object.values(vehicleMeta || {})
      .map((m) => Number(m?.base_asp || 0))
      .filter((n) => Number.isFinite(n) && n > 0);
    const avgAsp = aspValues.length ? aspValues.reduce((a, b) => a + b, 0) / aspValues.length : 0;
    return models.map((m) => {
      const key = String(m.model_key);
      const meta = vehicleMeta[key] || {};
      const name = meta?.name || key.replace(/_/g, ' ');
      // Prefer Excel ASP if present in summary; else fallback to meta ASP/avg
      const asp = Number((m.asp !== undefined ? m.asp : (meta?.base_asp ?? avgAsp ?? 0)));
      const margin = Number(meta?.margin_premium ?? 0.19);
      const deliveries = Number(m.deliveries || 0);
      const production = Number(m.production || 0);
      const revenue = m.revenue !== undefined ? Number(m.revenue) : deliveries * asp;
      const cost = revenue * (1 - margin);
      const profit = revenue - cost;
      const profitability = revenue > 0 ? (profit / revenue) * 100 : 0;
      return { key, name, deliveries, production, asp, margin, revenue, cost, profit, profitability };
    });
  };

  const modelRows = buildModelRows();
  const totals = {
    sold_units: Number((summary?.totals?.total_sold ?? summary?.totals?.total_deliveries) || 0),
    produced_units: Number(summary?.totals?.total_production || 0),
    delivered_units: Number(summary?.totals?.total_deliveries || 0),
  };

  const totalsFinancial = (() => {
    const revenue = modelRows.reduce((s, r) => s + (r.revenue || 0), 0);
    const cost = modelRows.reduce((s, r) => s + (r.cost || 0), 0);
    const profit = revenue - cost;
    const profitPct = revenue > 0 ? (profit / revenue) * 100 : 0;
    return { revenue, cost, profit, profitPct };
  })();

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading vehicle overview…</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">{error}</div>
      </div>
    );
  }

  if (!summary) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-gray-600">No vehicle data available</div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Floating Chat Button */}
      <FloatingChatButton />
      
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6 rounded-lg shadow-lg">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold mb-2">Data Overview — Vehicle Deliveries & Production</h2>
            <p className="text-blue-100">Excel-backed totals and per-model breakdown</p>
          </div>
          <div>
            <label className="block text-xs text-white/80 mb-1">Year</label>
            <select
              value={selectedYear || ''}
              onChange={(e) => setSelectedYear(parseInt(e.target.value))}
              className="px-3 py-2 rounded bg-gray-100 border border-gray-300 text-gray-900"
            >
              {years.map((y) => (
                <option key={y} value={y}>{y}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-sm border-l-4 border-blue-500">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Total Deliveries</h3>
          <div className="text-3xl font-bold text-blue-600">{formatUnits(summary.totals?.total_deliveries)}</div>
          <p className="text-sm text-gray-600 mt-1">Units in {selectedYear}</p>
          </div>
        <div className="bg-white p-6 rounded-lg shadow-sm border-l-4 border-emerald-500">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Total Production</h3>
          <div className="text-3xl font-bold text-emerald-600">{formatUnits(summary.totals?.total_production)}</div>
          <p className="text-sm text-gray-600 mt-1">Units in {selectedYear}</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-sm border-l-4 border-purple-500">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Models</h3>
          <div className="text-3xl font-bold text-purple-600">{(summary.models || []).length}</div>
          <p className="text-sm text-gray-600 mt-1">Models with data</p>
          </div>
        <div className="bg-white p-6 rounded-lg shadow-sm border-l-4 border-orange-500">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Quantity Sold</h3>
          <div className="text-3xl font-bold text-orange-600">{formatUnits(totals.sold_units)}</div>
          <p className="text-sm text-gray-600 mt-1">Aggregated units in {selectedYear}</p>
        </div>
          </div>

      {/* Aggregated Financials */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-sm border-l-4 border-sky-500">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Aggregated Revenue</h3>
          <div className="text-3xl font-bold text-sky-600">{formatCurrencyBillions(summary.totals?.total_revenue ?? totalsFinancial.revenue)}</div>
          <p className="text-sm text-gray-600 mt-1">From Excel if available; else computed</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-sm border-l-4 border-rose-500">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Aggregated Cost</h3>
          <div className="text-3xl font-bold text-rose-600">{formatCurrencyBillions(totalsFinancial.cost)}</div>
          <p className="text-sm text-gray-600 mt-1">Revenue × (1 − margin)</p>
          </div>
        <div className="bg-white p-6 rounded-lg shadow-sm border-l-4 border-lime-500">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Aggregated Profit</h3>
          <div className="text-3xl font-bold text-lime-600">{formatCurrencyBillions(totalsFinancial.profit)}</div>
          <p className="text-sm text-gray-600 mt-1">Profitability: {totalsFinancial.profitPct.toFixed(1)}%</p>
        </div>
      </div>

      {/* Per-Model Breakdown with financials */}
      <div className="bg-white rounded-lg shadow-sm border">
        <div className="p-6 border-b">
          <h3 className="text-lg font-semibold text-gray-900">Per-Model Deliveries & Production — {selectedYear}</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Deliveries</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Production</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">ASP</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Revenue</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Cost</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Profit</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Profitability</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {modelRows.map((r, idx) => (
                <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{r.name}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">{formatUnits(r.deliveries)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">{formatUnits(r.production)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">{formatNumber(r.asp)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">{formatNumberBillions(r.revenue)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">{formatNumberBillions(r.cost)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">{formatNumberBillions(r.profit)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">{r.profitability.toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Economic Variables (if available) */}
      {econColumns && econColumns.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border">
          <div className="p-6 border-b">
            <h3 className="text-lg font-semibold text-gray-900">Economic Variables Available</h3>
            <p className="text-sm text-gray-600 mt-1">From multivariate configuration</p>
          </div>
          <div className="p-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {econColumns.map((name, idx) => (
                <div key={idx} className="bg-gray-50 p-3 rounded-lg text-center">
                  <div className="text-sm font-medium text-gray-900">{String(name).replace(/_/g, ' ')}</div>
                  <div className="text-xs text-gray-600 mt-1">Monthly series</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DataOverview;