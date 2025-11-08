import React, { useState } from 'react';
import axios from 'axios';
import { formatCompactNumber } from '../lib/utils';
import { usePersistedSimulationState } from '../hooks/usePersistedSimulationState';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const CrossStatementSimulation = ({ scenario }) => {
  // Use persistent simulation state hook
  const {
    crossSimValues: simParams,
    setCrossSimValues: setSimParams,
    crossSimResults: simResults,
    setCrossSimResults: setSimResults,
    crossSimError: error,
    setCrossSimError: setError,
    crossAiInsights: aiInsights,
    setCrossAiInsights: setAiInsights,
    crossSimLoading: loading,
    setCrossSimLoading: setLoading,
    crossInsightLoading: insightLoading,
    setCrossInsightLoading: setInsightLoading
  } = usePersistedSimulationState(scenario, 'cross_statement');

  // State for expandable sections
  const [expandedSections, setExpandedSections] = useState({
    totalRevenue: false,
    totalCogs: false,
    operatingExpenses: false,
    operatingIncome: false,
    currentAssets: false,
    nonCurrentAssets: false,
    currentLiabilities: false,
    nonCurrentLiabilities: false,
    operatingActivities: false,
    investingActivities: false,
    financingActivities: false
  });

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const handleParamChange = (key, value) => {
    setSimParams(prev => ({ ...prev, [key]: value }));
  };

  const buildSimPayload = (includeAi = false) => {
    const payload = { include_ai_insights: includeAi };
    
    // Parameters that are percentages (need to be converted from % to decimal)
    const percentageParams = [
      'automotive_revenue_growth',
      'services_revenue_growth',
      'gross_margin_automotive',
      'gross_margin_services',
      'rd_as_percent_revenue',
      'sga_as_percent_revenue',
      'capex_as_percent_revenue',
      'tax_rate'
    ];
    
    // Parameters that are days (no conversion needed)
    const daysParams = [
      'days_sales_outstanding',
      'days_inventory_outstanding',
      'days_payable_outstanding'
    ];
    
    Object.entries(simParams).forEach(([k, v]) => {
      if (v !== '' && v !== null && v !== undefined) {
        const num = parseFloat(v);
        if (!Number.isNaN(num)) {
          // Convert percentage to decimal for percentage parameters
          if (percentageParams.includes(k)) {
            payload[k] = num / 100;
          } else {
            payload[k] = num;
          }
        }
      }
    });
    return payload;
  };

  const simulateStatements = async () => {
    try {
      setLoading(true);
      setError(null);
      setSimResults(null);
      setAiInsights(null);

      const payload = buildSimPayload(false);
      const response = await axios.post(
        `${API}/tesla/model/${scenario}/simulate-all-statements`, 
        payload, 
        { timeout: 30000 }
      );
      
      setSimResults(response.data);
      setLoading(false);
    } catch (err) {
      let msg = 'Simulation failed';
      const detail = err?.response?.data?.detail;
      if (typeof detail === 'string') {
        msg = detail;
      } else if (Array.isArray(detail)) {
        msg = detail.map(d => d?.msg || JSON.stringify(d)).join('; ');
      } else if (detail && typeof detail === 'object') {
        msg = JSON.stringify(detail);
      } else if (err?.message) {
        msg = err.message;
      }
      setError(msg);
      setLoading(false);
    }
  };

  const generateAiInsights = async () => {
    try {
      setInsightLoading(true);
      setError(null);
      setAiInsights(null);

      const payload = buildSimPayload(true);
      const response = await axios.post(
        `${API}/tesla/model/${scenario}/simulate-all-statements`, 
        payload, 
        { timeout: 30000 }
      );
      
      setAiInsights(response.data?.ai_insights || null);
      setInsightLoading(false);
    } catch (err) {
      let msg = 'Insight generation failed';
      const detail = err?.response?.data?.detail;
      if (typeof detail === 'string') msg = detail;
      else if (Array.isArray(detail)) msg = detail.map(d => d?.msg || JSON.stringify(d)).join('; ');
      else if (detail && typeof detail === 'object') msg = JSON.stringify(detail);
      else if (err?.message) msg = err.message;
      setError(msg);
      setInsightLoading(false);
    }
  };

  const resetSimulation = () => {
    setSimParams({
      automotive_revenue_growth: '',
      services_revenue_growth: '',
      gross_margin_automotive: '',
      gross_margin_services: '',
      rd_as_percent_revenue: '',
      sga_as_percent_revenue: '',
      days_sales_outstanding: '',
      days_inventory_outstanding: '',
      days_payable_outstanding: '',
      capex_as_percent_revenue: '',
      tax_rate: ''
    });
    setSimResults(null);
    setError(null);
    setAiInsights(null);
  };

  const sliderConfig = [
    // Row 1
    { key: 'automotive_revenue_growth', label: 'Automotive Revenue Growth', unit: '%', min: 0, max: 50, step: 1, default: 22, category: 'Revenue Drivers' },
    { key: 'services_revenue_growth', label: 'Services Revenue Growth', unit: '%', min: 0, max: 50, step: 1, default: 30, category: 'Revenue Drivers' },
    { key: 'gross_margin_automotive', label: 'Automotive Gross Margin', unit: '%', min: 0, max: 40, step: 0.5, default: 18, category: 'Margin Drivers' },
    // Row 2
    { key: 'gross_margin_services', label: 'Services Gross Margin', unit: '%', min: 0, max: 50, step: 0.5, default: 25, category: 'Margin Drivers' },
    { key: 'rd_as_percent_revenue', label: 'R&D as % of Revenue', unit: '%', min: 0, max: 10, step: 0.1, default: 3.5, category: 'OpEx Drivers' },
    { key: 'sga_as_percent_revenue', label: 'SG&A as % of Revenue', unit: '%', min: 0, max: 10, step: 0.1, default: 4.2, category: 'OpEx Drivers' },
    // Row 3
    { key: 'days_sales_outstanding', label: 'DSO', unit: 'days', min: 5, max: 60, step: 1, default: 15, category: 'Working Capital' },
    { key: 'days_inventory_outstanding', label: 'DIO', unit: 'days', min: 10, max: 90, step: 1, default: 56, category: 'Working Capital' },
    { key: 'days_payable_outstanding', label: 'DPO', unit: 'days', min: 10, max: 90, step: 1, default: 60, category: 'Working Capital' },
    // Row 4
    { key: 'capex_as_percent_revenue', label: 'CapEx % Rev', unit: '%', min: 0, max: 20, step: 0.5, default: 8, category: 'Investment & Tax' },
    { key: 'tax_rate', label: 'Tax Rate', unit: '%', min: 0, max: 40, step: 0.5, default: 21, category: 'Investment & Tax' }
  ];

  const renderSlider = (config) => {
    const value = simParams[config.key] !== '' ? parseFloat(simParams[config.key]) : config.default;
    
    return (
      <div key={config.key} className="space-y-2">
        <div className="flex justify-between items-center">
          <label className="text-sm font-medium text-gray-700">
            {config.label}
          </label>
          <div className="flex items-center space-x-2">
            <input
              type="number"
              value={value}
              onChange={(e) => handleParamChange(config.key, e.target.value)}
              className="w-16 px-2 py-1 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              step={config.step}
              min={config.min}
              max={config.max}
            />
            <span className="text-xs text-gray-500 w-8">{config.unit}</span>
          </div>
        </div>
        <input
          type="range"
          min={config.min}
          max={config.max}
          step={config.step}
          value={value}
          onChange={(e) => handleParamChange(config.key, e.target.value)}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
        />
        <div className="flex justify-between text-xs text-gray-500">
          <span>{config.min}{config.unit}</span>
          <span>{config.max}{config.unit}</span>
        </div>
      </div>
    );
  };

  const renderParameterControls = () => (
    <div className="bg-white p-6 rounded-lg shadow-sm border mb-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Cross-Statement Simulation Parameters</h3>
      
      {/* 3 Columns × 4 Rows Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {sliderConfig.map(config => renderSlider(config))}
      </div>

      {/* Action Buttons */}
      <div className="flex space-x-4 mt-6 pt-4 border-t">
        <button
          onClick={simulateStatements}
          disabled={loading}
          className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Simulating...' : 'Simulate All Statements'}
        </button>
        <button
          onClick={resetSimulation}
          className="bg-gray-500 text-white px-6 py-2 rounded-lg hover:bg-gray-600"
        >
          Reset
        </button>
      </div>

      {error && (
        <div className="mt-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}
    </div>
  );

  const renderKeyMetrics = () => {
    if (!simResults?.key_metrics) return null;

    const metrics = simResults.key_metrics;
    
    return (
      <div className="bg-white p-6 rounded-lg shadow-sm border mb-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Key Performance Metrics</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
            <h4 className="font-semibold text-blue-900 mb-1">Revenue CAGR</h4>
            <div className="text-2xl font-bold text-blue-600">{metrics.revenue_cagr}%</div>
            <p className="text-sm text-blue-700">5-year growth rate</p>
          </div>
          <div className="bg-green-50 p-4 rounded-lg border border-green-200">
            <h4 className="font-semibold text-green-900 mb-1">FCF Margin</h4>
            <div className="text-2xl font-bold text-green-600">{metrics.free_cash_flow_margin}%</div>
            <p className="text-sm text-green-700">Average margin</p>
          </div>
          <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
            <h4 className="font-semibold text-orange-900 mb-1">Working Capital</h4>
            <div className="text-2xl font-bold text-orange-600">{metrics.working_capital_days}</div>
            <p className="text-sm text-orange-700">Cash cycle (days)</p>
          </div>
          <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
            <h4 className="font-semibold text-purple-900 mb-1">ROIC</h4>
            <div className="text-2xl font-bold text-purple-600">{metrics.roic}%</div>
            <p className="text-sm text-purple-700">Return on invested capital</p>
          </div>
        </div>
      </div>
    );
  };

  const renderUpdatedStatements = () => {
    if (!simResults) return null;

    return (
      <div className="space-y-6">
        {/* Income Statement */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Updated Income Statement — All Years</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Line Item</th>
                  {[2025, 2026, 2027, 2028, 2029].map(year => (
                    <th key={year} className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">{year}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                <tr className="bg-blue-50 cursor-pointer hover:bg-blue-100" onClick={() => toggleSection('totalRevenue')}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 flex items-center">
                    <span className="mr-2">{expandedSections.totalRevenue ? '▼' : '►'}</span>
                    Total Revenue
                  </td>
                  {simResults.updated_statements.income_statements.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-semibold text-blue-600">
                      {formatCompactNumber(stmt.total_revenue)}
                    </td>
                  ))}
                </tr>
                {expandedSections.totalRevenue && (
                  <>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Automotive Revenue</td>
                      {simResults.updated_statements.income_statements.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.automotive_revenue)}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Services Revenue</td>
                      {simResults.updated_statements.income_statements.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.services_revenue)}
                        </td>
                      ))}
                    </tr>
                  </>
                )}
                <tr className="bg-red-50 cursor-pointer hover:bg-red-100" onClick={() => toggleSection('totalCogs')}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 flex items-center">
                    <span className="mr-2">{expandedSections.totalCogs ? '▼' : '►'}</span>
                    Total COGS
                  </td>
                  {simResults.updated_statements.income_statements.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-red-600">
                      ({formatCompactNumber(stmt.total_cogs)})
                    </td>
                  ))}
                </tr>
                <tr className="bg-green-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Gross Profit</td>
                  {simResults.updated_statements.income_statements.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-semibold text-green-600">
                      {formatCompactNumber(stmt.total_gross_profit)}
                    </td>
                  ))}
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">Gross Margin</td>
                  {simResults.updated_statements.income_statements.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                      {(stmt.gross_margin * 100).toFixed(1)}%
                    </td>
                  ))}
                </tr>
                <tr className="bg-gray-50 cursor-pointer hover:bg-gray-200" onClick={() => toggleSection('operatingExpenses')}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 flex items-center">
                    <span className="mr-2">{expandedSections.operatingExpenses ? '▼' : '►'}</span>
                    Operating Expenses
                  </td>
                  {simResults.updated_statements.income_statements.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-600">
                      ({formatCompactNumber(stmt.total_operating_expenses)})
                    </td>
                  ))}
                </tr>
                {expandedSections.operatingExpenses && (
                  <>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• R&D</td>
                      {simResults.updated_statements.income_statements.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          ({formatCompactNumber(stmt.research_development)})
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• SG&A</td>
                      {simResults.updated_statements.income_statements.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          ({formatCompactNumber(stmt.selling_general_admin)})
                        </td>
                      ))}
                    </tr>
                  </>
                )}
                <tr className="bg-purple-50 cursor-pointer hover:bg-purple-100" onClick={() => toggleSection('operatingIncome')}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 flex items-center">
                    <span className="mr-2">{expandedSections.operatingIncome ? '▼' : '►'}</span>
                    Operating Income
                  </td>
                  {simResults.updated_statements.income_statements.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-semibold text-purple-600">
                      {formatCompactNumber(stmt.operating_income)}
                    </td>
                  ))}
                </tr>
                {expandedSections.operatingIncome && (
                  <>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Operating Margin</td>
                      {simResults.updated_statements.income_statements.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {(stmt.operating_margin * 100).toFixed(1)}%
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Interest Income</td>
                      {simResults.updated_statements.income_statements.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.interest_income)}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Interest Expense</td>
                      {simResults.updated_statements.income_statements.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          ({formatCompactNumber(stmt.interest_expense)})
                        </td>
                      ))}
                    </tr>
                  </>
                )}
                <tr className="bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Pre-Tax Income</td>
                  {simResults.updated_statements.income_statements.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-semibold text-gray-900">
                      {formatCompactNumber(stmt.pretax_income)}
                    </td>
                  ))}
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">Income Tax Expense</td>
                  {simResults.updated_statements.income_statements.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                      ({formatCompactNumber(stmt.income_tax_expense)})
                    </td>
                  ))}
                </tr>
                <tr className="bg-yellow-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Net Income</td>
                  {simResults.updated_statements.income_statements.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-bold text-yellow-700">
                      {formatCompactNumber(stmt.net_income)}
                    </td>
                  ))}
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">Net Margin</td>
                  {simResults.updated_statements.income_statements.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                      {(stmt.net_margin * 100).toFixed(1)}%
                    </td>
                  ))}
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">EPS</td>
                  {simResults.updated_statements.income_statements.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                      ${stmt.earnings_per_share.toFixed(2)}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Balance Sheet */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Updated Balance Sheet — All Years</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Line Item</th>
                  {[2025, 2026, 2027, 2028, 2029].map(year => (
                    <th key={year} className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">{year}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                <tr className="bg-blue-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">ASSETS</td>
                  <td colSpan="5" className="px-6 py-4"></td>
                </tr>
                <tr className="cursor-pointer hover:bg-gray-100" onClick={() => toggleSection('currentAssets')}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 flex items-center">
                    <span className="mr-2">{expandedSections.currentAssets ? '▼' : '►'}</span>
                    Current Assets
                  </td>
                  {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-semibold text-gray-900">
                      {formatCompactNumber(stmt.total_current_assets)}
                    </td>
                  ))}
                </tr>
                {expandedSections.currentAssets && (
                  <>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Cash & Equivalents</td>
                      {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.cash_and_equivalents)}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Accounts Receivable</td>
                      {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.accounts_receivable)}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Inventory</td>
                      {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.inventory)}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Prepaid Expenses</td>
                      {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.prepaid_expenses)}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Other Current Assets</td>
                      {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.other_current_assets)}
                        </td>
                      ))}
                    </tr>
                  </>
                )}
                <tr className="cursor-pointer hover:bg-gray-100" onClick={() => toggleSection('nonCurrentAssets')}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 flex items-center">
                    <span className="mr-2">{expandedSections.nonCurrentAssets ? '▼' : '►'}</span>
                    Non-Current Assets
                  </td>
                  {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-semibold text-gray-900">
                      {formatCompactNumber(stmt.total_non_current_assets)}
                    </td>
                  ))}
                </tr>
                {expandedSections.nonCurrentAssets && (
                  <>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Net PP&E</td>
                      {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.net_ppe)}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Intangible Assets</td>
                      {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.intangible_assets)}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Other Non-Current Assets</td>
                      {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.other_non_current_assets)}
                        </td>
                      ))}
                    </tr>
                  </>
                )}
                <tr className="bg-green-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">Total Assets</td>
                  {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-bold text-green-600">
                      {formatCompactNumber(stmt.total_assets)}
                    </td>
                  ))}
                </tr>
                <tr className="bg-red-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">LIABILITIES</td>
                  <td colSpan="5" className="px-6 py-4"></td>
                </tr>
                <tr className="cursor-pointer hover:bg-gray-100" onClick={() => toggleSection('currentLiabilities')}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 flex items-center">
                    <span className="mr-2">{expandedSections.currentLiabilities ? '▼' : '►'}</span>
                    Current Liabilities
                  </td>
                  {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-semibold text-gray-900">
                      {formatCompactNumber(stmt.total_current_liabilities)}
                    </td>
                  ))}
                </tr>
                {expandedSections.currentLiabilities && (
                  <>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Accounts Payable</td>
                      {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.accounts_payable)}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Accrued Liabilities</td>
                      {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.accrued_liabilities)}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Current Portion of Debt</td>
                      {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.current_portion_debt)}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Other Current Liabilities</td>
                      {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.other_current_liabilities)}
                        </td>
                      ))}
                    </tr>
                  </>
                )}
                <tr className="cursor-pointer hover:bg-gray-100" onClick={() => toggleSection('nonCurrentLiabilities')}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 flex items-center">
                    <span className="mr-2">{expandedSections.nonCurrentLiabilities ? '▼' : '►'}</span>
                    Non-Current Liabilities
                  </td>
                  {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-semibold text-gray-900">
                      {formatCompactNumber(stmt.total_non_current_liabilities)}
                    </td>
                  ))}
                </tr>
                {expandedSections.nonCurrentLiabilities && (
                  <>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Long-Term Debt</td>
                      {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.long_term_debt)}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Other Non-Current Liabilities</td>
                      {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.other_non_current_liabilities)}
                        </td>
                      ))}
                    </tr>
                  </>
                )}
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">Total Liabilities</td>
                  {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-bold text-gray-900">
                      {formatCompactNumber(stmt.total_liabilities)}
                    </td>
                  ))}
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Shareholders' Equity</td>
                  {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-semibold">
                    </td>
                  ))}
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Retained Earnings</td>
                  {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                      {formatCompactNumber(stmt.retained_earnings)}
                    </td>
                  ))}
                </tr>
                <tr className="bg-purple-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">Total Shareholders' Equity</td>
                  {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-bold text-purple-600">
                      {formatCompactNumber(stmt.total_shareholders_equity)}
                    </td>
                  ))}
                </tr>
                <tr className="bg-green-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">Total Liabilities & Equity</td>
                  {simResults.updated_statements.balance_sheets.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-bold text-green-600">
                      {formatCompactNumber(stmt.total_liab_and_equity)}
                    </td>
                  ))}
                </tr>
                <tr className="bg-yellow-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">Balance Difference</td>
                  {simResults.updated_statements.balance_sheets.map((stmt, idx) => {
                    const diff = stmt.total_assets - stmt.total_liab_and_equity;
                    return (
                      <td key={idx} className={`px-6 py-4 whitespace-nowrap text-sm text-right font-bold ${Math.abs(diff) < 1000 ? 'text-green-600' : 'text-red-600'}`}>
                        {formatCompactNumber(diff)}
                      </td>
                    );
                  })}
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Cash Flow Statement */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Updated Cash Flow Statement — All Years</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Line Item</th>
                  {[2025, 2026, 2027, 2028, 2029].map(year => (
                    <th key={year} className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">{year}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                <tr className="bg-blue-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">OPERATING ACTIVITIES</td>
                  <td colSpan="5" className="px-6 py-4"></td>
                </tr>
                <tr className="bg-green-50 cursor-pointer hover:bg-green-100" onClick={() => toggleSection('operatingActivities')}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900 flex items-center">
                    <span className="mr-2">{expandedSections.operatingActivities ? '▼' : '►'}</span>
                    Operating Cash Flow
                  </td>
                  {simResults.updated_statements.cash_flow_statements.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-bold text-green-600">
                      {formatCompactNumber(stmt.operating_cash_flow)}
                    </td>
                  ))}
                </tr>
                {expandedSections.operatingActivities && (
                  <>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Net Income</td>
                      {simResults.updated_statements.cash_flow_statements.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.net_income)}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Depreciation & Amortization</td>
                      {simResults.updated_statements.cash_flow_statements.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.depreciation_amortization)}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Accounts Receivable</td>
                      {simResults.updated_statements.cash_flow_statements.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.change_accounts_receivable)}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Inventory</td>
                      {simResults.updated_statements.cash_flow_statements.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.change_inventory)}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Accounts Payable</td>
                      {simResults.updated_statements.cash_flow_statements.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.change_accounts_payable)}
                        </td>
                      ))}
                    </tr>
                  </>
                )}
                <tr className="bg-purple-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">INVESTING ACTIVITIES</td>
                  <td colSpan="5" className="px-6 py-4"></td>
                </tr>
                <tr className="bg-purple-50 cursor-pointer hover:bg-purple-100" onClick={() => toggleSection('investingActivities')}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900 flex items-center">
                    <span className="mr-2">{expandedSections.investingActivities ? '▼' : '►'}</span>
                    Investing Cash Flow
                  </td>
                  {simResults.updated_statements.cash_flow_statements.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-bold text-purple-600">
                      {formatCompactNumber(stmt.investing_cash_flow)}
                    </td>
                  ))}
                </tr>
                {expandedSections.investingActivities && (
                  <>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Capital Expenditures</td>
                      {simResults.updated_statements.cash_flow_statements.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          ({formatCompactNumber(Math.abs(stmt.capital_expenditures))})
                        </td>
                      ))}
                    </tr>
                  </>
                )}
                <tr className="bg-yellow-50 border-t-2 border-yellow-400">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">FREE CASH FLOW (OCF - CapEx)</td>
                  {simResults.updated_statements.cash_flow_statements.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-bold text-yellow-700">
                      {stmt.free_cash_flow !== undefined && stmt.free_cash_flow !== null 
                        ? formatCompactNumber(stmt.free_cash_flow)
                        : formatCompactNumber(stmt.operating_cash_flow - Math.abs(stmt.capital_expenditures))}
                    </td>
                  ))}
                </tr>
                <tr className="bg-red-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">FINANCING ACTIVITIES</td>
                  <td colSpan="5" className="px-6 py-4"></td>
                </tr>
                <tr className="bg-red-50 cursor-pointer hover:bg-red-100" onClick={() => toggleSection('financingActivities')}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900 flex items-center">
                    <span className="mr-2">{expandedSections.financingActivities ? '▼' : '►'}</span>
                    Financing Cash Flow
                  </td>
                  {simResults.updated_statements.cash_flow_statements.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-bold text-red-600">
                      {formatCompactNumber(stmt.financing_cash_flow)}
                    </td>
                  ))}
                </tr>
                {expandedSections.financingActivities && (
                  <>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Debt Proceeds</td>
                      {simResults.updated_statements.cash_flow_statements.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          {formatCompactNumber(stmt.debt_proceeds)}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Debt Repayments</td>
                      {simResults.updated_statements.cash_flow_statements.map((stmt, idx) => (
                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                          ({formatCompactNumber(Math.abs(stmt.debt_repayments))})
                        </td>
                      ))}
                    </tr>
                  </>
                )}
                <tr className="bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">Net Change in Cash</td>
                  {simResults.updated_statements.cash_flow_statements.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-semibold text-gray-900">
                      {formatCompactNumber(stmt.net_change_cash)}
                    </td>
                  ))}
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">Ending Cash Balance</td>
                  {simResults.updated_statements.cash_flow_statements.map((stmt, idx) => (
                    <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                      {formatCompactNumber(stmt.ending_cash)}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* AI Insights Button */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">AI Insights</h3>
            <button
              onClick={generateAiInsights}
              disabled={insightLoading}
              className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {insightLoading ? 'Generating...' : 'Generate AI Insights'}
            </button>
          </div>
          
          {aiInsights && (
            <div className="space-y-4">
              {/* Executive Summary */}
              <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                <h4 className="font-semibold text-blue-900 mb-2">Executive Summary</h4>
                <p className="text-blue-800">{aiInsights.executive_summary}</p>
              </div>

              {/* Financial Performance */}
              <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                <h4 className="font-semibold text-green-900 mb-2">Financial Performance</h4>
                <div className="space-y-2 text-green-800">
                  <p><strong>Revenue Impact:</strong> {aiInsights.financial_performance?.revenue_impact}</p>
                  <p><strong>Profitability:</strong> {aiInsights.financial_performance?.profitability_impact}</p>
                  <p><strong>Cash Generation:</strong> {aiInsights.financial_performance?.cash_generation}</p>
                </div>
              </div>

              {/* Balance Sheet Insights */}
              <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                <h4 className="font-semibold text-orange-900 mb-2">Balance Sheet Insights</h4>
                <div className="space-y-2 text-orange-800">
                  <p><strong>Asset Efficiency:</strong> {aiInsights.balance_sheet_insights?.asset_efficiency}</p>
                  <p><strong>Working Capital:</strong> {aiInsights.balance_sheet_insights?.working_capital}</p>
                  <p><strong>Financial Position:</strong> {aiInsights.balance_sheet_insights?.financial_position}</p>
                </div>
              </div>

              {/* Integrated Metrics */}
              <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                <h4 className="font-semibold text-purple-900 mb-2">Integrated Metrics</h4>
                <div className="space-y-2 text-purple-800">
                  <p><strong>Cash Conversion:</strong> {aiInsights.integrated_metrics?.cash_conversion}</p>
                  <p><strong>Capital Efficiency:</strong> {aiInsights.integrated_metrics?.capital_efficiency}</p>
                  <p><strong>Growth Sustainability:</strong> {aiInsights.integrated_metrics?.growth_sustainability}</p>
                </div>
              </div>

              {/* Key Risks & Recommendations */}
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-red-50 p-4 rounded-lg border border-red-200">
                  <h4 className="font-semibold text-red-900 mb-2">Key Risks</h4>
                  <ul className="list-disc list-inside text-red-800 space-y-1">
                    {aiInsights.key_risks?.map((risk, idx) => (
                      <li key={idx}>{risk}</li>
                    ))}
                  </ul>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                  <h4 className="font-semibold text-gray-900 mb-2">Strategic Recommendations</h4>
                  <ul className="list-disc list-inside text-gray-800 space-y-1">
                    {aiInsights.strategic_recommendations?.map((rec, idx) => (
                      <li key={idx}>{rec}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-purple-600 to-blue-600 text-white p-6 rounded-lg shadow-lg">
        <div className="flex justify-between items-start">
          <div>
            <h2 className="text-2xl font-bold mb-2">Cross-Statement Financial Simulation</h2>
            <p className="text-purple-100">
              Comprehensive financial modeling across Income Statement, Balance Sheet, and Cash Flow with AI-powered insights
            </p>
          </div>
          {/* Persistence Indicator */}
          {(simResults || aiInsights || Object.values(simParams).some(v => v !== '')) && (
            <div className="flex items-center space-x-2 bg-white/20 px-3 py-1 rounded-full">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              <span className="text-xs text-white/90">Data Saved</span>
            </div>
          )}
        </div>
      </div>

      {renderParameterControls()}
      {renderKeyMetrics()}
      {renderUpdatedStatements()}
    </div>
  );
};

export default CrossStatementSimulation;
