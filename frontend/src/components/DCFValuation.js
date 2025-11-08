import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { formatCompactNumber } from '../lib/utils';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const DCFValuation = ({ scenario, model, generateModel, loading }) => {
  const [sensitivityData, setSensitivityData] = useState(null);
  const [loadingSensitivity, setLoadingSensitivity] = useState(false);

  // Debug logging
  useEffect(() => {
    console.log('DCFValuation component - Props received:', {
      scenario,
      hasModel: !!model,
      model: model,
      loading
    });
  }, [scenario, model, loading]);

  const formatCurrency = (value) => {
    if (!value) return '$0';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  const formatPercent = (value) => {
    if (!value) return '0.0%';
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatLargeNumber = (value) => {
    if (!value && value !== 0) return '$0';
    // Use formatCompactNumber which handles negative numbers and proper formatting
    // It automatically formats: <1M -> normal, >=1M -> M, >=1B -> B
    return `$${formatCompactNumber(value)}`;
  };

  const formatNumber = (value) => {
    if (!value) return '0';
    return new Intl.NumberFormat('en-US').format(Math.round(value));
  };

  const fetchSensitivityAnalysis = async () => {
    if (!model) return;
    
    try {
      setLoadingSensitivity(true);
      const response = await axios.get(`${API}/tesla/sensitivity/${scenario}`);
      setSensitivityData(response.data);
    } catch (error) {
      console.error('Failed to fetch sensitivity analysis:', error);
    } finally {
      setLoadingSensitivity(false);
    }
  };

  useEffect(() => {
    if (model && model.dcf_valuation) {
      fetchSensitivityAnalysis();
    }
  }, [model, scenario]);

  if (!model) {
    return (
      <div className="text-center py-12">
        <div className="text-gray-600 mb-4">No DCF valuation available for {scenario} scenario</div>
        <button
          onClick={() => generateModel(scenario)}
          disabled={loading}
          className="bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white px-6 py-3 rounded-lg font-medium transition-colors"
        >
          {loading ? 'Generating...' : 'Generate DCF Model'}
        </button>
      </div>
    );
  }

  const dcf = model.dcf_valuation;
  const years = [2025, 2026, 2027, 2028, 2029];

  return (
    <div className="space-y-8">
      {/* DCF Summary */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6 border">
        <h3 className="text-xl font-bold text-gray-900 mb-4">
          DCF Valuation Summary - {scenario.charAt(0).toUpperCase() + scenario.slice(1)} Case
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-600 mb-2">
              {formatCurrency(dcf.price_per_share)}
            </div>
            <div className="text-sm text-gray-600">Target Price per Share</div>
          </div>
          
          <div className="text-center">
            <div className="text-3xl font-bold text-green-600 mb-2">
              {formatLargeNumber(dcf.enterprise_value)}
            </div>
            <div className="text-sm text-gray-600">Enterprise Value</div>
          </div>
          
          <div className="text-center">
            <div className="text-3xl font-bold text-purple-600 mb-2">
              {formatPercent(dcf.wacc)}
            </div>
            <div className="text-sm text-gray-600">WACC</div>
          </div>
        </div>
      </div>

      {/* DCF Components */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Free Cash Flow Projections */}
        <div className="bg-white rounded-lg border p-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-4">Free Cash Flow Projections</h4>
          <div className="space-y-3">
            {dcf.projected_free_cash_flows?.map((fcf, idx) => (
              <div key={idx} className="flex justify-between items-center py-2 border-b border-gray-100">
                <span className="font-medium text-gray-700">{years[idx]}</span>
                <span className="font-semibold text-green-600">{formatLargeNumber(fcf)}</span>
              </div>
            ))}
          </div>
        </div>

        {/* WACC Breakdown */}
        <div className="bg-white rounded-lg border p-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-4">WACC Components</h4>
          <div className="space-y-3">
            <div className="flex justify-between items-center py-2 border-b border-gray-100">
              <span className="text-gray-700">Cost of Equity</span>
              <span className="font-semibold text-blue-600">{formatPercent(dcf.cost_of_equity)}</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-gray-100">
              <span className="text-gray-700">After-tax Cost of Debt</span>
              <span className="font-semibold text-red-600">{formatPercent(dcf.cost_of_debt)}</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-gray-100">
              <span className="text-gray-700">Terminal Growth Rate</span>
              <span className="font-semibold text-purple-600">{formatPercent(dcf.terminal_growth_rate)}</span>
            </div>
            <div className="flex justify-between items-center py-3 bg-gray-50 rounded">
              <span className="font-medium text-gray-900">Weighted Average Cost of Capital</span>
              <span className="font-bold text-lg text-purple-700">{formatPercent(dcf.wacc)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Valuation Bridge */}
      <div className="bg-white rounded-lg border p-6">
        <h4 className="text-lg font-semibold text-gray-900 mb-4">Valuation Bridge</h4>
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <div className="text-lg font-bold text-blue-600">
              {formatLargeNumber(dcf.present_value_cash_flows)}
            </div>
            <div className="text-sm text-gray-600 mt-1">PV of FCF<br/>(5 Years)</div>
          </div>
          
          <div className="flex items-center justify-center">
            <div className="text-2xl text-gray-400">+</div>
          </div>
          
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-lg font-bold text-green-600">
              {formatLargeNumber(dcf.present_value_terminal)}
            </div>
            <div className="text-sm text-gray-600 mt-1">PV of<br/>Terminal Value</div>
          </div>
          
          <div className="flex items-center justify-center">
            <div className="text-2xl text-gray-400">+</div>
          </div>
          
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <div className="text-lg font-bold text-purple-600">
              {formatLargeNumber(dcf.net_cash)}
            </div>
            <div className="text-sm text-gray-600 mt-1">Net Cash<br/>Position</div>
          </div>
        </div>
        
        <div className="mt-6 text-center">
          <div className="inline-block p-4 bg-yellow-50 border-2 border-yellow-200 rounded-lg">
            <div className="text-2xl font-bold text-yellow-700 mb-1">
              {formatLargeNumber(dcf.equity_value)}
            </div>
            <div className="text-sm text-gray-700">Total Equity Value</div>
          </div>
        </div>
      </div>

      {/* Sensitivity Analysis */}
      {sensitivityData && (
        <div className="bg-white rounded-lg border p-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-4">Sensitivity Analysis</h4>
          <p className="text-sm text-gray-600 mb-4">
            Price per share sensitivity to WACC and Terminal Growth Rate assumptions
          </p>
          
          {loadingSensitivity ? (
            <div className="text-center py-4">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-red-600 mx-auto"></div>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead>
                  <tr>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                      Terminal Growth →<br/>WACC ↓
                    </th>
                    {sensitivityData.sensitivity_analysis?.growth_rates?.map((rate, idx) => (
                      <th key={idx} className="px-3 py-2 text-center text-xs font-medium text-gray-500 uppercase">
                        {formatPercent(rate)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sensitivityData.sensitivity_analysis?.wacc_rates?.map((wacc, waccIdx) => (
                    <tr key={waccIdx} className={waccIdx % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                      <td className="px-3 py-2 font-medium text-gray-900 text-sm">
                        {formatPercent(wacc)}
                      </td>
                      {sensitivityData.sensitivity_analysis.price_matrix[waccIdx]?.map((price, priceIdx) => {
                        const isBaseCase = waccIdx === 2 && priceIdx === 2; // Middle values
                        return (
                          <td 
                            key={priceIdx} 
                            className={`px-3 py-2 text-center text-sm ${
                              isBaseCase 
                                ? 'bg-yellow-100 font-bold text-yellow-800' 
                                : 'text-gray-700'
                            }`}
                          >
                            {price > 0 ? formatCurrency(price) : 'N/A'}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Key Assumptions */}
      <div className="bg-white rounded-lg border p-6">
        <h4 className="text-lg font-semibold text-gray-900 mb-4">Key Valuation Assumptions</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-3 bg-gray-50 rounded">
            <div className="font-semibold text-gray-900">{formatPercent(dcf.terminal_growth_rate)}</div>
            <div className="text-xs text-gray-600">Terminal Growth</div>
          </div>
          <div className="text-center p-3 bg-gray-50 rounded">
            <div className="font-semibold text-gray-900">{formatPercent(dcf.wacc)}</div>
            <div className="text-xs text-gray-600">WACC</div>
          </div>
          <div className="text-center p-3 bg-gray-50 rounded">
            <div className="font-semibold text-gray-900">{formatNumber(dcf.shares_outstanding / 1000000)}M</div>
            <div className="text-xs text-gray-600">Shares Outstanding</div>
          </div>
          <div className="text-center p-3 bg-gray-50 rounded">
            <div className="font-semibold text-gray-900">{formatLargeNumber(dcf.projected_free_cash_flows?.[4])}</div>
            <div className="text-xs text-gray-600">2029 FCF</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DCFValuation;