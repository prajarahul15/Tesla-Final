import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ScenarioComparison = ({ models, generateAllScenarios, loading }) => {
  const [comparisonData, setComparisonData] = useState(null);
  const [loadingComparison, setLoadingComparison] = useState(false);

  const formatCurrency = (value) => {
    if (!value) return '$0';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };

  const formatPercent = (value) => {
    if (!value) return '0.0%';
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatLargeNumber = (value) => {
    if (!value) return '$0';
    
    const absValue = Math.abs(value);
    const sign = value < 0 ? '-' : '';
    
    if (absValue >= 1000000000) {
      return `${sign}$${(absValue / 1000000000).toFixed(1)}B`;
    } else if (absValue >= 1000000) {
      return `${sign}$${(absValue / 1000000).toFixed(1)}M`;
    }
    return formatCurrency(value);
  };

  const fetchComparisonData = async () => {
    try {
      setLoadingComparison(true);
      const response = await axios.get(`${API}/tesla/comparison`);
      setComparisonData(response.data);
    } catch (error) {
      console.error('Failed to fetch comparison data:', error);
    } finally {
      setLoadingComparison(false);
    }
  };

  useEffect(() => {
    if (Object.keys(models).length === 3) {
      fetchComparisonData();
    }
  }, [models]);

  const scenarios = ['best', 'base', 'worst'];
  const scenarioInfo = {
    best: { name: 'Best Case', color: 'green', bgColor: 'bg-green-50', textColor: 'text-green-700' },
    base: { name: 'Base Case', color: 'blue', bgColor: 'bg-blue-50', textColor: 'text-blue-700' },
    worst: { name: 'Worst Case', color: 'red', bgColor: 'bg-red-50', textColor: 'text-red-700' }
  };

  if (Object.keys(models).length === 0) {
    return (
      <div className="text-center py-12">
        <div className="text-gray-600 mb-4">No scenario models generated yet</div>
        <button
          onClick={generateAllScenarios}
          disabled={loading}
          className="bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white px-6 py-3 rounded-lg font-medium transition-colors"
        >
          {loading ? 'Generating All Scenarios...' : 'Generate All Scenarios'}
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h3 className="text-xl font-bold text-gray-900">Scenario Comparison</h3>
        {Object.keys(models).length < 3 && (
          <button
            onClick={generateAllScenarios}
            disabled={loading}
            className="bg-gray-800 hover:bg-gray-900 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg font-medium transition-colors"
          >
            {loading ? 'Generating...' : 'Generate Missing Scenarios'}
          </button>
        )}
      </div>

      {/* Valuation Comparison */}
      <div className="bg-white rounded-lg border p-6">
        <h4 className="text-lg font-semibold text-gray-900 mb-6">DCF Valuation Comparison</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {scenarios.map(scenario => {
            const model = models[scenario];
            const info = scenarioInfo[scenario];
            
            return (
              <div key={scenario} className={`${info.bgColor} rounded-lg p-6 border-2 border-${info.color}-200`}>
                <h5 className={`font-bold text-lg ${info.textColor} mb-4`}>{info.name}</h5>
                {model ? (
                  <div className="space-y-3">
                    <div className="text-center">
                      <div className={`text-2xl font-bold ${info.textColor}`}>
                        {formatCurrency(model.dcf_valuation?.price_per_share)}
                      </div>
                      <div className="text-sm text-gray-600">Price per Share</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-semibold text-gray-800">
                        {formatLargeNumber(model.dcf_valuation?.enterprise_value)}
                      </div>
                      <div className="text-sm text-gray-600">Enterprise Value</div>
                    </div>
                    <div className="text-center">
                      <div className="font-medium text-gray-800">
                        WACC: {formatPercent(model.dcf_valuation?.wacc)}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center text-gray-500">
                    <div className="text-sm">Model not generated</div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Revenue Growth Comparison */}
      <div className="bg-white rounded-lg border p-6">
        <h4 className="text-lg font-semibold text-gray-900 mb-6">Revenue Growth Comparison</h4>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Year</th>
                {scenarios.map(scenario => (
                  <th key={scenario} className={`px-6 py-3 text-center text-xs font-medium ${scenarioInfo[scenario].textColor} uppercase tracking-wider`}>
                    {scenarioInfo[scenario].name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {[2025, 2026, 2027, 2028, 2029].map((year, yearIdx) => (
                <tr key={year} className={yearIdx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{year}</td>
                  {scenarios.map(scenario => {
                    const model = models[scenario];
                    const revenue = model?.income_statements?.[yearIdx]?.total_revenue;
                    
                    return (
                      <td key={scenario} className="px-6 py-4 whitespace-nowrap text-sm text-center">
                        {revenue ? (
                          <div>
                            <div className="font-semibold">{formatLargeNumber(revenue)}</div>
                            {yearIdx > 0 && model?.income_statements?.[yearIdx-1] && (
                              <div className="text-xs text-gray-500">
                                {formatPercent((revenue / model.income_statements[yearIdx-1].total_revenue) - 1)} YoY
                              </div>
                            )}
                          </div>
                        ) : (
                          <span className="text-gray-400">N/A</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Margin Comparison */}
      <div className="bg-white rounded-lg border p-6">
        <h4 className="text-lg font-semibold text-gray-900 mb-6">Profitability Margins (2029)</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {['gross_margin', 'operating_margin', 'net_margin'].map(marginType => {
            const marginName = marginType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
            
            return (
              <div key={marginType} className="bg-gray-50 rounded-lg p-4">
                <h5 className="font-medium text-gray-900 text-center mb-4">{marginName}</h5>
                <div className="space-y-2">
                  {scenarios.map(scenario => {
                    const model = models[scenario];
                    const finalYear = model?.income_statements?.[4];
                    const margin = finalYear?.[marginType];
                    const info = scenarioInfo[scenario];
                    
                    return (
                      <div key={scenario} className="flex justify-between items-center">
                        <span className={`font-medium ${info.textColor}`}>{info.name}:</span>
                        <span className="font-bold">
                          {margin ? formatPercent(margin) : 'N/A'}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Free Cash Flow Comparison */}
      <div className="bg-white rounded-lg border p-6">
        <h4 className="text-lg font-semibold text-gray-900 mb-6">Free Cash Flow Comparison</h4>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Year</th>
                {scenarios.map(scenario => (
                  <th key={scenario} className={`px-6 py-3 text-center text-xs font-medium ${scenarioInfo[scenario].textColor} uppercase tracking-wider`}>
                    {scenarioInfo[scenario].name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {[2025, 2026, 2027, 2028, 2029].map((year, yearIdx) => (
                <tr key={year} className={yearIdx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{year}</td>
                  {scenarios.map(scenario => {
                    const model = models[scenario];
                    const cfStatement = model?.cash_flow_statements?.[yearIdx];
                    
                    // Calculate FCF = Operating Cash Flow - |Capital Expenditures|
                    let fcf = null;
                    if (cfStatement) {
                      const ocf = cfStatement.operating_cash_flow || 0;
                      const capex = cfStatement.capital_expenditures || 0;
                      fcf = ocf - Math.abs(capex);
                    }
                    
                    return (
                      <td key={scenario} className="px-6 py-4 whitespace-nowrap text-sm text-center">
                        <div className="font-semibold">
                          {fcf !== null ? formatLargeNumber(fcf) : 'N/A'}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Summary Insights */}
      {comparisonData && (
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border p-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-4">Key Insights</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h5 className="font-medium text-gray-900 mb-3">Valuation Range</h5>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-600">Upside Potential:</span>
                  <span className="font-semibold text-green-600">
                    {comparisonData.comparison_summary?.valuation_comparison?.best ? 
                      formatCurrency(comparisonData.comparison_summary.valuation_comparison.best.price_per_share) : 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Base Case:</span>
                  <span className="font-semibold text-blue-600">
                    {comparisonData.comparison_summary?.valuation_comparison?.base ? 
                      formatCurrency(comparisonData.comparison_summary.valuation_comparison.base.price_per_share) : 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Downside Risk:</span>
                  <span className="font-semibold text-red-600">
                    {comparisonData.comparison_summary?.valuation_comparison?.worst ? 
                      formatCurrency(comparisonData.comparison_summary.valuation_comparison.worst.price_per_share) : 'N/A'}
                  </span>
                </div>
              </div>
            </div>

            <div>
              <h5 className="font-medium text-gray-900 mb-3">Growth Scenarios</h5>
              <div className="space-y-2">
                {scenarios.map(scenario => {
                  const cagr = comparisonData.comparison_summary?.revenue_comparison?.[scenario]?.['5yr_cagr'];
                  const info = scenarioInfo[scenario];
                  
                  return (
                    <div key={scenario} className="flex justify-between">
                      <span className={info.textColor}>{info.name}:</span>
                      <span className="font-semibold">
                        {cagr ? formatPercent(cagr) : 'N/A'} CAGR
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}

      {loadingComparison && (
        <div className="text-center py-4">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-red-600 mx-auto"></div>
          <p className="mt-2 text-gray-600">Loading comparison data...</p>
        </div>
      )}
    </div>
  );
};

export default ScenarioComparison;