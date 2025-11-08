import React from 'react';
import LoadingSpinner from './LoadingSpinner';
import { formatCompactNumber } from '../lib/utils';

const ScenarioTabs = ({ activeScenario, setActiveScenario, financialModels, generateModel, loading }) => {
  const scenarios = [
    { id: 'best', name: 'Best Case', color: 'green', description: 'Optimistic growth and margin assumptions' },
    { id: 'base', name: 'Base Case', color: 'blue', description: 'Most likely scenario based on current trends' },
    { id: 'worst', name: 'Worst Case', color: 'red', description: 'Conservative assumptions with headwinds' }
  ];

  const formatCurrency = (value) => {
    if (!value) return 'N/A';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };

  const formatPercent = (value) => {
    if (!value) return 'N/A';
    return `${(value * 100).toFixed(1)}%`;
  };

  return (
    <div>
      {/* Scenario Selection */}
      <div className="flex space-x-4 mb-8">
        {scenarios.map((scenario) => (
          <button
            key={scenario.id}
            onClick={() => setActiveScenario(scenario.id)}
            className={`flex-1 p-4 rounded-lg border-2 transition-all ${
              activeScenario === scenario.id
                ? `border-${scenario.color}-500 bg-${scenario.color}-50`
                : 'border-gray-200 hover:border-gray-300 bg-white'
            }`}
          >
            <div className="text-center">
              <h3 className={`font-semibold ${
                activeScenario === scenario.id ? `text-${scenario.color}-700` : 'text-gray-900'
              }`}>
                {scenario.name}
              </h3>
              <p className="text-sm text-gray-600 mt-1">{scenario.description}</p>
            </div>
          </button>
        ))}
      </div>

      {/* Current Scenario Details */}
      <div className="bg-gray-50 rounded-lg p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-900">
            {scenarios.find(s => s.id === activeScenario)?.name} Scenario
          </h3>
          {!financialModels[activeScenario] && (
            <button
              onClick={() => generateModel(activeScenario)}
              disabled={loading}
              className="bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white px-4 py-2 rounded font-medium transition-colors"
            >
              {loading ? 'Generating...' : 'Generate Model'}
            </button>
          )}
        </div>

        {loading && !financialModels[activeScenario] ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-red-600 mx-auto"></div>
            <p className="mt-2 text-gray-600">Generating financial model...</p>
          </div>
        ) : financialModels[activeScenario] ? (
          <div>
            {/* Key Metrics Summary */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
              <div className="bg-white p-4 rounded-lg border">
                <h4 className="font-medium text-gray-900 mb-2">2029 Revenue Projection</h4>
                <div className="text-2xl font-bold text-green-600">
                  ${formatCompactNumber(
                    financialModels[activeScenario].income_statements?.[4]?.total_revenue || 0
                  )}
                </div>
                <div className="text-sm text-gray-600 mt-1">
                  5-Year CAGR: {formatPercent(
                    financialModels[activeScenario].income_statements?.[4]?.total_revenue 
                      ? Math.pow(financialModels[activeScenario].income_statements[4].total_revenue / 96773000000, 1/5) - 1
                      : 0
                  )}
                </div>
              </div>

              <div className="bg-white p-4 rounded-lg border">
                <h4 className="font-medium text-gray-900 mb-2">DCF Valuation</h4>
                <div className="text-2xl font-bold text-blue-600">
                  {formatCurrency(financialModels[activeScenario].dcf_valuation?.price_per_share)}
                </div>
                <div className="text-sm text-gray-600 mt-1">
                  Per Share Price Target
                </div>
              </div>

              <div className="bg-white p-4 rounded-lg border">
                <h4 className="font-medium text-gray-900 mb-2">2029 Operating Margin</h4>
                <div className="text-2xl font-bold text-purple-600">
                  {formatPercent(
                    financialModels[activeScenario].income_statements?.[4]?.operating_margin
                  )}
                </div>
                <div className="text-sm text-gray-600 mt-1">
                  Target Operating Efficiency
                </div>
              </div>
            </div>

            {/* Financial Highlights */}
            <div className="bg-white rounded-lg border">
              <div className="p-4 border-b">
                <h4 className="font-semibold text-gray-900">5-Year Financial Highlights</h4>
              </div>
              <div className="p-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <div className="text-sm text-gray-600">Enterprise Value</div>
                    <div className="font-semibold">
                      ${(financialModels[activeScenario].dcf_valuation?.enterprise_value / 1000000000).toFixed(1)}B
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600">WACC</div>
                    <div className="font-semibold">
                      {formatPercent(financialModels[activeScenario].dcf_valuation?.wacc)}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600">Terminal Growth</div>
                    <div className="font-semibold">
                      {formatPercent(financialModels[activeScenario].dcf_valuation?.terminal_growth_rate)}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600">Final Year FCF</div>
                    <div className="font-semibold">
                      ${(financialModels[activeScenario].dcf_valuation?.projected_free_cash_flows?.[4] / 1000000000).toFixed(1)}B
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-600">
            Click "Generate Model" to create the {activeScenario} scenario financial model
          </div>
        )}
      </div>
    </div>
  );
};

export default ScenarioTabs;