import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Suspense, lazy } from 'react';
import { formatCompactNumber } from '../lib/utils';
const TeslaAIAgent = lazy(() => import('./TeslaAIAgent'));

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// PHASE 1: Vehicle Model Analysis Component
export const VehicleModelAnalysis = () => {
  const [vehicleData, setVehicleData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedScenario, setSelectedScenario] = useState('base');
  const [modelMeta, setModelMeta] = useState({});

  const fetchVehicleAnalysis = async (scenario) => {
    try {
      setLoading(true);
      // Fetch scenario-specific vehicle analysis
      const response = await axios.get(`${API}/tesla/vehicle-analysis/${scenario}`);

      if (response.data && response.data.success) {
        const analysis = response.data.vehicle_analysis;
        const trends2024 = analysis.vehicle_trends[2024] || {};

        // Map into display structure
        const vehicle_models = Object.entries(trends2024).reduce((acc, [key, val]) => {
          acc[key] = {
            name: (key || '').replace('_', ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
            deliveries_2024: val.deliveries || 0,
            segment: modelMeta[key]?.segment || 'Vehicle',
          };
          return acc;
        }, {});

        setVehicleData({ scenario, vehicle_models });
      }
      setLoading(false);
    } catch (error) {
      console.error('Error fetching vehicle analysis:', error);
      setLoading(false);
    }
  };

  useEffect(() => {
    // Load meta (names/segments) once to enrich cards (optional)
    const loadMeta = async () => {
      try {
        const metaResp = await axios.get(`${API}/ai/tesla-agent/vehicle-models`);
        if (metaResp.data?.success) {
          setModelMeta(metaResp.data.vehicle_models || {});
        }
      } catch {}
    };
    if (!Object.keys(modelMeta).length) loadMeta();
  }, []);

  useEffect(() => {
    fetchVehicleAnalysis(selectedScenario);
  }, [selectedScenario]);

  const formatNumber = (value) => formatCompactNumber(value);

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-blue-600 to-green-600 text-white p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-2">Vehicle Model Analysis</h2>
        <p className="text-blue-100">PHASE 1: Driver-based forecasting with vehicle model granularity</p>
      </div>

      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Scenario Selection</h3>
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

        {loading ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-2 text-gray-600">Loading vehicle analysis...</p>
          </div>
        ) : vehicleData ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {Object.entries(vehicleData.vehicle_models).map(([key, model]) => (
              <div key={key} className="bg-gray-50 p-4 rounded-lg border">
                <div className="flex justify-between items-start mb-3">
                  <h4 className="font-semibold text-gray-900">{model.name}</h4>
                  <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
                    {model.segment}
                  </span>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">2024 Deliveries:</span>
                    <span className="text-sm font-medium">{formatNumber(model.deliveries_2024)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Growth vs 2023:</span>
                    <span className="text-sm font-medium text-gray-500">N/A</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-600">
            No data available
          </div>
        )}

        {/* Embedded Tesla AI Agent synced to vehicle scenario */}
        <div className="mt-10">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">Interactive Tesla AI Agent</h3>
          <p className="text-sm text-gray-600 mb-4">Adjust ASP, cost and deliveries to see real-time impact aligned with the selected scenario.</p>
          <Suspense fallback={<div className="text-sm text-gray-600">Loading AI Agent…</div>}>
            <TeslaAIAgent />
          </Suspense>
        </div>

        <div className="mt-8 p-4 bg-blue-50 rounded-lg">
          <h4 className="font-semibold text-blue-900 mb-2">Key Features Implemented:</h4>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>✓ Driver-based revenue calculations (Deliveries × ASP)</li>
            <li>✓ Vehicle model granularity (6 models)</li>
            <li>✓ Scenario-specific growth assumptions</li>
            <li>✓ Historical actuals (2020-2023) integration</li>
            <li>✓ 10-year forecast horizon (2024-2033)</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

// PHASE 2: Business Segment Analysis Component  
export const BusinessSegmentAnalysis = () => {
  const [segmentData, setSegmentData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Mock segment data for demonstration
    const mockSegmentData = {
      segments: {
        automotive: {
          name: 'Automotive',
          revenue_2024: 118300000000,
          revenue_2033: 245600000000,
          margin_2024: 0.19,
          margin_2033: 0.22,
          cagr: 0.084
        },
        energy: {
          name: 'Energy Generation & Storage', 
          revenue_2024: 8100000000,
          revenue_2033: 35400000000,
          margin_2024: 0.22,
          margin_2033: 0.28,
          cagr: 0.176
        },
        services: {
          name: 'Services & Other',
          revenue_2024: 11200000000,
          revenue_2033: 28900000000,
          margin_2024: 0.45,
          margin_2033: 0.52,
          cagr: 0.111
        }
      }
    };
    
    setSegmentData(mockSegmentData);
    setLoading(false);
  }, []);

  const formatCurrency = (value) => {
    if (value >= 1000000000) {
      return `$${(value / 1000000000).toFixed(1)}B`;
    } else if (value >= 1000000) {
      return `$${(value / 1000000).toFixed(1)}M`;
    }
    return `$${value}`;
  };

  const formatPercent = (value) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-purple-600 to-blue-600 text-white p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-2">Business Segment Analysis</h2>
        <p className="text-purple-100">PHASE 2: Automotive, Energy & Storage, Services breakdown</p>
      </div>

      {loading ? (
        <div className="text-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600 mx-auto"></div>
          <p className="mt-2 text-gray-600">Loading segment analysis...</p>
        </div>
      ) : segmentData ? (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {Object.entries(segmentData.segments).map(([key, segment]) => (
              <div key={key} className="bg-white p-6 rounded-lg shadow-sm border">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">{segment.name}</h3>
                
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-gray-600">2024 Revenue</span>
                      <span className="font-semibold">{formatCurrency(segment.revenue_2024)}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">2033 Projected</span>
                      <span className="font-semibold text-green-600">{formatCurrency(segment.revenue_2033)}</span>
                    </div>
                  </div>

                  <div className="border-t pt-4">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-gray-600">Current Margin</span>
                      <span className="font-semibold">{formatPercent(segment.margin_2024)}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">Target Margin</span>
                      <span className="font-semibold text-blue-600">{formatPercent(segment.margin_2033)}</span>
                    </div>
                  </div>

                  <div className="border-t pt-4">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">10-Year CAGR</span>
                      <span className="font-bold text-purple-600">{formatPercent(segment.cagr)}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Segment Growth Trajectory</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-green-50 rounded-lg">
                <div>
                  <h4 className="font-semibold text-green-900">Energy Storage</h4>
                  <p className="text-sm text-green-700">Fastest growing segment with 17.6% CAGR</p>
                </div>
                <div className="text-2xl font-bold text-green-600">17.6%</div>
              </div>
              
              <div className="flex items-center justify-between p-4 bg-blue-50 rounded-lg">
                <div>
                  <h4 className="font-semibold text-blue-900">Services & Other</h4>
                  <p className="text-sm text-blue-700">High-margin business with recurring revenue</p>
                </div>
                <div className="text-2xl font-bold text-blue-600">11.1%</div>
              </div>
              
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div>
                  <h4 className="font-semibold text-gray-900">Automotive</h4>
                  <p className="text-sm text-gray-700">Core business with steady growth</p>
                </div>
                <div className="text-2xl font-bold text-gray-600">8.4%</div>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};

// PHASE 3: Bridge Analysis Component
export const BridgeAnalysis = () => {
  const [bridgeData, setBridgeData] = useState(null);
  const [selectedAnalysis, setSelectedAnalysis] = useState('revenue');

  useEffect(() => {
    // Mock bridge analysis data
    const mockBridgeData = {
      revenue_bridge: {
        base_revenue: 96800000000,
        final_revenue: 245600000000,
        total_change: 148800000000,
        components: {
          volume_effect: 89280000000,
          price_effect: -14880000000,
          mix_effect: 29760000000,
          energy_growth: 29400000000,
          services_growth: 15240000000
        }
      },
      cash_flow_bridge: {
        base_fcf: 7500000000,
        final_fcf: 35200000000,
        total_change: 27700000000,
        components: {
          operating_performance: 22160000000,
          working_capital_impact: -2770000000,
          capex_investment: -8310000000,
          other_items: 16620000000
        }
      }
    };
    
    setBridgeData(mockBridgeData);
  }, []);

  const formatCurrency = (value) => {
    if (Math.abs(value) >= 1000000000) {
      return `$${(value / 1000000000).toFixed(1)}B`;
    } else if (Math.abs(value) >= 1000000) {
      return `$${(value / 1000000).toFixed(1)}M`;
    }
    return `$${value}`;
  };

  const renderWaterfallBar = (label, value, baseValue) => {
    const isPositive = value >= 0;
    const percentage = Math.abs(value) / Math.abs(baseValue) * 100;
    
    return (
      <div className="flex items-center space-x-4 py-3">
        <div className="w-32 text-sm font-medium text-gray-900">{label}</div>
        <div className="flex-1 flex items-center">
          <div className="w-full bg-gray-200 rounded-full h-6 relative">
            <div 
              className={`h-6 rounded-full ${isPositive ? 'bg-green-500' : 'bg-red-500'}`}
              style={{width: `${Math.min(percentage, 100)}%`}}
            ></div>
            <span className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-white">
              {formatCurrency(value)}
            </span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-orange-600 to-red-600 text-white p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-2">Bridge Analysis</h2>
        <p className="text-orange-100">PHASE 3: Waterfall charts showing drivers of change</p>
      </div>

      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <div className="flex space-x-4 mb-6">
          <button
            onClick={() => setSelectedAnalysis('revenue')}
            className={`px-4 py-2 rounded-lg font-medium ${
              selectedAnalysis === 'revenue' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Revenue Bridge
          </button>
          <button
            onClick={() => setSelectedAnalysis('cashflow')}
            className={`px-4 py-2 rounded-lg font-medium ${
              selectedAnalysis === 'cashflow' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Cash Flow Bridge
          </button>
        </div>

        {bridgeData && (
          <div>
            {selectedAnalysis === 'revenue' ? (
              <div>
                <h3 className="text-lg font-semibold mb-4">Revenue Bridge (2024 → 2033)</h3>
                <div className="space-y-2">
                  <div className="flex justify-between items-center p-4 bg-blue-50 rounded-lg">
                    <span className="font-semibold">Base Revenue (2024)</span>
                    <span className="font-bold text-blue-600">{formatCurrency(bridgeData.revenue_bridge.base_revenue)}</span>
                  </div>
                  
                  {Object.entries(bridgeData.revenue_bridge.components).map(([key, value]) => 
                    renderWaterfallBar(
                      key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()), 
                      value, 
                      bridgeData.revenue_bridge.total_change
                    )
                  )}
                  
                  <div className="flex justify-between items-center p-4 bg-green-50 rounded-lg mt-4">
                    <span className="font-semibold">Final Revenue (2033)</span>
                    <span className="font-bold text-green-600">{formatCurrency(bridgeData.revenue_bridge.final_revenue)}</span>
                  </div>
                </div>
              </div>
            ) : (
              <div>
                <h3 className="text-lg font-semibold mb-4">Free Cash Flow Bridge (2024 → 2033)</h3>
                <div className="space-y-2">
                  <div className="flex justify-between items-center p-4 bg-blue-50 rounded-lg">
                    <span className="font-semibold">Base FCF (2024)</span>
                    <span className="font-bold text-blue-600">{formatCurrency(bridgeData.cash_flow_bridge.base_fcf)}</span>
                  </div>
                  
                  {Object.entries(bridgeData.cash_flow_bridge.components).map(([key, value]) => 
                    renderWaterfallBar(
                      key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()), 
                      value, 
                      bridgeData.cash_flow_bridge.total_change
                    )
                  )}
                  
                  <div className="flex justify-between items-center p-4 bg-green-50 rounded-lg mt-4">
                    <span className="font-semibold">Final FCF (2033)</span>
                    <span className="font-bold text-green-600">{formatCurrency(bridgeData.cash_flow_bridge.final_fcf)}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};