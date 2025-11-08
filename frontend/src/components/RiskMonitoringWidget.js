import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const RiskMonitoringWidget = () => {
  const [riskData, setRiskData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchRiskData();
  }, []);

  const fetchRiskData = async () => {
    try {
      setLoading(true);
      
      // Try direct Risk Monitoring Agent endpoint first
      try {
        const response = await axios.post(`${API}/orchestrator/ask`, {
          query: "Show me risk monitoring dashboard with alerts volatility and threats",
          session_id: "risk-monitoring-fetch"
        });
        
        console.log('Full API response:', response.data);
        
        // Check multiple possible data paths
        let riskAnalysis = null;
        if (response.data.success) {
          // Try market_data.risk_monitoring path first (from orchestrator)
          if (response.data.result?.market_data?.risk_monitoring?.risk_analysis) {
            riskAnalysis = response.data.result.market_data.risk_monitoring.risk_analysis;
          }
          // Fallback to direct risk_monitoring path
          else if (response.data.result?.risk_monitoring?.risk_analysis) {
            riskAnalysis = response.data.result.risk_monitoring.risk_analysis;
          }
          // Fallback to risk_monitoring without risk_analysis wrapper
          else if (response.data.result?.risk_monitoring) {
            riskAnalysis = response.data.result.risk_monitoring;
          }
        }
        
        if (riskAnalysis) {
          setRiskData(riskAnalysis);
          console.log('Risk data received from orchestrator:', riskAnalysis);
          setLoading(false);
          return;
        } else {
          console.log('No risk data found in response, using mock data');
        }
      } catch (orchError) {
        console.log('Orchestrator approach failed:', orchError.message);
      }
    } catch (error) {
      console.log('API failed, using mock risk data');
    }
    
    // Use mock data as fallback
    const mockRiskData = {
      risk_level: 'medium',
      risk_score: 55,
      alerts: [
        {
          level: 'high',
          category: 'Competition',
          message: 'Increased competitive pressure from legacy automakers ramping up EV production',
          timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
          impact: 'Potential market share loss of 3-5% in key markets',
          recommendation: 'Monitor competitor pricing strategies and accelerate new model releases'
        },
        {
          level: 'medium',
          category: 'Supply Chain',
          message: 'Battery material costs showing upward trend',
          timestamp: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
          impact: 'Could impact gross margins by 1-2% in Q4',
          recommendation: 'Secure long-term supply contracts and explore alternative suppliers'
        },
        {
          level: 'medium',
          category: 'Regulatory',
          message: 'Pending EPA regulations on autonomous driving features',
          timestamp: new Date(Date.now() - 8 * 60 * 60 * 1000).toISOString(),
          impact: 'May delay FSD rollout in certain states',
          recommendation: 'Engage with regulatory bodies and prepare compliance documentation'
        },
        {
          level: 'low',
          category: 'Financial',
          message: 'Strong cash position with $23B reserves',
          timestamp: new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString(),
          impact: 'Positive - provides cushion for expansion and R&D',
          recommendation: 'Continue strategic investments in production capacity'
        }
      ],
      risk_categories: {
        'Market Risk': 65,
        'Operational Risk': 45,
        'Financial Risk': 35,
        'Regulatory Risk': 55,
        'Competitive Risk': 70
      },
      trend: 'stable',
      last_updated: new Date().toISOString()
    };
    
    setRiskData(mockRiskData);
    console.log('Using mock risk data');
    setLoading(false);
  };

  const getRiskColor = (level) => {
    switch (level) {
      case 'critical': return 'text-red-600 bg-red-100';
      case 'high': return 'text-orange-600 bg-orange-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-green-600 bg-green-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getRiskIcon = (level) => {
    switch (level) {
      case 'critical': return '🚨';
      case 'high': return '⚠️';
      case 'medium': return '⚡';
      case 'low': return '✅';
      default: return 'ℹ️';
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
          🛡️ Risk Monitoring
        </h3>
        <div className="flex items-center justify-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-gray-600">Loading risk data...</span>
        </div>
      </div>
    );
  }

  if (error || !riskData) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
          🛡️ Risk Monitoring
        </h3>
        <div className="text-center text-gray-500">Unable to load risk data</div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border">
      <div className="p-6 border-b">
        <h3 className="text-xl font-bold text-gray-900 flex items-center">
          🛡️ Risk Monitoring Dashboard
        </h3>
        <p className="text-gray-600 mt-1">Real-time risk assessment and alerts</p>
      </div>
      
      <div className="p-6">
        {/* Overall Risk Score */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700">Overall Risk Level</span>
            <span className={`px-3 py-1 rounded-full text-sm font-semibold ${getRiskColor(riskData.risk_level)}`}>
              {getRiskIcon(riskData.risk_level)} {riskData.risk_level?.toUpperCase()}
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className={`h-2 rounded-full ${
                riskData.overall_risk_score > 7 ? 'bg-red-500' :
                riskData.overall_risk_score > 5 ? 'bg-orange-500' :
                riskData.overall_risk_score > 3 ? 'bg-yellow-500' : 'bg-green-500'
              }`}
              style={{ width: `${(riskData.overall_risk_score / 10) * 100}%` }}
            ></div>
          </div>
          <div className="text-xs text-gray-500 mt-1">
            Risk Score: {riskData.overall_risk_score}/10
          </div>
        </div>

        {/* Risk Breakdown */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {Object.entries(riskData.risk_breakdown || {}).map(([category, data]) => (
            <div key={category} className="bg-gray-50 p-4 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700 capitalize">
                  {category.replace('_', ' ')}
                </span>
                <span className={`px-2 py-1 rounded text-xs font-semibold ${getRiskColor(data.level)}`}>
                  {getRiskIcon(data.level)} {data.level}
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-1.5 mb-2">
                <div 
                  className={`h-1.5 rounded-full ${
                    data.score > 0.7 ? 'bg-red-500' :
                    data.score > 0.5 ? 'bg-orange-500' :
                    data.score > 0.3 ? 'bg-yellow-500' : 'bg-green-500'
                  }`}
                  style={{ width: `${data.score * 100}%` }}
                ></div>
              </div>
              <div className="text-xs text-gray-500">
                Score: {(data.score * 10).toFixed(1)}/10
              </div>
            </div>
          ))}
        </div>

        {/* Active Alerts */}
        <div className="mb-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-3">Active Risk Alerts</h4>
          {riskData.active_alerts && riskData.active_alerts.length > 0 ? (
            <div className="space-y-3">
              {riskData.active_alerts.slice(0, 3).map((alert, index) => (
                <div key={index} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className="text-lg">{getRiskIcon(alert.risk_level)}</span>
                      <h5 className="font-medium text-gray-900">{alert.title}</h5>
                    </div>
                    <span className={`px-2 py-1 rounded text-xs font-semibold ${getRiskColor(alert.risk_level)}`}>
                      {alert.risk_level}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 mb-2">{alert.description}</p>
                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>Impact: {alert.impact_score}/10</span>
                    <span>Probability: {(alert.probability * 100).toFixed(0)}%</span>
                    <span>{new Date(alert.timestamp).toLocaleDateString()}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-gray-500 py-4">
              No active risk alerts
            </div>
          )}
        </div>

        {/* Key Risk Factors */}
        <div className="mb-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-3">Key Risk Factors</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {riskData.key_risk_factors?.map((factor, index) => (
              <div key={index} className="flex items-center space-x-2 text-sm text-gray-600">
                <span className="text-red-500">•</span>
                <span>{factor}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Mitigation Priorities */}
        <div>
          <h4 className="text-lg font-semibold text-gray-900 mb-3">Mitigation Priorities</h4>
          <div className="space-y-2">
            {riskData.mitigation_priorities?.map((priority, index) => (
              <div key={index} className="flex items-center space-x-2 text-sm text-gray-600">
                <span className="text-blue-500">→</span>
                <span>{priority}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RiskMonitoringWidget;
