import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ProactiveInsights = ({ scenario = 'base', modelData = null }) => {
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastKey, setLastKey] = useState(null);

  const computeCacheKey = (sc, md) => {
    const base = `pi:${sc || 'base'}`;
    if (!md) return `${base}:nomodel`;
    try {
      const str = JSON.stringify(md);
      // Simple fast hash (djb2)
      let h = 5381;
      for (let i = 0; i < str.length; i++) h = ((h << 5) + h) + str.charCodeAt(i);
      return `${base}:${(h >>> 0).toString(16)}`;
    } catch {
      return `${base}:nomodel`;
    }
  };

  useEffect(() => {
    if (!scenario) return;
    const key = computeCacheKey(scenario, modelData);
    setLastKey(key);
    // Try cache first to avoid refetch when user navigates away and back
    const cached = sessionStorage.getItem(key);
    if (cached) {
      try {
        setInsights(JSON.parse(cached));
        setLoading(false);
        setError(null);
        return; // serve from cache; no fetch
      } catch {}
    }
    // No cache → fetch
    fetchProactiveInsights(false, key);
  }, [scenario, modelData]);

  const fetchProactiveInsights = async (force = true, keyOverride = null) => {
    try {
      const key = keyOverride || computeCacheKey(scenario, modelData);
      
      // ALWAYS clear cache on refresh to avoid stale error responses
      if (force && key) {
        sessionStorage.removeItem(key);
        console.log('🗑️ Cleared cached insights for:', scenario);
      }
      
      if (!force) {
        const cached = sessionStorage.getItem(key);
        if (cached) {
          try {
            const parsedCache = JSON.parse(cached);
            // Don't use cached fallback responses
            if (parsedCache.key_insights?.[0]?.title !== "AI Analysis Unavailable") {
              setInsights(parsedCache);
              setLoading(false);
              setError(null);
              console.log('✅ Using cached insights');
              return;
            } else {
              console.log('⚠️ Cached response is fallback, fetching fresh...');
              sessionStorage.removeItem(key);
            }
          } catch {}
        }
      }

      setLoading(true);
      setError(null);

      // Calculate free_cash_flow if missing (backend cache issue workaround)
      let enrichedModelData = modelData;
      if (modelData && modelData.cash_flow_statements) {
        const firstCF = modelData.cash_flow_statements[0];
        console.log('📊 Model Data Cash Flow Fields:', Object.keys(firstCF));
        console.log('💰 Has free_cash_flow?', 'free_cash_flow' in firstCF);
        
        if (!('free_cash_flow' in firstCF)) {
          console.warn('⚠️ free_cash_flow field missing from backend - calculating in frontend...');
          
          // Calculate FCF for each year: FCF = Operating Cash Flow - |CapEx|
          enrichedModelData = {
            ...modelData,
            cash_flow_statements: modelData.cash_flow_statements.map(cf => ({
              ...cf,
              free_cash_flow: cf.operating_cash_flow - Math.abs(cf.capital_expenditures)
            }))
          };
          
          console.log('✅ Frontend calculated FCF for all years:', 
            enrichedModelData.cash_flow_statements.map(cf => 
              `${cf.year}: $${(cf.free_cash_flow / 1e9).toFixed(2)}B`
            ).join(', ')
          );
        } else {
          console.log('✅ FCF field present in backend data:', firstCF.free_cash_flow);
        }
      }

      const response = await axios.post(`${API}/ai/proactive-insights`, {
        scenario: scenario,
        model_data: enrichedModelData
      });

      if (response.data.success) {
        setInsights(response.data.insights);
        try {
          const cacheKey = keyOverride || computeCacheKey(scenario, modelData);
          sessionStorage.setItem(cacheKey, JSON.stringify(response.data.insights));
        } catch {}
      }
    } catch (err) {
      setError('Failed to fetch proactive insights');
      console.error('Proactive insights error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getInsightIcon = (type) => {
    const icons = {
      growth: '📈',
      risk: '⚠️',
      opportunity: '💡',
      financial_strength: '💪',
      operational_efficiency: '⚙️',
      product_mix: '🚗',
      competitive_pressure: '🏁',
      growth_concern: '📉',
      system_status: '🤖'
    };
    return icons[type] || '🔍';
  };

  const getImpactColor = (impact) => {
    switch (impact) {
      case 'positive': return 'text-green-600 bg-green-50 border-green-200';
      case 'negative': return 'text-red-600 bg-red-50 border-red-200';
      default: return 'text-blue-600 bg-blue-50 border-blue-200';
    }
  };

  const getRiskColor = (level) => {
    switch (level) {
      case 'high': return 'text-red-600 bg-red-50';
      case 'medium': return 'text-yellow-600 bg-yellow-50';
      case 'low': return 'text-green-600 bg-green-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-3 text-gray-600">Generating AI insights...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="text-center py-8">
          <div className="text-red-600 mb-2">⚠️ Error</div>
          <div className="text-gray-600">{error}</div>
          <button
            onClick={fetchProactiveInsights}
            className="mt-4 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!insights) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="text-center py-8 text-gray-600">
          No insights available. Click "Generate Insights" to analyze the model.
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-blue-600 text-white p-6 rounded-lg shadow-lg">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold mb-2">🤖 AI Proactive Insights</h2>
            <p className="text-purple-100">Intelligent analysis for {scenario.charAt(0).toUpperCase() + scenario.slice(1)} scenario</p>
          </div>
          <button
            onClick={() => fetchProactiveInsights(true, lastKey)}
            className="bg-white/20 hover:bg-white/30 text-white px-4 py-2 rounded-lg transition-colors"
          >
            🔄 Refresh Insights
          </button>
        </div>
      </div>

      {/* Key Insights */}
      {insights.key_insights && insights.key_insights.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border">
          <div className="p-6 border-b">
            <h3 className="text-lg font-semibold text-gray-900 flex items-center">
              <span className="mr-2">🎯</span>
              Key Insights
            </h3>
          </div>
          <div className="p-6 space-y-4">
            {insights.key_insights.map((insight, idx) => (
              <div key={idx} className={`p-4 rounded-lg border ${getImpactColor(insight.impact)}`}>
                <div className="flex items-start space-x-3">
                  <span className="text-2xl">{getInsightIcon(insight.type)}</span>
                  <div className="flex-1">
                    <h4 className="font-semibold mb-1">{insight.title}</h4>
                    <p className="text-sm mb-2">{insight.description}</p>
                    <div className="flex items-center space-x-4 text-xs">
                      <span className="font-medium">Confidence: {(insight.confidence * 100).toFixed(0)}%</span>
                      <span className="capitalize">Impact: {insight.impact}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Risk Alerts */}
      {insights.risk_alerts && insights.risk_alerts.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border">
          <div className="p-6 border-b">
            <h3 className="text-lg font-semibold text-gray-900 flex items-center">
              <span className="mr-2">⚠️</span>
              Risk Alerts
            </h3>
          </div>
          <div className="p-6 space-y-4">
            {insights.risk_alerts.map((alert, idx) => (
              <div key={idx} className="p-4 rounded-lg border border-red-200 bg-red-50">
                <div className="flex items-start space-x-3">
                  <span className="text-2xl">🚨</span>
                  <div className="flex-1">
                    <h4 className="font-semibold text-red-900 mb-1">{alert.title}</h4>
                    <p className="text-sm text-red-700 mb-2">{alert.description}</p>
                    <div className="flex items-center space-x-4 text-xs text-red-600">
                      <span className="font-medium">Confidence: {(alert.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Opportunities */}
      {insights.opportunities && insights.opportunities.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border">
          <div className="p-6 border-b">
            <h3 className="text-lg font-semibold text-gray-900 flex items-center">
              <span className="mr-2">💡</span>
              Opportunities
            </h3>
          </div>
          <div className="p-6 space-y-4">
            {insights.opportunities.map((opportunity, idx) => (
              <div key={idx} className="p-4 rounded-lg border border-green-200 bg-green-50">
                <div className="flex items-start space-x-3">
                  <span className="text-2xl">🌟</span>
                  <div className="flex-1">
                    <h4 className="font-semibold text-green-900 mb-1">{opportunity.title}</h4>
                    <p className="text-sm text-green-700 mb-2">{opportunity.description}</p>
                    <div className="flex items-center space-x-4 text-xs text-green-600">
                      <span className="font-medium">Confidence: {(opportunity.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      {insights.recommendations && insights.recommendations.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border">
          <div className="p-6 border-b">
            <h3 className="text-lg font-semibold text-gray-900 flex items-center">
              <span className="mr-2">📋</span>
              Strategic Recommendations
            </h3>
          </div>
          <div className="p-6 space-y-4">
            {insights.recommendations.map((rec, idx) => (
              <div key={idx} className="p-4 rounded-lg border border-blue-200 bg-blue-50">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <h4 className="font-semibold text-blue-900">{rec.title}</h4>
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        rec.priority === 'high' ? 'bg-red-100 text-red-800' :
                        rec.priority === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-green-100 text-green-800'
                      }`}>
                        {rec.priority} priority
                      </span>
                    </div>
                    <p className="text-sm text-blue-700 mb-2">{rec.description}</p>
                    <div className="text-xs text-blue-600">
                      <span className="font-medium">Timeline: {rec.timeline}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Market Context */}
      {insights.market_context && insights.market_context.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border">
          <div className="p-6 border-b">
            <h3 className="text-lg font-semibold text-gray-900 flex items-center">
              <span className="mr-2">🌍</span>
              Market Context
            </h3>
          </div>
          <div className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {insights.market_context.map((context, idx) => (
                <div key={idx} className="p-4 rounded-lg bg-gray-50 border">
                  <h4 className="font-semibold text-gray-900 mb-2">{context.factor}</h4>
                  <p className="text-sm text-gray-700 mb-2">{context.description}</p>
                  <span className={`px-2 py-1 text-xs rounded-full ${
                    context.relevance === 'high' ? 'bg-red-100 text-red-800' :
                    context.relevance === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-green-100 text-green-800'
                  }`}>
                    {context.relevance} relevance
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProactiveInsights;