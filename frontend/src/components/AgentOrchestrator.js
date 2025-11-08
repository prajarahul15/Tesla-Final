import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const AgentOrchestrator = () => {
  const [query, setQuery] = useState('');
  const [scenario, setScenario] = useState('base');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [availableAgents, setAvailableAgents] = useState([]);
  const [sessionId, setSessionId] = useState(null);

  useEffect(() => {
    fetchAvailableAgents();
    // Generate session ID
    setSessionId(`orchestrator-${Date.now()}`);
  }, []);

  const fetchAvailableAgents = async () => {
    try {
      const response = await axios.get(`${API}/orchestrator/agents`);
      setAvailableAgents(response.data.agents || []);
    } catch (error) {
      console.error('Error fetching agents:', error);
    }
  };

  const executeQuery = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setResult(null);

    try {
      const response = await axios.post(`${API}/orchestrator/ask`, {
        query: query,
        session_id: sessionId,
        context: {
          scenario: scenario,
          year: 2025
        }
      }, {
        timeout: 90000
      });

      setResult(response.data);
    } catch (error) {
      console.error('Orchestration error:', error);
      setResult({
        success: false,
        error: error.response?.data?.detail || error.message
      });
    } finally {
      setLoading(false);
    }
  };

  const sampleQueries = [
    "Forecast revenue for next 12 months and analyze the risks",
    "Simulate 20% revenue growth and show impact on all financial statements",
    "What are the key drivers of profitability and how can we improve margins?",
    "Compare best and worst case scenarios and provide strategic recommendations",
    "Analyze cash flow trends and predict working capital needs"
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 text-white p-6 rounded-lg shadow-lg">
        <div className="flex items-center space-x-3 mb-2">
          <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
          </svg>
          <div>
            <h2 className="text-3xl font-bold">🤖 Agent Orchestrator</h2>
            <p className="text-purple-100 mt-1">Multi-agent AI system with intelligent task coordination</p>
          </div>
        </div>
        <div className="mt-4 flex items-center space-x-4 text-sm">
          <div className="flex items-center space-x-2 bg-white/20 px-3 py-1 rounded-full">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <span>{availableAgents.length} Agents Active</span>
          </div>
          <div className="bg-white/20 px-3 py-1 rounded-full">
            Session: {sessionId?.slice(-8)}
          </div>
        </div>
      </div>

      {/* Available Agents */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Available AI Agents</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {availableAgents.map((agent, index) => (
            <div key={index} className="bg-gradient-to-br from-blue-50 to-purple-50 p-4 rounded-lg border-2 border-blue-200">
              <h4 className="font-semibold text-gray-900 mb-1">{agent.name}</h4>
              <p className="text-sm text-gray-600 mb-2">{agent.description}</p>
              <div className="flex flex-wrap gap-1">
                {agent.capabilities?.map((cap, i) => (
                  <span key={i} className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded-full">
                    {cap}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Query Interface */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Ask the Orchestrator</h3>
        
        {/* Scenario Selector */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">Scenario Context</label>
          <select
            value={scenario}
            onChange={(e) => setScenario(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
          >
            <option value="best">Best Case</option>
            <option value="base">Base Case</option>
            <option value="worst">Worst Case</option>
          </select>
        </div>

        {/* Query Input */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">Your Query</label>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a complex question that may require multiple agents..."
            rows={4}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
          />
        </div>

        {/* Sample Queries */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">Sample Queries (Click to use)</label>
          <div className="flex flex-wrap gap-2">
            {sampleQueries.map((sample, index) => (
              <button
                key={index}
                onClick={() => setQuery(sample)}
                className="text-xs px-3 py-2 bg-purple-100 text-purple-700 rounded-lg hover:bg-purple-200 transition-colors"
              >
                {sample}
              </button>
            ))}
          </div>
        </div>

        {/* Execute Button */}
        <button
          onClick={executeQuery}
          disabled={loading || !query.trim()}
          className="w-full bg-gradient-to-r from-purple-600 to-pink-600 text-white py-3 px-6 rounded-lg font-medium hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          {loading ? (
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
              Orchestrating Agents...
            </div>
          ) : (
            <div className="flex items-center justify-center">
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Execute Orchestrated Query
            </div>
          )}
        </button>
      </div>

      {/* Results */}
      {result && (
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Orchestration Results</h3>

          {result.success ? (
            <div className="space-y-4">
              {/* Execution Metadata */}
              <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-4 rounded-lg border-2 border-purple-200">
                <h4 className="font-semibold text-gray-900 mb-2">📊 Execution Details</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Task Type:</span>
                    <div className="font-medium text-purple-700">{result.task_type}</div>
                  </div>
                  <div>
                    <span className="text-gray-600">Agents Used:</span>
                    <div className="font-medium text-blue-700">{result.agents_used?.length || 0}</div>
                  </div>
                  <div>
                    <span className="text-gray-600">Tasks Executed:</span>
                    <div className="font-medium text-green-700">{result.tasks_executed}</div>
                  </div>
                  <div>
                    <span className="text-gray-600">Session ID:</span>
                    <div className="font-medium text-gray-700 text-xs">{result.session_id?.slice(-12)}</div>
                  </div>
                </div>
                
                {/* Agents Used Badges */}
                <div className="mt-3 flex flex-wrap gap-2">
                  <span className="text-xs text-gray-600">Agents:</span>
                  {result.agents_used?.map((agent, i) => (
                    <span key={i} className="text-xs px-2 py-1 bg-purple-600 text-white rounded-full">
                      {agent.replace(/_/g, ' ')}
                    </span>
                  ))}
                </div>
              </div>

              {/* Executive Summary */}
              {result.result?.executive_summary && (
                <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                  <h4 className="font-semibold text-blue-900 mb-2">Executive Summary</h4>
                  <p className="text-blue-800">{result.result.executive_summary}</p>
                </div>
              )}

              {/* Key Insights */}
              {result.result?.key_insights && (
                <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                  <h4 className="font-semibold text-green-900 mb-2">Key Insights</h4>
                  <ul className="list-disc list-inside space-y-1 text-green-800">
                    {result.result.key_insights.map((insight, i) => (
                      <li key={i}>{insight}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Recommendations */}
              {result.result?.recommendations && (
                <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                  <h4 className="font-semibold text-orange-900 mb-2">Recommendations</h4>
                  <ul className="list-disc list-inside space-y-1 text-orange-800">
                    {result.result.recommendations.map((rec, i) => (
                      <li key={i}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Detailed Findings */}
              {result.result?.detailed_findings && (
                <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                  <h4 className="font-semibold text-purple-900 mb-2">Detailed Findings</h4>
                  <div className="space-y-2 text-sm">
                    {Object.entries(result.result.detailed_findings).map(([agent, findings], i) => (
                      <div key={i}>
                        <span className="font-medium text-purple-900">{agent}:</span>
                        <span className="text-purple-800 ml-2">{findings}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Next Steps */}
              {result.result?.next_steps && (
                <div className="bg-indigo-50 p-4 rounded-lg border border-indigo-200">
                  <h4 className="font-semibold text-indigo-900 mb-2">Suggested Next Steps</h4>
                  <ol className="list-decimal list-inside space-y-1 text-indigo-800">
                    {result.result.next_steps.map((step, i) => (
                      <li key={i}>{step}</li>
                    ))}
                  </ol>
                </div>
              )}

              {/* Raw Results (Collapsible) */}
              <details className="bg-gray-50 p-4 rounded-lg border">
                <summary className="font-semibold text-gray-900 cursor-pointer hover:text-purple-600">
                  Raw Agent Results (Click to expand)
                </summary>
                <pre className="mt-3 text-xs text-gray-700 overflow-x-auto bg-white p-3 rounded border">
                  {JSON.stringify(result.result?.raw_results || result.result, null, 2)}
                </pre>
              </details>
            </div>
          ) : (
            <div className="bg-red-50 p-4 rounded-lg border border-red-200">
              <h4 className="font-semibold text-red-900 mb-2">Error</h4>
              <p className="text-red-800">{result.error || 'Unknown error occurred'}</p>
            </div>
          )}
        </div>
      )}

      {/* How It Works */}
      <div className="bg-gradient-to-r from-gray-50 to-gray-100 p-6 rounded-lg border">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">🔍 How Agent Orchestration Works</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div className="space-y-2">
            <h4 className="font-medium text-purple-900">1. Query Analysis</h4>
            <p className="text-gray-700">Orchestrator analyzes your question to determine complexity and required capabilities</p>
            
            <h4 className="font-medium text-purple-900 mt-3">2. Agent Selection</h4>
            <p className="text-gray-700">Intelligently routes to single or multiple agents based on query requirements</p>
            
            <h4 className="font-medium text-purple-900 mt-3">3. Task Decomposition</h4>
            <p className="text-gray-700">Breaks complex queries into atomic sub-tasks with dependencies</p>
          </div>
          
          <div className="space-y-2">
            <h4 className="font-medium text-purple-900">4. Parallel Execution</h4>
            <p className="text-gray-700">Executes independent tasks in parallel for efficiency</p>
            
            <h4 className="font-medium text-purple-900 mt-3">5. Context Sharing</h4>
            <p className="text-gray-700">Agents share results through a centralized memory store</p>
            
            <h4 className="font-medium text-purple-900 mt-3">6. Result Synthesis</h4>
            <p className="text-gray-700">AI aggregates and synthesizes insights from all agents into a coherent response</p>
          </div>
        </div>

        {/* Architecture Diagram */}
        <div className="mt-6 p-4 bg-white rounded-lg border">
          <h4 className="font-medium text-gray-900 mb-3 text-center">Orchestration Flow</h4>
          <div className="flex items-center justify-center space-x-2 text-xs">
            <div className="px-3 py-2 bg-blue-100 text-blue-800 rounded-lg font-medium">User Query</div>
            <span>→</span>
            <div className="px-3 py-2 bg-purple-100 text-purple-800 rounded-lg font-medium">Orchestrator</div>
            <span>→</span>
            <div className="px-3 py-2 bg-green-100 text-green-800 rounded-lg font-medium">Multi-Agent</div>
            <span>→</span>
            <div className="px-3 py-2 bg-orange-100 text-orange-800 rounded-lg font-medium">Synthesis</div>
            <span>→</span>
            <div className="px-3 py-2 bg-pink-100 text-pink-800 rounded-lg font-medium">Response</div>
          </div>
        </div>
      </div>

      {/* Examples */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">💡 Example Multi-Agent Workflows</h3>
        <div className="space-y-3">
          <div className="p-4 bg-blue-50 rounded-lg border-l-4 border-blue-500">
            <p className="text-sm font-medium text-blue-900 mb-1">Query: "Forecast revenue and analyze risks"</p>
            <p className="text-xs text-blue-700">
              <strong>Agents:</strong> Prophet Forecasting → Proactive Insights<br/>
              <strong>Flow:</strong> Generate forecast, then analyze risks and opportunities
            </p>
          </div>
          
          <div className="p-4 bg-green-50 rounded-lg border-l-4 border-green-500">
            <p className="text-sm font-medium text-green-900 mb-1">Query: "Simulate 25% growth and show impact"</p>
            <p className="text-xs text-green-700">
              <strong>Agents:</strong> Tesla AI Simulator → Cross-Statement Analyst<br/>
              <strong>Flow:</strong> Run simulation, then analyze impact across all statements
            </p>
          </div>
          
          <div className="p-4 bg-purple-50 rounded-lg border-l-4 border-purple-500">
            <p className="text-sm font-medium text-purple-900 mb-1">Query: "Compare scenarios and recommend best strategy"</p>
            <p className="text-xs text-purple-700">
              <strong>Agents:</strong> Multiple agents (Proactive Insights, Cross-Statement, Tesla FA)<br/>
              <strong>Flow:</strong> Analyze each scenario, compare results, synthesize recommendation
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgentOrchestrator;

