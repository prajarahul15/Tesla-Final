import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ScenarioTabs from './ScenarioTabs';
import FinancialStatements from './FinancialStatements';
import DataOverview from './DataOverview';
import ForecastPage from './ForecastPage';
import { VehicleModelAnalysis, BusinessSegmentAnalysis, BridgeAnalysis } from './EnhancedTeslaComponents';
import LoadingSpinner from './LoadingSpinner';
import ProactiveInsights from './ProactiveInsights';
import VehicleForecast from './VehicleForecast';
import VehicleForecastAgentBased from './VehicleForecastAgentBased';
import NewVehicleModels from './NewVehicleModels';
import TeslaFAChat from './TeslaFAChat';
import ExecutiveDashboard from './ExecutiveDashboard';
import AgentOrchestrator from './AgentOrchestrator';
import MarketInsightsPage from './MarketInsightsPage';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const TeslaDashboard = () => {
  const [activeScenario, setActiveScenario] = useState('base');
  const [teslaData, setTeslaData] = useState(null);
  const [financialModels, setFinancialModels] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('executive-dashboard');

  useEffect(() => {
    // Load Tesla overview and automatically generate all scenarios on startup
    const initializeDashboard = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Fetch Tesla overview
        const overviewResponse = await axios.get(`${API}/tesla/overview`);
        setTeslaData(overviewResponse.data);
        
        // Generate all scenarios automatically
        const scenarios = ['best', 'base', 'worst'];
        console.log('🚀 Auto-generating all scenarios on startup...');
        
        for (const scenario of scenarios) {
          console.log(`Generating ${scenario} scenario...`);
          const response = await axios.post(`${API}/tesla/model/${scenario}`);
          
          if (response.data.success) {
            setFinancialModels(prev => ({
              ...prev,
              [scenario]: response.data.model
            }));
            console.log(`✅ Generated ${scenario} model successfully`);
          }
          
          // Small delay between requests
          if (scenario !== 'worst') {
            await new Promise(resolve => setTimeout(resolve, 500));
          }
        }
        
        console.log('✅ All scenarios loaded successfully');
        setLoading(false);
      } catch (err) {
        console.error('❌ Failed to initialize dashboard:', err);
        setError('Failed to load financial models');
        setLoading(false);
      }
    };
    
    initializeDashboard();
  }, []); // Empty dependency array - only run once on mount

  const fetchTeslaOverview = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API}/tesla/overview`);
      setTeslaData(response.data);
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch Tesla overview data');
      setLoading(false);
    }
  };

  const generateFinancialModel = async (scenario) => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.post(`${API}/tesla/model/${scenario}`);
      
      if (response.data.success) {
        setFinancialModels(prev => ({
          ...prev,
          [scenario]: response.data.model
        }));
        console.log(`✅ Generated ${scenario} model successfully:`, response.data.model);
      } else {
        setError(`Failed to generate ${scenario} scenario model`);
      }
      setLoading(false);
    } catch (err) {
      console.error(`❌ Error generating ${scenario} model:`, err);
      setError(`Failed to generate ${scenario} scenario model: ${err.message}`);
      setLoading(false);
    }
  };

  const generateAllScenarios = async () => {
    setLoading(true);
    setError(null);
    try {
      const scenarios = ['best', 'base', 'worst'];
      
      // Generate scenarios sequentially to avoid overwhelming the backend
      for (const scenario of scenarios) {
        console.log(`Generating ${scenario} scenario...`);
        await generateFinancialModel(scenario);
        await new Promise(resolve => setTimeout(resolve, 1000)); // Small delay between requests
      }
      
      console.log('✅ All scenarios generated successfully');
    } catch (err) {
      console.error('❌ Failed to generate all scenarios:', err);
      setError('Failed to generate all scenarios');
    }
    setLoading(false);
  };

  if (loading && !teslaData) {
    return <LoadingSpinner message="Loading Tesla Financial Model..." />;
  }

  if (error) {
    return (
      <div className="min-h-screen tesla-gray-light-bg flex items-center justify-center">
        <div className="tesla-alert-error">
          {error}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen tesla-gray-light-bg">
      {/* Header */}
      <header className="tesla-white-bg shadow-sm tesla-gray-border border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="tesla-text-3xl tesla-font-bold tesla-black">Tesla Financial Model & Analytics</h1>
            </div>
            <div className="flex space-x-3">
              <button
                onClick={() => generateFinancialModel(activeScenario)}
                className="tesla-button-primary"
              >
                Generate {activeScenario.charAt(0).toUpperCase() + activeScenario.slice(1)} Model
              </button>
              <button
                onClick={generateAllScenarios}
                className="tesla-button-secondary flex items-center"
              >
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Regenerate All Scenarios
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Tesla Overview */}
        {teslaData && activeTab === 'overview' && (
          <div className="tesla-card mb-8">
            <h2 className="tesla-text-xl tesla-font-semibold tesla-black mb-4">Tesla Overview (Base Year 2024)</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="tesla-text-2xl tesla-font-bold tesla-red">
                  {(teslaData.tesla_base_data.total_deliveries / 1000000).toFixed(2)}M
                </div>
                <div className="tesla-text-sm tesla-gray">Total Deliveries</div>
              </div>
              <div className="text-center">
                <div className="tesla-text-2xl tesla-font-bold tesla-green">
                  ${(teslaData.tesla_base_data.total_revenue / 1000000000).toFixed(1)}B
                </div>
                <div className="tesla-text-sm tesla-gray">Total Revenue</div>
              </div>
              <div className="text-center">
                <div className="tesla-text-2xl tesla-font-bold tesla-blue">
                  ${(teslaData.tesla_base_data.net_income / 1000000000).toFixed(1)}B
                </div>
                <div className="tesla-text-sm tesla-gray">Net Income</div>
              </div>
              <div className="text-center">
                <div className="tesla-text-2xl tesla-font-bold tesla-orange">
                  ${(teslaData.tesla_base_data.cash_and_equivalents / 1000000000).toFixed(1)}B
                </div>
                <div className="tesla-text-sm tesla-gray">Cash & Equivalents</div>
              </div>
            </div>
          </div>
        )}

        {/* Navigation Tabs */}
        <div className="tesla-card mb-8">
          <div className="tesla-gray-border border-b">
            <nav className="flex space-x-8 px-6">
              {[
                { id: 'executive-dashboard', name: 'Executive Dashboard' },
                { id: 'vehicle-forecast', name: 'Vehicle Forecast' },
                { id: 'new-vehicle-models', name: 'New Vehicle Models' },
                { id: 'overview', name: 'Tesla Overview' },
                { id: 'statements', name: 'Financial Statements' },
                { id: 'tesla-fa-chat', name: 'Ask Tesla FA' },
                { id: 'agent-orchestrator', name: '🤖 Agent Orchestrator' },
                { id: 'market-insights', name: '📊 Market Intelligence' },
                { id: 'ai-insights', name: 'AI Insights' }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`py-4 px-1 border-b-2 tesla-font-medium tesla-text-sm ${
                    activeTab === tab.id
                      ? 'tesla-red-border tesla-red'
                      : 'border-transparent tesla-gray hover:tesla-black hover:tesla-gray-border'
                  }`}
                >
                  {tab.name}
                </button>
              ))}
            </nav>
          </div>

          <div className="p-6">
            {activeTab === 'executive-dashboard' && (
              <ExecutiveDashboard />
            )}

            {activeTab === 'agent-orchestrator' && (
              <AgentOrchestrator />
            )}

            {activeTab === 'overview' && (
              <ScenarioTabs 
                activeScenario={activeScenario}
                setActiveScenario={setActiveScenario}
                financialModels={financialModels}
                generateModel={generateFinancialModel}
                loading={loading}
              />
            )}

            {activeTab === 'new-vehicle-models' && (
              <NewVehicleModels />
            )}

            {activeTab === 'market-insights' && (
              <MarketInsightsPage />
            )}

            {activeTab === 'tesla-fa-chat' && (
              <TeslaFAChat />
            )}

            {activeTab === 'statements' && (
              <FinancialStatements 
                scenario={activeScenario}
                model={financialModels[activeScenario]}
                generateModel={generateFinancialModel}
                loading={loading}
                models={financialModels}
                generateAllScenarios={generateAllScenarios}
              />
            )}

            {activeTab === 'data-overview' && (
              <DataOverview />
            )}

            {activeTab === 'vehicle-forecast' && (
              <VehicleForecastAgentBased />
            )}

            {activeTab === 'ai-insights' && (
              <ProactiveInsights 
                scenario={activeScenario}
                modelData={financialModels[activeScenario]}
              />
            )}

          </div>
        </div>
      </main>
    </div>
  );
};

export default TeslaDashboard;