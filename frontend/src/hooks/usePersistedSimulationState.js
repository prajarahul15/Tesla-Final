import { useState, useEffect, useCallback } from 'react';

/**
 * Custom hook for managing persisted simulation state across page navigation
 * Uses sessionStorage to maintain state per scenario and simulation type
 */
export const usePersistedSimulationState = (scenario, simulationType) => {
  const getStorageKey = (key) => `simulation_${simulationType}_${scenario}_${key}`;
  
  // Helper to get stored value or default
  const getStoredValue = useCallback((key, defaultValue) => {
    try {
      const stored = sessionStorage.getItem(getStorageKey(key));
      return stored ? JSON.parse(stored) : defaultValue;
    } catch (error) {
      console.warn(`Failed to parse stored value for ${key}:`, error);
      return defaultValue;
    }
  }, [scenario, simulationType]);

  // Helper to store value
  const setStoredValue = useCallback((key, value) => {
    try {
      sessionStorage.setItem(getStorageKey(key), JSON.stringify(value));
    } catch (error) {
      console.warn(`Failed to store value for ${key}:`, error);
    }
  }, [scenario, simulationType]);

  // Income Statement simulation state
  const [incomeSimValues, setIncomeSimValues] = useState(() => 
    getStoredValue('incomeSimValues', {
      automotive_revenue_growth: '',
      services_revenue_growth: '',
      gross_margin_automotive: '',
      gross_margin_services: '',
      rd_as_percent_revenue: '',
      sga_as_percent_revenue: ''
    })
  );

  const [updatedIncome, setUpdatedIncome] = useState(() => 
    getStoredValue('updatedIncome', null)
  );

  const [incomeSimError, setIncomeSimError] = useState(() => 
    getStoredValue('incomeSimError', null)
  );

  const [incomeAiInsights, setIncomeAiInsights] = useState(() => 
    getStoredValue('incomeAiInsights', null)
  );

  // Cross Statement simulation state
  const [crossSimValues, setCrossSimValues] = useState(() => 
    getStoredValue('crossSimValues', {
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
      depreciation_rate: '',
      tax_rate: ''
    })
  );

  const [crossSimResults, setCrossSimResults] = useState(() => 
    getStoredValue('crossSimResults', null)
  );

  const [crossSimError, setCrossSimError] = useState(() => 
    getStoredValue('crossSimError', null)
  );

  const [crossAiInsights, setCrossAiInsights] = useState(() => 
    getStoredValue('crossAiInsights', null)
  );

  // Loading states (not persisted as they should reset on navigation)
  const [incomeSimLoading, setIncomeSimLoading] = useState(false);
  const [incomeInsightLoading, setIncomeInsightLoading] = useState(false);
  const [crossSimLoading, setCrossSimLoading] = useState(false);
  const [crossInsightLoading, setCrossInsightLoading] = useState(false);

  // Persist income simulation state
  useEffect(() => {
    setStoredValue('incomeSimValues', incomeSimValues);
  }, [incomeSimValues, setStoredValue]);

  useEffect(() => {
    setStoredValue('updatedIncome', updatedIncome);
  }, [updatedIncome, setStoredValue]);

  useEffect(() => {
    setStoredValue('incomeSimError', incomeSimError);
  }, [incomeSimError, setStoredValue]);

  useEffect(() => {
    setStoredValue('incomeAiInsights', incomeAiInsights);
  }, [incomeAiInsights, setStoredValue]);

  // Persist cross statement simulation state
  useEffect(() => {
    setStoredValue('crossSimValues', crossSimValues);
  }, [crossSimValues, setStoredValue]);

  useEffect(() => {
    setStoredValue('crossSimResults', crossSimResults);
  }, [crossSimResults, setStoredValue]);

  useEffect(() => {
    setStoredValue('crossSimError', crossSimError);
  }, [crossSimError, setStoredValue]);

  useEffect(() => {
    setStoredValue('crossAiInsights', crossAiInsights);
  }, [crossAiInsights, setStoredValue]);

  // Clear all stored state for this scenario and simulation type
  const clearPersistedState = useCallback(() => {
    const keys = [
      'incomeSimValues', 'updatedIncome', 'incomeSimError', 'incomeAiInsights',
      'crossSimValues', 'crossSimResults', 'crossSimError', 'crossAiInsights'
    ];
    
    keys.forEach(key => {
      sessionStorage.removeItem(getStorageKey(key));
    });

    // Reset to default values
    setIncomeSimValues({
      automotive_revenue_growth: '',
      services_revenue_growth: '',
      gross_margin_automotive: '',
      gross_margin_services: '',
      rd_as_percent_revenue: '',
      sga_as_percent_revenue: ''
    });
    setUpdatedIncome(null);
    setIncomeSimError(null);
    setIncomeAiInsights(null);

    setCrossSimValues({
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
      depreciation_rate: '',
      tax_rate: ''
    });
    setCrossSimResults(null);
    setCrossSimError(null);
    setCrossAiInsights(null);
  }, [scenario, simulationType]);

  return {
    // Income Statement simulation state
    incomeSimValues,
    setIncomeSimValues,
    updatedIncome,
    setUpdatedIncome,
    incomeSimError,
    setIncomeSimError,
    incomeAiInsights,
    setIncomeAiInsights,
    incomeSimLoading,
    setIncomeSimLoading,
    incomeInsightLoading,
    setIncomeInsightLoading,

    // Cross Statement simulation state
    crossSimValues,
    setCrossSimValues,
    crossSimResults,
    setCrossSimResults,
    crossSimError,
    setCrossSimError,
    crossAiInsights,
    setCrossAiInsights,
    crossSimLoading,
    setCrossSimLoading,
    crossInsightLoading,
    setCrossInsightLoading,

    // Utility functions
    clearPersistedState
  };
};
