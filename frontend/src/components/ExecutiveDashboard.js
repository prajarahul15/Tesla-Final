import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import FloatingChatButton from './FloatingChatButton';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ExecutiveDashboard = () => {
  const [years, setYears] = useState([]);
  const [selectedYear, setSelectedYear] = useState(null);
  const [summary, setSummary] = useState(null);
  const [energyServicesData, setEnergyServicesData] = useState(null);
  const [segmentData, setSegmentData] = useState(null);
  const [vehicleMeta, setVehicleMeta] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showSegmentTrends, setShowSegmentTrends] = useState(false);

  useEffect(() => {
    fetchYears();
    fetchVehicleMeta();
  }, []);

  useEffect(() => {
    if (selectedYear) {
      fetchSummary(selectedYear);
      fetchEnergyServicesData(selectedYear);
    }
  }, [selectedYear]);

  useEffect(() => {
    // Fetch segment data after summary is loaded (energy/services is optional)
    if (summary) {
      fetchSegmentData();
    }
  }, [summary, energyServicesData]);

  const fetchYears = async () => {
    try {
      const res = await axios.get(`${API}/vehicles/years`);
      const ys = res.data?.years || [];
      setYears(ys);
      if (ys && ys.length) {
        const current = Math.max(...ys);
        setSelectedYear(current);
      }
    } catch (err) {
      setError('Failed to fetch available years');
      setLoading(false);
      console.error('Years fetch error:', err);
    }
  };

  const fetchSummary = async (year) => {
    try {
      setLoading(true);
      const res = await axios.get(`${API}/vehicles/summary/${year}`);
      setSummary(res.data?.summary || null);
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch vehicle summary');
      setLoading(false);
      console.error('Summary fetch error:', err);
    }
  };

  const fetchVehicleMeta = async () => {
    try {
      const resp = await axios.get(`${API}/ai/tesla-agent/vehicle-models`);
      if (resp.data?.success) {
        setVehicleMeta(resp.data.vehicle_models || {});
      }
    } catch (err) {
      console.warn('Vehicle meta fetch error:', err);
    }
  };

  const fetchEnergyServicesData = async (year) => {
    try {
      const res = await axios.get(`${API}/energy-services/summary/${year}`);
      if (res.data?.success) {
        setEnergyServicesData(res.data);
      } else {
        console.warn('Energy/Services data fetch failed:', res.data);
        setEnergyServicesData(null);
      }
    } catch (err) {
      console.warn('Energy/Services fetch error:', err);
      setEnergyServicesData(null);
    }
  };

  const fetchSegmentData = async () => {
    try {
      // Calculate automotive financial totals from summary data
      const models = summary?.models || [];
      const aspValues = Object.values(vehicleMeta || {})
        .map((m) => Number(m?.base_asp || 0))
        .filter((n) => Number.isFinite(n) && n > 0);
      const avgAsp = aspValues.length ? aspValues.reduce((a, b) => a + b, 0) / aspValues.length : 0;
      
      const modelRows = models.map((m) => {
        const key = String(m.model_key);
        const meta = vehicleMeta[key] || {};
        const asp = Number((m.asp !== undefined ? m.asp : (meta?.base_asp ?? avgAsp ?? 0)));
        const margin = Number(meta?.margin_premium ?? 0.19);
        const deliveries = Number(m.deliveries || 0);
        const revenue = m.revenue !== undefined ? Number(m.revenue) : deliveries * asp;
        const cost = revenue * (1 - margin);
        const profit = revenue - cost;
        return { revenue, cost, profit };
      });
      
      const automotiveRevenue = modelRows.reduce((s, r) => s + (r.revenue || 0), 0);
      const automotiveCost = modelRows.reduce((s, r) => s + (r.cost || 0), 0);
      const automotiveProfit = automotiveRevenue - automotiveCost;
      const automotiveMargin = automotiveRevenue > 0 ? (automotiveProfit / automotiveRevenue) : 0;
      
      // Calculate YoY growth and CAGR for Automotive
      let yoyGrowth = null;
      let cagr = null;
      
      try {
        // Fetch previous year data for YoY
        if (selectedYear && selectedYear > Math.min(...years)) {
          const prevYear = selectedYear - 1;
          const prevYearRes = await axios.get(`${API}/vehicles/summary/${prevYear}`);
          const prevYearSummary = prevYearRes.data?.summary;
          
          if (prevYearSummary && prevYearSummary.totals && prevYearSummary.totals.total_revenue) {
            const prevRevenue = prevYearSummary.totals.total_revenue;
            if (prevRevenue > 0 && automotiveRevenue > 0) {
              yoyGrowth = (automotiveRevenue - prevRevenue) / prevRevenue;
            }
          }
        }
        
        // Calculate CAGR for all available historical years
        if (years && years.length > 1) {
          const sortedYears = [...years].sort((a, b) => a - b);
          const firstYear = sortedYears[0];
          const lastYear = selectedYear || sortedYears[sortedYears.length - 1];
          
          if (lastYear > firstYear) {
            // Fetch first year data
            const firstYearRes = await axios.get(`${API}/vehicles/summary/${firstYear}`);
            const firstYearRevenue = firstYearRes.data?.summary?.totals?.total_revenue;
            
            if (firstYearRevenue > 0 && automotiveRevenue > 0) {
              const numYears = lastYear - firstYear;
              cagr = Math.pow(automotiveRevenue / firstYearRevenue, 1 / numYears) - 1;
            }
          }
        }
      } catch (err) {
        console.warn('Error calculating growth metrics:', err);
      }
      
      // Dynamic data from Energy & Services API
      const energyData = energyServicesData?.energy || {};
      const servicesData = energyServicesData?.services || {};
      
      const segmentData = {
        segments: {
          automotive: {
            name: 'Automotive',
            revenue_current: automotiveRevenue,
            margin_current: automotiveMargin,
            yoy_growth: yoyGrowth,
            cagr: cagr,
            description: 'Electric vehicles and automotive sales'
          },
          energy: {
            name: 'Energy & Storage',
            revenue_current: energyData.revenue || 0,
            margin_current: energyData.margin || 0,
            yoy_growth: energyData.yoy_growth !== undefined ? energyData.yoy_growth : null,
            cagr: energyData.cagr !== undefined ? energyData.cagr : null,
            description: 'Solar panels, Powerwall, and energy solutions'
          },
          services: {
            name: 'Services & Other',
            revenue_current: servicesData.revenue || 0,
            margin_current: servicesData.margin || 0,
            yoy_growth: servicesData.yoy_growth !== undefined ? servicesData.yoy_growth : null,
            cagr: servicesData.cagr !== undefined ? servicesData.cagr : null,
            description: 'Service, insurance, and software subscriptions'
          }
        }
      };
      setSegmentData(segmentData);
    } catch (err) {
      console.warn('Segment data fetch error:', err);
    }
  };

  const formatNumber = (v) => {
    if (v === null || v === undefined) return '0';
    try {
      return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(Number(v));
    } catch {
      return String(v);
    }
  };

  const formatCurrencyBillions = (v) => {
    try {
      const value = Number(v || 0);
      const billions = value / 1000000000;
      if (billions >= 1) {
        return `$${billions.toFixed(1)}B`;
      } else if (value >= 1000000) {
        const millions = value / 1000000;
        return `$${millions.toFixed(0)}M`;
      }
      return `$${value.toFixed(0)}`;
    } catch {
      return '$0.0B';
    }
  };

  const formatPercent = (value) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  // Build per-model financials
  const buildModelRows = () => {
    const models = summary?.models || [];
    const aspValues = Object.values(vehicleMeta || {})
      .map((m) => Number(m?.base_asp || 0))
      .filter((n) => Number.isFinite(n) && n > 0);
    const avgAsp = aspValues.length ? aspValues.reduce((a, b) => a + b, 0) / aspValues.length : 0;
    
    return models.map((m) => {
      const key = String(m.model_key);
      const meta = vehicleMeta[key] || {};
      const name = meta?.name || key.replace(/_/g, ' ');
      const asp = Number((m.asp !== undefined ? m.asp : (meta?.base_asp ?? avgAsp ?? 0)));
      const margin = Number(meta?.margin_premium ?? 0.19);
      const sold = Number(m.sold || m.deliveries || 0); // Use sold if available, fallback to deliveries
      const deliveries = Number(m.deliveries || 0);
      const production = Number(m.production || 0);
      const revenue = m.revenue !== undefined ? Number(m.revenue) : deliveries * asp;
      const cost = revenue * (1 - margin);
      const profit = revenue - cost;
      const profitability = revenue > 0 ? (profit / revenue) * 100 : 0;
      return { key, name, sold, deliveries, production, asp, margin, revenue, cost, profit, profitability };
    });
  };

  const modelRows = buildModelRows();
  
  const totals = {
    sold_units: Number((summary?.totals?.total_sold ?? summary?.totals?.total_deliveries) || 0),
    produced_units: Number(summary?.totals?.total_production || 0),
    delivered_units: Number(summary?.totals?.total_deliveries || 0),
  };

  // Automotive-only totals for Per-Model Performance table
  const automotiveTotals = useMemo(() => {
    const automotiveRevenue = modelRows.reduce((s, r) => s + (r.revenue || 0), 0);
    const automotiveCost = modelRows.reduce((s, r) => s + (r.cost || 0), 0);
    const automotiveProfit = automotiveRevenue - automotiveCost;
    const automotiveProfitPct = automotiveRevenue > 0 ? (automotiveProfit / automotiveRevenue) * 100 : 0;
    
    return {
      revenue: automotiveRevenue,
      cost: automotiveCost,
      profit: automotiveProfit,
      profitPct: automotiveProfitPct
    };
  }, [modelRows]);

  const totalsFinancial = useMemo(() => {
    // Automotive revenue and cost (from Excel)
    const automotiveRevenue = modelRows.reduce((s, r) => s + (r.revenue || 0), 0);
    const automotiveCost = modelRows.reduce((s, r) => s + (r.cost || 0), 0);
    
    // Energy & Services revenue and cost (from API)
    const energyRevenue = energyServicesData?.energy?.revenue || 0;
    const energyCogs = energyServicesData?.energy?.cogs || 0;
    const servicesRevenue = energyServicesData?.services?.revenue || 0;
    const servicesCogs = energyServicesData?.services?.cogs || 0;
    
    
    // Total = Automotive + Energy + Services
    const revenue = automotiveRevenue + energyRevenue + servicesRevenue;
    const cost = automotiveCost + energyCogs + servicesCogs;
    const profit = revenue - cost;
    const profitPct = revenue > 0 ? (profit / revenue) * 100 : 0;
    
    return { 
      revenue, 
      cost, 
      profit, 
      profitPct,
      automotiveRevenue,
      energyRevenue,
      servicesRevenue,
      automotiveMargin: automotiveRevenue > 0 ? ((automotiveRevenue - automotiveCost) / automotiveRevenue * 100) : 0,
      energyMargin: energyRevenue > 0 ? ((energyRevenue - energyCogs) / energyRevenue * 100) : 0,
      servicesMargin: servicesRevenue > 0 ? ((servicesRevenue - servicesCogs) / servicesRevenue * 100) : 0
    };
  }, [modelRows, energyServicesData]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading executive dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">{error}</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Floating Chat Button */}
      <FloatingChatButton />
      
      {/* Header with Controls */}
      <div className="bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 text-white p-6 rounded-lg shadow-lg">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-bold mb-2">📊 Executive Dashboard</h2>
            <p className="text-blue-100">Comprehensive view of Tesla's operational and financial performance</p>
          </div>
          <div>
            <label className="block text-xs text-white/80 mb-1">Year</label>
            <select
              value={selectedYear || ''}
              onChange={(e) => setSelectedYear(parseInt(e.target.value))}
              className="px-4 py-2 rounded-lg bg-white text-gray-900 border-2 border-white/30 font-medium"
            >
              {years.map((y) => (
                <option key={y} value={y}>{y}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Section 1: Financial Performance */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
          <span className="w-1 h-6 bg-green-600 mr-3"></span>
          Financial Performance
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gradient-to-br from-sky-50 to-sky-100 p-5 rounded-lg border-2 border-sky-200 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-sky-900">Total Revenue</h4>
              <svg className="w-5 h-5 text-sky-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div className="text-3xl font-bold text-sky-700">{formatCurrencyBillions(totalsFinancial.revenue)}</div>
            <p className="text-xs text-sky-600 mt-1">From all segments</p>
          </div>

          <div className="bg-gradient-to-br from-rose-50 to-rose-100 p-5 rounded-lg border-2 border-rose-200 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-rose-900">Total Cost</h4>
              <svg className="w-5 h-5 text-rose-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
              </svg>
            </div>
            <div className="text-3xl font-bold text-rose-700">{formatCurrencyBillions(totalsFinancial.cost)}</div>
            <p className="text-xs text-rose-600 mt-1">Operating costs</p>
          </div>

          <div className="bg-gradient-to-br from-lime-50 to-lime-100 p-5 rounded-lg border-2 border-lime-200 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-lime-900">Net Profit</h4>
              <svg className="w-5 h-5 text-lime-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
              </svg>
            </div>
            <div className="text-3xl font-bold text-lime-700">{formatCurrencyBillions(totalsFinancial.profit)}</div>
            <p className="text-xs text-lime-600 mt-1">Margin: {totalsFinancial.profitPct.toFixed(1)}%</p>
          </div>
        </div>
      </div>

      {/* Section 2: Business Segments */}
      {segmentData && (
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold text-gray-900 flex items-center">
              <span className="w-1 h-6 bg-purple-600 mr-3"></span>
              Business Segment Performance
            </h3>
            <button
              onClick={() => setShowSegmentTrends(!showSegmentTrends)}
              className="text-sm px-4 py-2 bg-purple-100 text-purple-700 rounded-lg hover:bg-purple-200 transition-colors"
            >
              {showSegmentTrends ? 'Hide' : 'Show'} Trends
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {Object.entries(segmentData.segments).map(([key, segment]) => (
              <div key={key} className={`p-6 rounded-lg border-2 shadow-sm hover:shadow-lg transition-all ${
                key === 'automotive' ? 'bg-gradient-to-br from-blue-50 to-blue-100 border-blue-300' :
                key === 'energy' ? 'bg-gradient-to-br from-green-50 to-green-100 border-green-300' :
                'bg-gradient-to-br from-orange-50 to-orange-100 border-orange-300'
              }`}>
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-lg font-bold text-gray-900">{segment.name}</h4>
                  <span className={`text-2xl ${
                    key === 'automotive' ? 'text-blue-600' :
                    key === 'energy' ? 'text-green-600' :
                    'text-orange-600'
                  }`}>
                    {key === 'automotive' ? '🚗' : key === 'energy' ? '⚡' : '🔧'}
                  </span>
                </div>
                
                <p className="text-xs text-gray-600 mb-4">{segment.description}</p>

                {/* Revenue Metrics */}
                <div className="space-y-3 mb-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-700">Current Revenue</span>
                    <span className="font-bold text-gray-900">{formatCurrencyBillions(segment.revenue_current)}</span>
                  </div>
                </div>

                {/* Margin Metrics */}
                <div className="border-t-2 border-gray-200 pt-3 space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-700">Current Margin</span>
                    <span className="font-semibold">{formatPercent(segment.margin_current)}</span>
                  </div>
                </div>

                {/* Growth Metrics - Same for All Segments */}
                <div className="mt-4 pt-3 border-t-2 border-gray-200 space-y-2">
                  {/* YoY Growth */}
                  {segment.yoy_growth !== null && (
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-700">YoY Growth</span>
                      <div className={`px-3 py-1 rounded-full font-bold text-sm ${
                        segment.yoy_growth >= 0 
                          ? 'bg-green-100 text-green-800' 
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {segment.yoy_growth >= 0 ? '+' : ''}{formatPercent(segment.yoy_growth)}
                      </div>
                    </div>
                  )}
                  {/* CAGR */}
                  {segment.cagr !== null && (
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-700">CAGR (Historical)</span>
                      <div className={`px-3 py-1 rounded-full font-bold text-sm ${
                        segment.cagr >= 0 
                          ? 'bg-blue-100 text-blue-800' 
                          : 'bg-gray-100 text-gray-800'
                      }`}>
                        {segment.cagr >= 0 ? '+' : ''}{formatPercent(segment.cagr)}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Segment Trends - Expandable */}
          {showSegmentTrends && (
            <div className="mt-6 p-6 bg-gradient-to-r from-purple-50 to-indigo-50 rounded-lg border-2 border-purple-200">
              <h4 className="text-lg font-bold text-gray-900 mb-4">📈 Segment Growth Trajectory</h4>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="flex items-center justify-between p-4 bg-green-100 rounded-lg border-2 border-green-300">
                  <div>
                    <h5 className="font-bold text-green-900">Energy & Storage</h5>
                    <p className="text-sm text-green-700">Fastest growing segment</p>
                  </div>
                  <div className="text-3xl font-bold text-green-600">
                    {energyServicesData?.energy?.cagr ? (energyServicesData.energy.cagr * 100).toFixed(1) + '%' : 'N/A'}
                  </div>
                </div>

                <div className="flex items-center justify-between p-4 bg-orange-100 rounded-lg border-2 border-orange-300">
                  <div>
                    <h5 className="font-bold text-orange-900">Services & Other</h5>
                    <p className="text-sm text-orange-700">High-margin recurring revenue</p>
                  </div>
                  <div className="text-3xl font-bold text-orange-600">
                    {energyServicesData?.services?.cagr ? (energyServicesData.services.cagr * 100).toFixed(1) + '%' : 'N/A'}
                  </div>
                </div>

                <div className="flex items-center justify-between p-4 bg-blue-100 rounded-lg border-2 border-blue-300">
                  <div>
                    <h5 className="font-bold text-blue-900">Automotive</h5>
                    <p className="text-sm text-blue-700">Core business with steady growth</p>
                  </div>
                  <div className="text-3xl font-bold text-blue-600">
                    {segmentData?.segments?.automotive?.cagr ? (segmentData.segments.automotive.cagr * 100).toFixed(1) + '%' : 'N/A'}
                  </div>
                </div>
              </div>

              {/* Revenue Mix */}
              <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white p-4 rounded-lg border">
                  <h5 className="font-semibold text-gray-900 mb-3">Revenue Mix Evolution</h5>
                  <div className="space-y-2">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-700">Automotive</span>
                        <span className="font-medium">{(totalsFinancial.automotiveRevenue / totalsFinancial.revenue * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div className="bg-blue-600 h-2 rounded-full" style={{width: `${totalsFinancial.automotiveRevenue / totalsFinancial.revenue * 100}%`}}></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-700">Energy & Storage</span>
                        <span className="font-medium">{(totalsFinancial.energyRevenue / totalsFinancial.revenue * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div className="bg-green-600 h-2 rounded-full" style={{width: `${totalsFinancial.energyRevenue / totalsFinancial.revenue * 100}%`}}></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-700">Services & Other</span>
                        <span className="font-medium">{(totalsFinancial.servicesRevenue / totalsFinancial.revenue * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div className="bg-orange-600 h-2 rounded-full" style={{width: `${totalsFinancial.servicesRevenue / totalsFinancial.revenue * 100}%`}}></div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-white p-4 rounded-lg border">
                  <h5 className="font-semibold text-gray-900 mb-3">Margin Comparison</h5>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-700">Automotive</span>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-500">{totalsFinancial.automotiveMargin.toFixed(1)}%</span>
                        <span className="text-sm">→</span>
                        <span className="text-sm font-bold text-blue-600">22.0%</span>
                        <span className="text-xs text-green-600">+{((22.0 - totalsFinancial.automotiveMargin) * 100).toFixed(0)}bps</span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-700">Energy & Storage</span>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-500">{totalsFinancial.energyMargin.toFixed(1)}%</span>
                        <span className="text-sm">→</span>
                        <span className="text-sm font-bold text-green-600">28.0%</span>
                        <span className="text-xs text-green-600">+{((28.0 - totalsFinancial.energyMargin) * 100).toFixed(0)}bps</span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-700">Services & Other</span>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-500">{totalsFinancial.servicesMargin.toFixed(1)}%</span>
                        <span className="text-sm">→</span>
                        <span className="text-sm font-bold text-orange-600">52.0%</span>
                        <span className="text-xs text-green-600">+{((52.0 - totalsFinancial.servicesMargin) * 100).toFixed(0)}bps</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Section 3: Key Insights & Strategic Focus Areas */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gradient-to-br from-indigo-50 to-indigo-100 p-6 rounded-lg shadow-sm border-2 border-indigo-200">
          <h3 className="text-lg font-bold text-indigo-900 mb-3 flex items-center">
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            Key Insights
          </h3>
          <ul className="space-y-2 text-sm text-indigo-800">
            <li className="flex items-start">
              <span className="text-green-500 mr-2">✓</span>
              <span>Energy segment shows highest growth potential ({energyServicesData?.energy?.cagr ? (energyServicesData.energy.cagr * 100).toFixed(1) : 'N/A'}% CAGR)</span>
            </li>
            <li className="flex items-start">
              <span className="text-green-500 mr-2">✓</span>
              <span>Energy maintains highest margins ({totalsFinancial.energyMargin.toFixed(1)}%) with solar & storage solutions</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-500 mr-2">•</span>
              <span>Automotive remains core revenue driver ({(totalsFinancial.automotiveRevenue / totalsFinancial.revenue * 100).toFixed(1)}% of total)</span>
            </li>
          </ul>
        </div>

        <div className="bg-gradient-to-br from-amber-50 to-amber-100 p-6 rounded-lg shadow-sm border-2 border-amber-200">
          <h3 className="text-lg font-bold text-amber-900 mb-3 flex items-center">
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Strategic Focus Areas
          </h3>
          <ul className="space-y-2 text-sm text-amber-800">
            <li className="flex items-start">
              <span className="text-amber-600 mr-2">→</span>
              <span>Accelerate Energy segment to 20%+ revenue mix (currently {(totalsFinancial.energyRevenue / totalsFinancial.revenue * 100).toFixed(1)}%)</span>
            </li>
            <li className="flex items-start">
              <span className="text-amber-600 mr-2">→</span>
              <span>Expand Services margin to 25%+ through software subscriptions (currently {totalsFinancial.servicesMargin.toFixed(1)}%)</span>
            </li>
            <li className="flex items-start">
              <span className="text-amber-600 mr-2">→</span>
              <span>Drive Automotive margins to 22%+ via cost optimization (currently {totalsFinancial.automotiveMargin.toFixed(1)}%)</span>
            </li>
          </ul>
        </div>
      </div>

      {/* Section 4: Key Operational Metrics */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
          <span className="w-1 h-6 bg-blue-600 mr-3"></span>
          Key Operational Metrics
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-5 rounded-lg border-2 border-blue-200 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-blue-900">Total Deliveries</h4>
              <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
              </svg>
            </div>
            <div className="text-3xl font-bold text-blue-700">{formatNumber(totals.delivered_units)}</div>
            <p className="text-xs text-blue-600 mt-1">Units in {selectedYear}</p>
          </div>

          <div className="bg-gradient-to-br from-emerald-50 to-emerald-100 p-5 rounded-lg border-2 border-emerald-200 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-emerald-900">Total Production</h4>
              <svg className="w-5 h-5 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
              </svg>
            </div>
            <div className="text-3xl font-bold text-emerald-700">{formatNumber(totals.produced_units)}</div>
            <p className="text-xs text-emerald-600 mt-1">Units in {selectedYear}</p>
          </div>

          <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-5 rounded-lg border-2 border-purple-200 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-purple-900">Vehicle Models</h4>
              <svg className="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
              </svg>
            </div>
            <div className="text-3xl font-bold text-purple-700">{(summary?.models || []).length}</div>
            <p className="text-xs text-purple-600 mt-1">Active models</p>
          </div>

          <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-5 rounded-lg border-2 border-orange-200 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-orange-900">Quantity Sold</h4>
              <svg className="w-5 h-5 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            </div>
            <div className="text-3xl font-bold text-orange-700">{formatNumber(totals.sold_units)}</div>
            <p className="text-xs text-orange-600 mt-1">Units in {selectedYear}</p>
          </div>
        </div>
      </div>

      {/* Section 5: Per-Model Breakdown Table */}
      <div className="bg-white rounded-lg shadow-sm border">
        <div className="p-6 border-b flex items-center justify-between">
          <h3 className="text-xl font-bold text-gray-900 flex items-center">
            <span className="w-1 h-6 bg-indigo-600 mr-3"></span>
            Per-Model Performance — {selectedYear}
          </h3>
          <div className="text-sm text-gray-600">
            <span className="font-medium">{modelRows.length}</span> models tracked
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gradient-to-r from-gray-50 to-gray-100">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-bold text-gray-700 uppercase tracking-wider">Model</th>
                <th className="px-6 py-3 text-right text-xs font-bold text-gray-700 uppercase tracking-wider">Quantity Sold</th>
                <th className="px-6 py-3 text-right text-xs font-bold text-gray-700 uppercase tracking-wider">Deliveries</th>
                <th className="px-6 py-3 text-right text-xs font-bold text-gray-700 uppercase tracking-wider">Production</th>
                <th className="px-6 py-3 text-right text-xs font-bold text-gray-700 uppercase tracking-wider">ASP</th>
                <th className="px-6 py-3 text-right text-xs font-bold text-gray-700 uppercase tracking-wider">Revenue</th>
                <th className="px-6 py-3 text-right text-xs font-bold text-gray-700 uppercase tracking-wider">Cost</th>
                <th className="px-6 py-3 text-right text-xs font-bold text-gray-700 uppercase tracking-wider">Profit</th>
                <th className="px-6 py-3 text-right text-xs font-bold text-gray-700 uppercase tracking-wider">Margin %</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {modelRows.map((r, idx) => (
                <tr key={idx} className={`${idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'} hover:bg-blue-50 transition-colors`}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-semibold text-gray-900">{r.name}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">{formatNumber(r.sold)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">{formatNumber(r.deliveries)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">{formatNumber(r.production)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">${formatNumber(r.asp)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-right font-medium text-sky-700">{formatCurrencyBillions(r.revenue)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-right font-medium text-rose-700">{formatCurrencyBillions(r.cost)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-right font-medium text-lime-700">{formatCurrencyBillions(r.profit)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-right">
                    <span className={`px-2 py-1 rounded-full text-xs font-bold ${
                      r.profitability >= 20 ? 'bg-green-100 text-green-800' :
                      r.profitability >= 15 ? 'bg-blue-100 text-blue-800' :
                      r.profitability >= 10 ? 'bg-orange-100 text-orange-800' :
                      'bg-red-100 text-red-800'
                    }`}>
                      {r.profitability.toFixed(1)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
            <tfoot className="bg-gradient-to-r from-gray-100 to-gray-200">
              <tr className="font-bold">
                <td className="px-6 py-4 text-sm text-gray-900">TOTAL</td>
                <td className="px-6 py-4 text-sm text-right text-blue-900">{formatNumber(totals.sold_units)}</td>
                <td className="px-6 py-4 text-sm text-right text-blue-900">{formatNumber(totals.delivered_units)}</td>
                <td className="px-6 py-4 text-sm text-right text-emerald-900">{formatNumber(totals.produced_units)}</td>
                <td className="px-6 py-4 text-sm text-right text-gray-700">-</td>
                <td className="px-6 py-4 text-sm text-right text-sky-900">{formatCurrencyBillions(automotiveTotals.revenue)}</td>
                <td className="px-6 py-4 text-sm text-right text-rose-900">{formatCurrencyBillions(automotiveTotals.cost)}</td>
                <td className="px-6 py-4 text-sm text-right text-lime-900">{formatCurrencyBillions(automotiveTotals.profit)}</td>
                <td className="px-6 py-4 text-sm text-right">
                  <span className="px-2 py-1 bg-indigo-200 text-indigo-900 rounded-full text-xs font-bold">
                    {automotiveTotals.profitPct.toFixed(1)}%
                  </span>
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>
    </div>
  );
};

export default ExecutiveDashboard;

