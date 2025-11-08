import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Custom RadarChart component for feature importance (rectangular)
function RadarChart({ data, width = 400, height = 250 }) {
  const centerX = width / 2;
  const centerY = height / 2;
  const radiusX = width * 0.35;
  const radiusY = height * 0.35;
  const numAxes = data.length;
  
  // Generate points for each axis (elliptical)
  const points = data.map((item, index) => {
    const angle = (index * 2 * Math.PI) / numAxes - Math.PI / 2;
    const valueX = item.importance * radiusX;
    const valueY = item.importance * radiusY;
    const x = centerX + valueX * Math.cos(angle);
    const y = centerY + valueY * Math.sin(angle);
    return { x, y, label: item.feature, value: item.importance, angle };
  });
  
  // Create path for the radar shape
  const pathData = points.map((point, index) => 
    `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`
  ).join(' ') + ' Z';
  
  return (
    <div className="flex flex-col items-center">
      <svg width={width} height={height} className="border rounded-lg bg-gray-50">
        {/* Grid ellipses */}
        {[0.2, 0.4, 0.6, 0.8, 1.0].map((scale, i) => (
          <ellipse
            key={i}
            cx={centerX}
            cy={centerY}
            rx={radiusX * scale}
            ry={radiusY * scale}
            fill="none"
            stroke="#e5e7eb"
            strokeWidth="1"
          />
        ))}
        
        {/* Axes lines */}
        {data.map((_, index) => {
          const angle = (index * 2 * Math.PI) / numAxes - Math.PI / 2;
          const x = centerX + radiusX * Math.cos(angle);
          const y = centerY + radiusY * Math.sin(angle);
          return (
            <line
              key={index}
              x1={centerX}
              y1={centerY}
              x2={x}
              y2={y}
              stroke="#d1d5db"
              strokeWidth="1"
            />
          );
        })}
        
        {/* Radar shape */}
        <path
          d={pathData}
          fill="rgba(147, 51, 234, 0.2)"
          stroke="#9333ea"
          strokeWidth="2"
        />
        
        {/* Data points */}
        {points.map((point, index) => (
          <circle
            key={index}
            cx={point.x}
            cy={point.y}
            r="4"
            fill="#9333ea"
          />
        ))}
        
        {/* Labels */}
        {points.map((point, index) => {
          const angle = point.angle;
          const labelRadiusX = radiusX + 30;
          const labelRadiusY = radiusY + 20;
          const labelX = centerX + labelRadiusX * Math.cos(angle);
          const labelY = centerY + labelRadiusY * Math.sin(angle);
          
          return (
            <text
              key={index}
              x={labelX}
              y={labelY}
              textAnchor="middle"
              dominantBaseline="middle"
              className="text-xs font-medium fill-gray-700"
            >
              {point.label}
            </text>
          );
        })}
      </svg>
      
    </div>
  );
}

// Custom BarChart component for feature importance
function BarChart({ data, width = 400, height = 250 }) {
  const padding = { top: 20, right: 20, bottom: 60, left: 50 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  
  const maxValue = Math.max(...data.map(d => d.importance));
  const barWidth = chartWidth / data.length * 0.8;
  const barSpacing = chartWidth / data.length;
  
  return (
    <svg width={width} height={height} className="border rounded-lg bg-white">
      {/* Y-axis grid lines */}
      {[0, 0.25, 0.5, 0.75, 1.0].map((tick, i) => {
        const y = padding.top + chartHeight - (tick * chartHeight);
        return (
          <g key={i}>
            <line
              x1={padding.left}
              y1={y}
              x2={padding.left + chartWidth}
              y2={y}
              stroke="#e5e7eb"
              strokeWidth="1"
            />
            <text
              x={padding.left - 5}
              y={y}
              textAnchor="end"
              dominantBaseline="middle"
              className="text-xs fill-gray-600"
            >
              {(tick * 100).toFixed(0)}%
            </text>
          </g>
        );
      })}
      
      {/* Bars */}
      {data.map((item, index) => {
        const barHeight = (item.importance / maxValue) * chartHeight;
        const x = padding.left + (index * barSpacing) + (barSpacing - barWidth) / 2;
        const y = padding.top + chartHeight - barHeight;
        
        return (
          <g key={index}>
            <rect
              x={x}
              y={y}
              width={barWidth}
              height={barHeight}
              fill="#9333ea"
              className="hover:fill-purple-700 transition-colors"
            />
            <text
              x={x + barWidth / 2}
              y={y - 5}
              textAnchor="middle"
              className="text-xs font-medium fill-gray-700"
            >
              {(item.importance * 100).toFixed(1)}%
            </text>
          </g>
        );
      })}
      
      {/* X-axis labels */}
      {data.map((item, index) => {
        const x = padding.left + (index * barSpacing) + barSpacing / 2;
        const y = padding.top + chartHeight + 15;
        
        return (
          <text
            key={index}
            x={x}
            y={y}
            textAnchor="end"
            transform={`rotate(-45, ${x}, ${y})`}
            className="text-xs fill-gray-700"
          >
            {item.feature}
          </text>
        );
      })}
      
      {/* X-axis line */}
      <line
        x1={padding.left}
        y1={padding.top + chartHeight}
        x2={padding.left + chartWidth}
        y2={padding.top + chartHeight}
        stroke="#374151"
        strokeWidth="2"
      />
      
      {/* Y-axis line */}
      <line
        x1={padding.left}
        y1={padding.top}
        x2={padding.left}
        y2={padding.top + chartHeight}
        stroke="#374151"
        strokeWidth="2"
      />
    </svg>
  );
}

// Custom LineChart component (enhanced for continuous forecasting)
function LineChart({ series, xLabels = [], height=220, padding=32, showXLabels=true, forecastData=null }) {
  const containerRef = useRef(null);
  const [w, setW] = useState(600);
  const [hoverX, setHoverX] = useState(null);
  const [tooltip, setTooltip] = useState(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const update = () => setW(el.clientWidth || 600);
    update();
    if (typeof ResizeObserver !== 'undefined') {
      const ro = new ResizeObserver(update);
      ro.observe(el);
      window.addEventListener('resize', update);
      return () => { try { ro.disconnect(); } catch {} window.removeEventListener('resize', update); };
    } else {
      window.addEventListener('resize', update);
      return () => window.removeEventListener('resize', update);
    }
  }, []);

  const width = Math.max(300, w);
  const allPoints = series.flatMap(s => s.points);
  const minY = Math.min(...allPoints.filter(p => p !== null));
  const maxY = Math.max(...allPoints.filter(p => p !== null));
  const range = maxY - minY;
  const yMin = minY - range * 0.1;
  const yMax = maxY + range * 0.1;

  const chartArea = {
    left: padding,
    right: width - padding,
    top: padding,
    bottom: height - padding,
    width: width - 2 * padding,
    height: height - 2 * padding
  };

  const xScale = (i) => chartArea.left + (i / Math.max(1, xLabels.length - 1)) * chartArea.width;
  const yScale = (val) => chartArea.bottom - ((val - yMin) / (yMax - yMin)) * chartArea.height;

  const handleMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const relativeX = (x - chartArea.left) / chartArea.width;
    const index = Math.round(relativeX * (xLabels.length - 1));
    
    if (index >= 0 && index < xLabels.length) {
      setHoverX(index);
      
      // Multi-series tooltip for forecasting data
      const tooltipData = series.map(s => ({
        name: s.name,
        value: s.points[index],
        color: s.color
      })).filter(d => d.value !== null);
      
      if (tooltipData.length > 0) {
        const rect = e.currentTarget.getBoundingClientRect();
        setTooltip({ 
          x: e.clientX, 
          y: e.clientY,
          chartX: e.clientX - rect.left,
          chartY: e.clientY - rect.top,
          data: tooltipData, 
          label: xLabels[index] 
        });
      }
    }
  };

  return (
    <div ref={containerRef} className="relative">
      <svg 
        width={width} 
        height={height} 
        onMouseMove={handleMouseMove}
        onMouseLeave={() => { setHoverX(null); setTooltip(null); }}
        className="cursor-crosshair"
      >
        {/* Horizontal grid lines */}
        {Array.from({length: 5}, (_, i) => (
          <line
            key={i}
            x1={chartArea.left}
            y1={chartArea.bottom - (i * chartArea.height / 4)}
            x2={chartArea.right}
            y2={chartArea.bottom - (i * chartArea.height / 4)}
            stroke="#e5e7eb"
            strokeWidth="1"
            strokeDasharray="2,2"
          />
        ))}
        
        {/* Chart border */}
        <rect
          x={chartArea.left}
          y={chartArea.top}
          width={chartArea.width}
          height={chartArea.height}
          fill="none"
          stroke="#d1d5db"
          strokeWidth="1"
        />
        
        {/* Hover line */}
        {hoverX !== null && (
          <line
            x1={xScale(hoverX)}
            y1={chartArea.top}
            x2={xScale(hoverX)}
            y2={chartArea.bottom}
            stroke="#6b7280"
            strokeWidth="1"
            strokeDasharray="3,3"
          />
        )}
        
        {/* Data series - multiple lines for actual, test, and forecast */}
        {series.map((s, idx) => {
          // Create path data for each series
          const pathData = s.points
            .map((point, i) => {
              if (point !== null) {
                return `${i === 0 || s.points[i-1] === null ? 'M' : 'L'} ${xScale(i)} ${yScale(point)}`;
              }
              return '';
            })
            .filter(Boolean)
            .join(' ');
          
          return (
            <g key={idx}>
              <path
                d={pathData}
                fill="none"
                stroke={s.color}
                strokeWidth="2"
                strokeDasharray={s.dashArray || "none"}
              />
              {/* Data points removed for cleaner line chart */}
            </g>
          );
        })}
        
        {/* X-axis labels - conditional - show only unique years */}
        {showXLabels && (() => {
          // Extract unique years and their positions
          const yearPositions = {};
          xLabels.forEach((label, i) => {
            if (label) {
              const year = label.split('-')[0];
              if (!yearPositions[year]) {
                yearPositions[year] = i;
              }
            }
          });
          
          return Object.entries(yearPositions).map(([year, position]) => (
            <text
              key={year}
              x={xScale(position)}
              y={height - 5}
              textAnchor="middle"
              fontSize="10"
              fill="#6b7280"
              fontWeight="500"
            >
              {year}
            </text>
          ));
        })()}
      </svg>
      
      {/* Tooltip */}
      {tooltip && (
        <div 
          className="absolute bg-white shadow-lg border rounded px-3 py-2 text-xs pointer-events-none z-20"
          style={{ 
            left: Math.min(Math.max(tooltip.chartX - 60, 0), w - 120), 
            top: Math.max(tooltip.chartY - 60, 10),
            minWidth: '120px'
          }}
        >
          <div className="font-semibold text-gray-800 mb-1">{tooltip.label}</div>
          {tooltip.data.map((d, i) => (
            <div key={i} className="flex justify-between items-center">
              <span style={{ color: d.color }} className="font-medium">{d.name}:</span>
              <span className="ml-2 text-gray-800 font-medium">{typeof d.value === 'number' ? d.value.toLocaleString(undefined, {maximumFractionDigits: 0}) : d.value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const MetricForecasting = ({ scenario }) => {
  const [selectedMetric, setSelectedMetric] = useState('revenue_millions');
  const [forecastType, setForecastType] = useState('univariate');
  const [selectedMetrics, setSelectedMetrics] = useState(['revenue_millions']);
  const [forecastHorizon, setForecastHorizon] = useState(12);
  const [testPeriod, setTestPeriod] = useState(6);
  const [forecastResults, setForecastResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const hasAutoGeneratedRef = useRef(false);

  // Available metrics for forecasting
  const availableMetrics = [
    { id: 'revenue_millions', name: 'Revenue', unit: 'millions USD', color: '#3B82F6' },
    { id: 'cogs_millions', name: 'Cost of Goods Sold', unit: 'millions USD', color: '#EF4444' },
    { id: 'rd_expense_millions', name: 'R&D Expense', unit: 'millions USD', color: '#10B981' },
    { id: 'sga_expense_millions', name: 'SG&A Expense', unit: 'millions USD', color: '#F59E0B' }
  ];

  const generateForecast = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const endpoint = forecastType === 'univariate' ? '/forecast/univariate' : '/forecast/multivariate';
      const payload = forecastType === 'univariate' 
        ? { metric: selectedMetric, horizon: forecastHorizon, test_months: testPeriod }
        : { metrics: selectedMetrics, horizon: forecastHorizon, test_months: testPeriod };
      
      const response = await axios.post(`${API}${endpoint}`, payload);
      setForecastResults(response.data);
    } catch (err) {
      setError('Failed to generate forecast. Please try again.');
      console.error('Forecast error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleMetricToggle = (metricId) => {
    setSelectedMetrics(prev => 
      prev.includes(metricId) 
        ? prev.filter(id => id !== metricId)
        : [...prev, metricId]
    );
  };

  const selectAllMetrics = () => {
    setSelectedMetrics(availableMetrics.map(m => m.id));
  };

  const clearAllMetrics = () => {
    setSelectedMetrics([]);
  };

  // Auto-generate forecast when component mounts (like VehicleForecast)
  useEffect(() => {
    // Only auto-generate once on initial mount
    if (hasAutoGeneratedRef.current) return;
    
    const autoGenerateForecast = async () => {
      hasAutoGeneratedRef.current = true;
      
      // For multivariate on initial load, select all metrics to show comprehensive analysis
      if (forecastType === 'multivariate') {
        const allMetricIds = availableMetrics.map(m => m.id);
        setSelectedMetrics(allMetricIds);
        // Wait for state update, then generate
        setTimeout(async () => {
          setLoading(true);
          setError(null);
          try {
            const endpoint = '/forecast/multivariate';
            const payload = { metrics: allMetricIds, horizon: forecastHorizon, test_months: testPeriod };
            const response = await axios.post(`${API}${endpoint}`, payload);
            setForecastResults(response.data);
          } catch (err) {
            setError('Failed to generate forecast. Please try again.');
            console.error('Forecast error:', err);
          } finally {
            setLoading(false);
          }
        }, 100);
        return;
      }
      
      // For univariate, generate forecast automatically with default values
      setLoading(true);
      setError(null);
      try {
        const endpoint = '/forecast/univariate';
        const payload = { metric: selectedMetric, horizon: forecastHorizon, test_months: testPeriod };
        const response = await axios.post(`${API}${endpoint}`, payload);
        setForecastResults(response.data);
      } catch (err) {
        setError('Failed to generate forecast. Please try again.');
        console.error('Forecast error:', err);
      } finally {
        setLoading(false);
      }
    };
    
    autoGenerateForecast();
  }, []); // Only run once on mount

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-blue-600 text-white p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-2">🔮 Metric Forecasting</h2>
        <p className="text-purple-100">
          Advanced time series forecasting for Tesla's key financial metrics using machine learning
        </p>
      </div>

      {/* Parameter Selection */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">📊 Metric Selection</h3>
        
        {/* Forecast Type Toggle */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">Forecast Type</label>
          <div className="flex space-x-4">
            <label className="flex items-center">
              <input
                type="radio"
                value="univariate"
                checked={forecastType === 'univariate'}
                onChange={(e) => setForecastType(e.target.value)}
                className="mr-2"
              />
              <span className="text-sm">Univariate (Single Metric)</span>
            </label>
            <label className="flex items-center">
              <input
                type="radio"
                value="multivariate"
                checked={forecastType === 'multivariate'}
                onChange={(e) => setForecastType(e.target.value)}
                className="mr-2"
              />
              <span className="text-sm">Multivariate (Multiple Metrics)</span>
            </label>
          </div>
        </div>

        {/* Single Metric Selection */}
        {forecastType === 'univariate' && (
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">Select Metric</label>
            <select
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
              className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            >
              {availableMetrics.map(metric => (
                <option key={metric.id} value={metric.id}>
                  {metric.name} ({metric.unit})
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Multiple Metrics Selection */}
        {forecastType === 'multivariate' && (
          <div className="mb-6">
            <div className="flex justify-between items-center mb-3">
              <label className="block text-sm font-medium text-gray-700">Select Metrics</label>
              <div className="space-x-2">
                <button
                  onClick={selectAllMetrics}
                  className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
                >
                  Select All
                </button>
                <button
                  onClick={clearAllMetrics}
                  className="text-xs px-2 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
                >
                  Clear All
                </button>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {availableMetrics.map(metric => (
                <label key={metric.id} className="flex items-center p-3 border rounded-lg hover:bg-gray-50">
                  <input
                    type="checkbox"
                    checked={selectedMetrics.includes(metric.id)}
                    onChange={() => handleMetricToggle(metric.id)}
                    className="mr-3"
                  />
                  <div>
                    <div className="font-medium text-sm">{metric.name}</div>
                    <div className="text-xs text-gray-500">{metric.unit}</div>
                  </div>
                </label>
              ))}
            </div>
          </div>
        )}

        {/* Configuration */}
        <div className="grid grid-cols-2 gap-6 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Forecast Horizon</label>
            <select
              value={forecastHorizon}
              onChange={(e) => setForecastHorizon(Number(e.target.value))}
              className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            >
              <option value={6}>6 months</option>
              <option value={12}>12 months</option>
              <option value={18}>18 months</option>
              <option value={24}>24 months</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Test Period</label>
            <select
              value={testPeriod}
              onChange={(e) => setTestPeriod(Number(e.target.value))}
              className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            >
              <option value={3}>3 months</option>
              <option value={6}>6 months</option>
              <option value={9}>9 months</option>
              <option value={12}>12 months</option>
            </select>
          </div>
        </div>

        {/* Generate Button */}
        <button
          onClick={generateForecast}
          disabled={loading || (forecastType === 'multivariate' && selectedMetrics.length === 0)}
          className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? (
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
              Generating Forecast...
            </div>
          ) : (
            '🔮 Generate Forecast'
          )}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      {/* Results Section */}
      {forecastResults && (() => {
        // Extract data based on forecast type
        let displayData;
        let allMetricsData = {};
        if (forecastResults.forecast_type === 'multivariate') {
          // For multivariate, get data from all metrics in results
          allMetricsData = forecastResults.results || {};
          const firstMetric = Object.keys(allMetricsData)[0];
          displayData = firstMetric ? allMetricsData[firstMetric] : {};
        } else {
          // For univariate, data is at root level
          displayData = forecastResults;
        }
        
        // Helper function to get metric display name
        const getMetricDisplayName = () => {
          if (forecastResults.forecast_type === 'multivariate') {
            // For multivariate, show all selected metrics
            const metricNames = selectedMetrics
              .map(metricId => {
                const metric = availableMetrics.find(m => m.id === metricId);
                return metric ? `${metric.name} (${metric.unit})` : null;
              })
              .filter(Boolean);
            return metricNames.length > 0 ? `Multivariate: ${metricNames.join(', ')}` : 'Multivariate';
          } else {
            // For univariate, show the selected metric
            const metric = availableMetrics.find(m => m.id === selectedMetric);
            return metric ? `${metric.name} (${metric.unit})` : 'Metric';
          }
        };
        
        const metricDisplayName = getMetricDisplayName();
        
        return (
        <div className="space-y-6">


          {/* Model Performance - Match Vehicle Forecast Format */}
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Model Performance <span className="text-sm font-normal text-gray-600">({metricDisplayName})</span>
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-blue-50 p-3 rounded">
                <div className="text-sm text-blue-800">MAE (test)</div>
                <div className="text-2xl font-bold text-blue-700">{(displayData.accuracy?.mae || 0).toFixed(0)}</div>
              </div>
              <div className="bg-green-50 p-3 rounded">
                <div className="text-sm text-green-800">MAPE (test)</div>
                <div className="text-2xl font-bold text-green-700">{(displayData.accuracy?.mape || 0).toFixed(1)}%</div>
              </div>
              <div className="bg-purple-50 p-3 rounded">
                <div className="text-sm text-purple-800">Total Forecast</div>
                <div className="text-2xl font-bold text-purple-700">
                  {displayData.forecast_table?.reduce((sum, row) => sum + (row.forecast || 0), 0).toFixed(0)}
                </div>
              </div>
            </div>

            {/* Legend */}
            <div className="mb-4">
              <div className="flex flex-wrap items-center gap-4 text-sm">
                {forecastResults.forecast_type === 'multivariate' ? 
                  // For multivariate: show all metrics with their series types
                  Object.entries(allMetricsData).flatMap(([metricName, metricData], index) => {
                    const colors = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6", "#06B6D4", "#84CC16", "#F97316"];
                    const baseColor = colors[index % colors.length];
                    const metricDisplayName = metricName.replace('_millions', '').replace(/_/g, ' ').toUpperCase();
                    
                    return [
                      {
                        name: `${metricDisplayName} (Actual)`,
                        color: baseColor,
                        dash: false
                      },
                      {
                        name: `${metricDisplayName} (Test)`,
                        color: baseColor,
                        dash: true
                      },
                      {
                        name: `${metricDisplayName} (Forecast)`,
                        color: baseColor,
                        dash: true
                      }
                    ].map((item, subIndex) => (
                      <div key={`${index}-${subIndex}`} className="flex items-center">
                        <div 
                          className="w-4 h-0.5 mr-2" 
                          style={{ 
                            backgroundColor: item.dash ? 'transparent' : item.color,
                            borderTop: item.dash ? `2px dashed ${item.color}` : 'none'
                          }}
                        ></div>
                        <span className="text-xs">{item.name}</span>
                      </div>
                    ));
                  }) :
                  // For univariate: show standard legend
                  [
                    { color: "#3B82F6", label: "History (Actual)" },
                    { color: "#F59E0B", label: "Test (Pred)", dash: true },
                    { color: "#EF4444", label: "Forecast", dash: true }
                  ].map((item, index) => (
                    <div key={index} className="flex items-center">
                      <div 
                        className="w-4 h-0.5 mr-2" 
                        style={{ 
                          backgroundColor: item.color,
                          borderTop: item.dash ? `2px dashed ${item.color}` : 'none'
                        }}
                      ></div>
                      <span>{item.label}</span>
                    </div>
                  ))
                }
              </div>
            </div>

            {/* Chart and Table Layout - Match Vehicle Forecast */}
            <div className="grid grid-cols-1 md:grid-cols-10 gap-6 mt-6">
              {/* Chart - 70% */}
              <div className="md:col-span-7">
              <LineChart
                series={forecastResults.forecast_type === 'multivariate' ? 
                  // For multivariate: show all metrics with actual, test, and forecast data
                  Object.entries(allMetricsData).flatMap(([metricName, metricData], index) => {
                    const colors = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6", "#06B6D4", "#84CC16", "#F97316"];
                    const baseColor = colors[index % colors.length];
                    const metricDisplayName = metricName.replace('_millions', '').replace(/_/g, ' ').toUpperCase();
                    
                    return [
                      {
                        name: `${metricDisplayName} (Actual)`,
                        points: metricData.plot_data?.map(d => {
                          if (d.actual !== null && d.actual !== undefined && d.actual !== '') return Number(d.actual);
                          return null;
                        }) || [],
                        color: baseColor
                      },
                      {
                        name: `${metricDisplayName} (Test Pred)`,
                        points: metricData.plot_data?.map(d => {
                          if (d.test_forecast !== null && d.test_forecast !== undefined && d.test_forecast !== '') return Number(d.test_forecast);
                          return null;
                        }) || [],
                        color: baseColor,
                        dashArray: "5,5"
                      },
                      {
                        name: `${metricDisplayName} (Forecast)`,
                        points: metricData.plot_data?.map(d => {
                          if (d.future_forecast !== null && d.future_forecast !== undefined && d.future_forecast !== '') return Number(d.future_forecast);
                          return null;
                        }) || [],
                        color: baseColor,
                        dashArray: "8,4"
                      }
                    ];
                  }) :
                  // For univariate: show the standard three series
                  [
                    {
                      name: "History (Actual)",
                      points: displayData.plot_data?.map(d => {
                        // Show actual values for all periods where available
                        if (d.actual !== null && d.actual !== undefined && d.actual !== '') return Number(d.actual);
                        return null;
                      }) || [],
                      color: "#3B82F6"
                    },
                    {
                      name: "Test (Pred)",
                      points: displayData.plot_data?.map(d => {
                        // Show test predictions only for test period
                        if (d.test_forecast !== null && d.test_forecast !== undefined && d.test_forecast !== '') return Number(d.test_forecast);
                        return null;
                      }) || [],
                      color: "#F59E0B",
                      dashArray: "5,5"
                    },
                    {
                      name: "Forecast",
                      points: displayData.plot_data?.map(d => {
                        // Show future forecasts only for future period
                        if (d.future_forecast !== null && d.future_forecast !== undefined && d.future_forecast !== '') return Number(d.future_forecast);
                        return null;
                      }) || [],
                      color: "#EF4444",
                      dashArray: "8,4"
                    }
                  ]
                }
                xLabels={displayData.plot_data?.map(d => d.date) || []}
                height={275}
                showXLabels={false}
                forecastData={displayData.plot_data || []}
              />
              </div>

              {/* Table - 30% */}
              <div className="md:col-span-3">
                <div className="h-72 overflow-y-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-100 sticky top-0 border-b-2 border-gray-200">
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-bold text-gray-700 uppercase tracking-wider whitespace-nowrap">
                          📅 Date
                        </th>
                        <th className="px-4 py-3 text-right text-xs font-bold text-gray-700 uppercase tracking-wider">
                          📊 Actual
                        </th>
                        <th className="px-4 py-3 text-right text-xs font-bold text-blue-600 uppercase tracking-wider">
                          🎯 Test Prediction
                        </th>
                        <th className="px-4 py-3 text-right text-xs font-bold text-gray-700 uppercase tracking-wider">
                          📈 Accuracy (MAPE)
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {/* Test Period Data - Show all three columns: Actual, Test Prediction, Accuracy (MAPE) */}
                      {(() => {
                        const testRows = displayData.plot_data?.filter(row => {
                          const testVal = (row.test_forecast !== null && row.test_forecast !== undefined && row.test_forecast !== '');
                          return testVal;
                        }) || [];
                        console.log(`Found ${testRows.length} test period rows:`, testRows);
                        return testRows;
                      })().map((row, idx) => {
                        // Parse values carefully
                        const actualVal = (row.actual !== null && row.actual !== undefined && row.actual !== '') ? Number(row.actual) : null;
                        const testVal = (row.test_forecast !== null && row.test_forecast !== undefined && row.test_forecast !== '') ? Number(row.test_forecast) : null;
                        
                        // Calculate MAPE accuracy if both values are available
                        let accuracy = null;
                        if (actualVal !== null && testVal !== null && actualVal !== 0) {
                          accuracy = Math.abs((actualVal - testVal) / actualVal) * 100;
                        }
                        
                        return (
                          <tr key={`test-${idx}`} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                            {/* Date Column */}
                            <td className="px-4 py-2 text-sm text-gray-900 whitespace-nowrap font-medium">
                              {row.date}
                            </td>
                            {/* Actual Column */}
                            <td className="px-4 py-2 text-sm text-right text-gray-700 font-medium">
                              {actualVal !== null ? actualVal.toLocaleString(undefined, {maximumFractionDigits: 0}) : '-'}
                            </td>
                            {/* Test Prediction Column */}
                            <td className="px-4 py-2 text-sm text-right text-blue-600 font-medium">
                              {testVal !== null ? testVal.toLocaleString(undefined, {maximumFractionDigits: 0}) : '-'}
                            </td>
                            {/* Accuracy (MAPE) Column */}
                            <td className="px-4 py-2 text-sm text-right font-medium">
                              <span className={accuracy !== null ? (accuracy < 5 ? 'text-green-600' : accuracy < 10 ? 'text-orange-500' : 'text-red-500') : 'text-gray-400'}>
                                {accuracy !== null ? accuracy.toFixed(1) + '%' : '-'}
                              </span>
                            </td>
                          </tr>
                        );
                      }).filter(Boolean)}
                    </tbody>
                  </table>
                </div>
                {/* Debug Info */}
                <div className="mt-2 text-xs text-gray-500 text-center">
                  Test Period: {displayData.plot_data?.filter(row => row.test_forecast !== null && row.test_forecast !== undefined && row.test_forecast !== '').length || 0} months
                </div>
              </div>
            </div>
          </div>

          {/* Macroeconomic Indicators Charts */}
          {forecastResults.forecast_type === 'multivariate' && (
            <div className="bg-white p-6 rounded-lg shadow-sm border">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">📊 Macroeconomic Indicators</h3>
              
              {/* First Row - 3 indicators */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">GDP Growth (%)</h4>
                  <div className="h-48">
                    <LineChart
                      series={[{
                        name: "GDP Growth",
                        points: displayData.plot_data?.map(d => d.gdp_growth || null) || [],
                        color: "#3B82F6"
                      }]}
                      xLabels={displayData.plot_data?.map(d => d.date) || []}
                      height={192}
                      showXLabels={true}
                      forecastData={displayData.plot_data || []}
                    />
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Inflation Rate (%)</h4>
                  <div className="h-48">
                    <LineChart
                      series={[{
                        name: "Inflation Rate",
                        points: displayData.plot_data?.map(d => d.inflation_rate || null) || [],
                        color: "#EF4444"
                      }]}
                      xLabels={displayData.plot_data?.map(d => d.date) || []}
                      height={192}
                      showXLabels={true}
                      forecastData={displayData.plot_data || []}
                    />
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Interest Rate (%)</h4>
                  <div className="h-48">
                    <LineChart
                      series={[{
                        name: "Interest Rate",
                        points: displayData.plot_data?.map(d => d.interest_rate || null) || [],
                        color: "#10B981"
                      }]}
                      xLabels={displayData.plot_data?.map(d => d.date) || []}
                      height={192}
                      showXLabels={true}
                      forecastData={displayData.plot_data || []}
                    />
                  </div>
                </div>
              </div>

              {/* Second Row - 3 indicators */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Unemployment Rate (%)</h4>
                  <div className="h-48">
                    <LineChart
                      series={[{
                        name: "Unemployment Rate",
                        points: displayData.plot_data?.map(d => d.unemployment_rate || null) || [],
                        color: "#F59E0B"
                      }]}
                      xLabels={displayData.plot_data?.map(d => d.date) || []}
                      height={192}
                      showXLabels={true}
                      forecastData={displayData.plot_data || []}
                    />
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Consumer Confidence</h4>
                  <div className="h-48">
                    <LineChart
                      series={[{
                        name: "Consumer Confidence",
                        points: displayData.plot_data?.map(d => d.consumer_confidence || null) || [],
                        color: "#8B5CF6"
                      }]}
                      xLabels={displayData.plot_data?.map(d => d.date) || []}
                      height={192}
                      showXLabels={true}
                      forecastData={displayData.plot_data || []}
                    />
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Oil Price ($/barrel)</h4>
                  <div className="h-48">
                    <LineChart
                      series={[{
                        name: "Oil Price",
                        points: displayData.plot_data?.map(d => d.oil_price || null) || [],
                        color: "#06B6D4"
                      }]}
                      xLabels={displayData.plot_data?.map(d => d.date) || []}
                      height={192}
                      showXLabels={true}
                      forecastData={displayData.plot_data || []}
                    />
                  </div>
                </div>
              </div>

              {/* Third Row - EV Market Size and Variable Importance */}
              <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
                <div className="md:col-span-2">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">EV Market Size (Billions)</h4>
                  <div className="h-48">
                    <LineChart
                      series={[{
                        name: "EV Market Size",
                        points: displayData.plot_data?.map(d => d.ev_market_size || null) || [],
                        color: "#84CC16"
                      }]}
                      xLabels={displayData.plot_data?.map(d => d.date) || []}
                      height={192}
                      showXLabels={true}
                      forecastData={displayData.plot_data || []}
                    />
                  </div>
                </div>
                <div className="md:col-span-3">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Variable Importance Analysis</h4>
                  <div className="h-48 flex items-center justify-center">
                    <RadarChart 
                      data={forecastResults.feature_importance} 
                      width={400}
                      height={192}
                    />
                  </div>
                </div>
              </div>
              
              {/* Fourth Row - Feature Importance Bar Chart */}
              <div className="mt-6">
                <h4 className="text-sm font-medium text-gray-700 mb-2">Feature Importance (Bar Chart)</h4>
                <div className="flex justify-center">
                  <BarChart 
                    data={forecastResults.feature_importance} 
                    width={800}
                    height={300}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Forecast Results Table */}
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              📋 Future Forecast Results <span className="text-sm font-normal text-gray-600">({metricDisplayName})</span>
            </h3>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Month</th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Forecast</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {displayData.forecast_table?.map((row, idx) => (
                    <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{row.date}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">{(row.forecast || 0).toFixed(0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

        </div>
        );
      })()}
    </div>
  );
};

export default MetricForecasting;
