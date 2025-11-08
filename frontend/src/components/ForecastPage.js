import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ForecastPage = () => {
  const [lineups, setLineups] = useState([]);
  const [selectedLineup, setSelectedLineup] = useState('');
  const [forecastType, setForecastType] = useState('univariate');
  const [monthsAhead, setMonthsAhead] = useState(12);
  const [forecastResult, setForecastResult] = useState(null);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchLineups();
  }, []);

  const fetchLineups = async () => {
    try {
      const response = await axios.get(`${API}/analytics/lineups`);
      setLineups(response.data.lineups);
      if (response.data.lineups.length > 0) {
        setSelectedLineup(response.data.lineups[0].lineup);
      }
    } catch (err) {
      setError('Failed to fetch lineups');
      console.error('Lineups error:', err);
    }
  };

  const generateForecast = async () => {
    if (!selectedLineup) return;

    setLoading(true);
    setError(null);
    setForecastResult(null);

    try {
      const requestData = {
        lineup: selectedLineup,
        forecast_type: forecastType,
        months_ahead: monthsAhead
      };

      const response = await axios.post(`${API}/analytics/forecast`, requestData);
      setForecastResult(response.data.forecast);
    } catch (err) {
      setError('Failed to generate forecast');
      console.error('Forecast error:', err);
    } finally {
      setLoading(false);
    }
  };

  const compareForecasts = async () => {
    if (!selectedLineup) return;

    setLoading(true);
    setError(null);
    setComparisonResult(null);

    try {
      const requestData = {
        lineup: selectedLineup,
        months_ahead: monthsAhead
      };

      const response = await axios.post(`${API}/analytics/compare-forecasts`, requestData);
      setComparisonResult(response.data.comparison);
    } catch (err) {
      setError('Failed to compare forecasts');
      console.error('Comparison error:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (value) => {
    if (!value) return '£0';
    return new Intl.NumberFormat('en-GB', {
      style: 'currency',
      currency: 'GBP',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-GB', {
      year: 'numeric',
      month: 'short'
    });
  };

  const selectedLineupInfo = lineups.find(l => l.lineup === selectedLineup);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-blue-600 text-white p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-2">Advanced Forecasting Analytics</h2>
        <p className="text-purple-100">Univariate and multivariate forecasting with economic variables</p>
      </div>

      {/* Forecast Controls */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Forecast Configuration</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Lineup
            </label>
            <select
              value={selectedLineup}
              onChange={(e) => setSelectedLineup(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {lineups.map((lineup) => (
                <option key={lineup.lineup} value={lineup.lineup}>
                  {lineup.lineup} ({lineup.profile})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Forecast Method
            </label>
            <select
              value={forecastType}
              onChange={(e) => setForecastType(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="univariate">Univariate (Actual Only)</option>
              <option value="multivariate">Multivariate (Economic Variables)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Forecast Horizon
            </label>
            <select
              value={monthsAhead}
              onChange={(e) => setMonthsAhead(parseInt(e.target.value))}
              className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value={6}>6 months</option>
              <option value={12}>12 months</option>
              <option value={18}>18 months</option>
              <option value={24}>24 months</option>
            </select>
          </div>

          <div className="flex items-end">
            <div className="grid grid-cols-1 gap-2 w-full">
              <button
                onClick={generateForecast}
                disabled={loading || !selectedLineup}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-4 py-3 rounded-lg font-medium transition-colors"
              >
                {loading ? 'Generating...' : 'Generate Forecast'}
              </button>
              <button
                onClick={compareForecasts}
                disabled={loading || !selectedLineup}
                className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white px-4 py-3 rounded-lg font-medium transition-colors"
              >
                Compare Methods
              </button>
            </div>
          </div>
        </div>

        {/* Selected Lineup Info */}
        {selectedLineupInfo && (
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <h4 className="font-semibold text-gray-900 mb-2">Selected Lineup Information</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <span className="text-sm text-gray-600">Profile: </span>
                <span className="text-sm font-medium">{selectedLineupInfo.profile}</span>
              </div>
              <div>
                <span className="text-sm text-gray-600">Line Item: </span>
                <span className="text-sm font-medium">{selectedLineupInfo.line_item}</span>
              </div>
              <div>
                <span className="text-sm text-gray-600">Total Actual: </span>
                <span className="text-sm font-medium">{formatCurrency(selectedLineupInfo.total_actual)}</span>
              </div>
              <div>
                <span className="text-sm text-gray-600">Records: </span>
                <span className="text-sm font-medium">{selectedLineupInfo.records}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      {/* Single Forecast Results */}
      {forecastResult && (
        <div className="bg-white rounded-lg shadow-sm border">
          <div className="p-6 border-b">
            <h3 className="text-lg font-semibold text-gray-900">
              Forecast Results - {forecastResult.forecast_type.charAt(0).toUpperCase() + forecastResult.forecast_type.slice(1)}
            </h3>
            <p className="text-sm text-gray-600 mt-1">
              {monthsAhead}-month forecast for {forecastResult.lineup}
            </p>
          </div>
          
          <div className="p-6">
            {/* Forecast Summary */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-semibold text-blue-900 mb-2">Average Forecast</h4>
                <div className="text-2xl font-bold text-blue-600">
                  {formatCurrency(
                    forecastResult.forecasts.reduce((sum, f) => sum + f.forecast, 0) / forecastResult.forecasts.length
                  )}
                </div>
                <p className="text-sm text-blue-700 mt-1">Monthly average</p>
              </div>
              
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-semibold text-green-900 mb-2">Total Forecast</h4>
                <div className="text-2xl font-bold text-green-600">
                  {formatCurrency(
                    forecastResult.forecasts.reduce((sum, f) => sum + f.forecast, 0)
                  )}
                </div>
                <p className="text-sm text-green-700 mt-1">{monthsAhead}-month total</p>
              </div>
              
              <div className="bg-purple-50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-900 mb-2">Model Type</h4>
                <div className="text-lg font-bold text-purple-600">
                  {forecastResult.model_metrics.model_type.toUpperCase()}
                </div>
                <p className="text-sm text-purple-700 mt-1">
                  {forecastResult.model_metrics.model_type === 'multivariate' ? 'With economic variables' : 'Actual values only'}
                </p>
              </div>
            </div>

            {/* Detailed Forecast Table */}
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Month
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Forecast
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Months Ahead
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {forecastResult.forecasts.map((forecast, idx) => (
                    <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {formatDate(forecast.date)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-right font-semibold text-blue-600">
                        {formatCurrency(forecast.forecast)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                        {forecast.month_ahead}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Forecast Comparison Results */}
      {comparisonResult && (
        <div className="bg-white rounded-lg shadow-sm border">
          <div className="p-6 border-b">
            <h3 className="text-lg font-semibold text-gray-900">
              Forecast Methods Comparison
            </h3>
            <p className="text-sm text-gray-600 mt-1">
              Univariate vs Multivariate forecasting for {comparisonResult.lineup}
            </p>
          </div>
          
          <div className="p-6">
            {/* Comparison Summary */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-semibold text-blue-900 mb-2">Univariate Method</h4>
                <div className="text-xl font-bold text-blue-600">
                  {formatCurrency(
                    comparisonResult.comparison_data.reduce((sum, d) => sum + d.univariate_forecast, 0)
                  )}
                </div>
                <p className="text-sm text-blue-700 mt-1">Total forecast (Actual only)</p>
              </div>
              
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-semibold text-green-900 mb-2">Multivariate Method</h4>
                <div className="text-xl font-bold text-green-600">
                  {formatCurrency(
                    comparisonResult.comparison_data.reduce((sum, d) => sum + d.multivariate_forecast, 0)
                  )}
                </div>
                <p className="text-sm text-green-700 mt-1">Total forecast (With economics)</p>
              </div>
            </div>

            {/* Detailed Comparison Table */}
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Month
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Univariate
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Multivariate
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Difference
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {comparisonResult.comparison_data.map((data, idx) => (
                    <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {formatDate(data.date)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-blue-600">
                        {formatCurrency(data.univariate_forecast)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-green-600">
                        {formatCurrency(data.multivariate_forecast)}
                      </td>
                      <td className={`px-6 py-4 whitespace-nowrap text-sm text-right font-semibold ${
                        data.difference >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {data.difference >= 0 ? '+' : ''}{formatCurrency(data.difference)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Feature Importance (if available) */}
            {comparisonResult.multivariate_importance && (
              <div className="mt-8">
                <h4 className="font-semibold text-gray-900 mb-4">Top Feature Importance (Multivariate Model)</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(comparisonResult.multivariate_importance)
                    .sort(([,a], [,b]) => b - a)
                    .slice(0, 8)
                    .map(([feature, importance], idx) => (
                      <div key={idx} className="bg-gray-50 p-3 rounded-lg">
                        <div className="text-sm font-medium text-gray-900">
                          {feature.replace(/_/g, ' ')}
                        </div>
                        <div className="text-lg font-bold text-purple-600">
                          {(importance * 100).toFixed(1)}%
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ForecastPage;