import React from 'react';

/**
 * TemporalContextBadge Component
 * Displays temporal query context when detected
 * 
 * @param {Object} props
 * @param {Object} props.temporalContext - Temporal information from backend
 */
const TemporalContextBadge = ({ temporalContext }) => {
  if (!temporalContext || !temporalContext.has_temporal) {
    return null;
  }

  const getTemporalTypeLabel = (type) => {
    const labels = {
      comparison: 'Year Comparison',
      trend: 'Trend Analysis',
      relative: 'Relative Time',
      range: 'Date Range',
      yoy: 'Year-over-Year'
    };
    return labels[type] || 'Temporal Analysis';
  };

  const getTemporalIcon = (type) => {
    if (type === 'comparison') {
      return (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      );
    }
    if (type === 'trend') {
      return (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
        </svg>
      );
    }
    return (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    );
  };

  const formatYears = (years) => {
    if (!years || years.length === 0) return '';
    if (years.length === 1) return `${years[0]}`;
    if (years.length === 2) return `${years[0]} vs ${years[1]}`;
    if (years.length <= 5) return years.join(', ');
    return `${years[0]}-${years[years.length - 1]}`;
  };

  const type = temporalContext.type;
  const years = temporalContext.years || [];
  const quarters = temporalContext.quarters || [];

  return (
    <div className="mb-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
      <div className="flex items-center space-x-2">
        <div className="flex-shrink-0">
          {getTemporalIcon(type)}
        </div>
        <div className="flex-1">
          <div className="flex items-center space-x-2">
            <span className="text-xs font-semibold text-blue-700">
              {getTemporalTypeLabel(type)}
            </span>
          </div>
          {years.length > 0 && (
            <div className="text-xs text-blue-600 mt-1">
              Time Periods: {formatYears(years)}
            </div>
          )}
          {quarters.length > 0 && (
            <div className="text-xs text-blue-600 mt-1">
              Quarters: {quarters.join(', ')}
            </div>
          )}
          {temporalContext.comparison && (
            <div className="text-xs text-blue-500 mt-1 italic">
              Side-by-side comparison provided
            </div>
          )}
          {temporalContext.trend_analysis && (
            <div className="text-xs text-blue-500 mt-1 italic">
              Trend analysis included
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TemporalContextBadge;


