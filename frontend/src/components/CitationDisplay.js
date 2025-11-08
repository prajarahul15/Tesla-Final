import React from 'react';

/**
 * CitationDisplay Component
 * Displays verified citations from AI responses
 * 
 * @param {Object} props
 * @param {Array} props.citations - Array of citation objects from backend
 * @param {Object} props.citationData - Full citation data object from backend
 */
const CitationDisplay = ({ citations, citationData }) => {
  // Extract citations from citationData if provided
  const verifiedCitations = citations || citationData?.verified_citations || [];
  const totalCitations = citationData?.total_citations || verifiedCitations.length;
  const verifiedCount = citationData?.verified_count || verifiedCitations.length;
  
  if (!verifiedCitations || verifiedCitations.length === 0) {
    return null;
  }

  const getCitationBadgeColor = (confidence) => {
    if (confidence >= 0.8) return 'bg-green-100 text-green-800 border-green-300';
    if (confidence >= 0.6) return 'bg-yellow-100 text-yellow-800 border-yellow-300';
    return 'bg-gray-100 text-gray-800 border-gray-300';
  };

  const formatSourceName = (source) => {
    if (!source) return 'Unknown Source';
    // Remove .pdf extension and clean up filename
    return source.replace('.pdf', '').replace(/[-_]/g, ' ');
  };

  return (
    <div className="mt-4 p-4 bg-white rounded-lg border border-gray-200 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <h4 className="text-sm font-semibold text-gray-700">
            Sources Cited
          </h4>
        </div>
        <span className="text-xs text-gray-500">
          {verifiedCount}/{totalCitations} verified
        </span>
      </div>
      
      <div className="space-y-2">
        {verifiedCitations.map((citation, index) => {
          const confidence = citation.confidence || citation.match_score || 0;
          const source = citation.source || 'Unknown';
          const docType = citation.type || 'Document';
          const year = citation.year || '';
          
          return (
            <div
              key={index}
              className={`p-3 rounded-md border ${getCitationBadgeColor(confidence)}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-2 mb-1">
                    <span className="text-xs font-mono bg-white px-1.5 py-0.5 rounded">
                      [{index + 1}]
                    </span>
                    <span className="text-sm font-medium">
                      {docType}
                      {year && ` (${year})`}
                    </span>
                    {confidence >= 0.8 && (
                      <span className="text-xs text-green-700 flex items-center">
                        <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                        Verified
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-gray-600 mt-1">
                    Source: {formatSourceName(source)}
                  </div>
                  {confidence > 0 && (
                    <div className="text-xs text-gray-500 mt-1">
                      Confidence: {(confidence * 100).toFixed(0)}%
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
      
      {citationData?.unverified_citations && citationData.unverified_citations.length > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-200">
          <div className="text-xs text-gray-500 mb-2">
            Unverified Citations ({citationData.unverified_count}):
          </div>
          <div className="space-y-1">
            {citationData.unverified_citations.map((citation, index) => (
              <div key={index} className="text-xs text-gray-400 italic">
                • {citation.source || citation.text || 'Unknown'}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default CitationDisplay;


