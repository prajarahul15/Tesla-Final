/**
 * Citation Parser Utility
 * Extracts citations from AI response text
 * Fallback if backend citation extraction is unavailable
 */

/**
 * Parse citations from response text
 * Looks for patterns like:
 * - "According to [Type] ([Source])"
 * - "([Type] ([Source]))"
 * - "from [Type]"
 * 
 * @param {string} responseText - AI response text
 * @returns {Array} Array of citation objects
 */
export const parseCitationsFromText = (responseText) => {
  if (!responseText || typeof responseText !== 'string') {
    return [];
  }

  const citations = [];
  
  // Pattern 1: "According to [Type] ([Source])"
  const pattern1 = /according\s+to\s+(?:the\s+)?([^(]+?)\s*\(([^)]+)\)/gi;
  let match;
  
  while ((match = pattern1.exec(responseText)) !== null) {
    citations.push({
      type: match[1].trim(),
      source: match[2].trim(),
      text: match[0],
      startPos: match.index,
      endPos: match.index + match[0].length,
      confidence: 0.7 // Lower confidence for parsed citations
    });
  }
  
  // Pattern 2: "([Type] ([Source]))"
  const pattern2 = /\(([A-Za-z0-9\s\-]+?)\s*\(([^)]+)\)\)/gi;
  while ((match = pattern2.exec(responseText)) !== null) {
    // Avoid duplicates
    const isDuplicate = citations.some(c => 
      c.source === match[2].trim() && c.type === match[1].trim()
    );
    if (!isDuplicate) {
      citations.push({
        type: match[1].trim(),
        source: match[2].trim(),
        text: match[0],
        startPos: match.index,
        endPos: match.index + match[0].length,
        confidence: 0.6
      });
    }
  }
  
  // Pattern 3: Extract year from citations
  citations.forEach(citation => {
    const yearMatch = citation.text.match(/\b(19|20)\d{2}\b/);
    if (yearMatch) {
      citation.year = parseInt(yearMatch[0]);
    }
    
    // Detect document type
    const textLower = citation.text.toLowerCase();
    if (textLower.includes('10-k') || textLower.includes('10k')) {
      citation.type = '10-K Report';
    } else if (textLower.includes('impact report')) {
      citation.type = 'Impact Report';
    } else if (textLower.includes('annual report')) {
      citation.type = 'Annual Report';
    } else if (textLower.includes('10-q')) {
      citation.type = '10-Q Report';
    }
  });
  
  return citations;
};

/**
 * Format citation for display
 * @param {Object} citation - Citation object
 * @returns {string} Formatted citation string
 */
export const formatCitation = (citation) => {
  if (!citation) return '';
  
  const type = citation.type || 'Document';
  const source = citation.source || 'Unknown';
  const year = citation.year ? ` (${citation.year})` : '';
  
  return `${type}${year} - ${source}`;
};

/**
 * Replace citation text with clickable links (optional)
 * @param {string} text - Response text
 * @param {Array} citations - Citation objects
 * @returns {string} Text with citation markers
 */
export const markCitationsInText = (text, citations) => {
  if (!citations || citations.length === 0) {
    return text;
  }
  
  // Sort by position (reverse to replace from end to start)
  const sortedCitations = [...citations].sort((a, b) => b.startPos - a.startPos);
  
  let modifiedText = text;
  sortedCitations.forEach((citation, index) => {
    const marker = `[${index + 1}]`;
    modifiedText = 
      modifiedText.substring(0, citation.startPos) +
      marker +
      modifiedText.substring(citation.endPos);
  });
  
  return modifiedText;
};


