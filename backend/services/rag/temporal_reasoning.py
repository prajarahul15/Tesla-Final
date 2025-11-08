"""
Temporal Reasoning Service
Detects temporal queries and enhances retrieval with temporal context
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class TemporalReasoningService:
    """Service for temporal query detection and reasoning"""
    
    def __init__(self):
        """Initialize temporal reasoning service"""
        # Patterns for temporal expressions
        self.year_pattern = re.compile(r'\b(19|20)\d{2}\b')
        self.quarter_pattern = re.compile(r'\bQ[1-4]\s*(?:of\s*)?(?:20\d{2}|202\d)\b', re.IGNORECASE)
        self.relative_patterns = [
            (re.compile(r'\b(last|past|previous|recent)\s+(?:year|years|quarter|quarters|month|months)\b', re.IGNORECASE), 'relative'),
            (re.compile(r'\b(over|in|during|since|from)\s+(?:the\s+)?(?:last|past)\s+\d+\s+(?:year|years|quarter|quarters|month|months)\b', re.IGNORECASE), 'range'),
            (re.compile(r'\b(trend|trending|evolution|change)\s+(?:over|in|during)\b', re.IGNORECASE), 'trend'),
            (re.compile(r'\b(compare|comparison|vs|versus|compared\s+to)\b', re.IGNORECASE), 'comparison'),
            (re.compile(r'\b(annual|yearly|year-over-year|yoy|y\/y)\b', re.IGNORECASE), 'yoy'),
        ]
        logger.info("🕐 Temporal Reasoning Service initialized")
    
    def detect_temporal_query(self, query: str) -> Dict:
        """
        Detect temporal aspects in a query
        
        Args:
            query: User query text
            
        Returns:
            Dictionary with temporal information
        """
        query_lower = query.lower()
        temporal_info = {
            "has_temporal": False,
            "type": None,
            "years": [],
            "quarters": [],
            "date_range": None,
            "comparison": False,
            "trend_analysis": False
        }
        
        # Extract years (4-digit years only, avoid extracting "20" from "2024")
        year_matches = re.findall(r'\b(20\d{2})\b', query)  # Only 2000-2099
        if not year_matches:
            # Fallback: also check 1900s
            year_matches = re.findall(r'\b(19\d{2})\b', query)
        
        if year_matches:
            years = [int(year) for year in set(year_matches)]
            temporal_info["years"] = sorted(years)
            temporal_info["has_temporal"] = True
        
        # Extract quarters
        quarters = self.quarter_pattern.findall(query)
        if quarters:
            temporal_info["quarters"] = quarters
            temporal_info["has_temporal"] = True
        
        # Check for temporal patterns
        for pattern, pattern_type in self.relative_patterns:
            if pattern.search(query):
                temporal_info["has_temporal"] = True
                if pattern_type == 'comparison':
                    temporal_info["comparison"] = True
                    temporal_info["type"] = "comparison"
                elif pattern_type == 'trend':
                    temporal_info["trend_analysis"] = True
                    temporal_info["type"] = "trend"
                elif pattern_type == 'range':
                    temporal_info["type"] = "range"
                    # Extract number from range
                    range_match = re.search(r'(?:last|past)\s+(\d+)\s+(?:year|years|quarter|quarters)', query_lower)
                    if range_match:
                        years_back = int(range_match.group(1))
                        current_year = datetime.now().year
                        temporal_info["years"] = list(range(current_year - years_back, current_year + 1))
                elif pattern_type == 'relative':
                    temporal_info["type"] = "relative"
                    # Determine relative year
                    if 'last year' in query_lower or 'previous year' in query_lower:
                        temporal_info["years"] = [datetime.now().year - 1]
                    elif 'last quarter' in query_lower:
                        # Could add quarter logic here
                        pass
        
        # Detect comparison patterns (e.g., "2024 vs 2023")
        if temporal_info["comparison"] or len(temporal_info["years"]) >= 2:
            temporal_info["comparison"] = True
            temporal_info["type"] = "comparison"
        
        # Detect trend queries
        if any(word in query_lower for word in ['trend', 'evolution', 'change over time', 'over the years']):
            temporal_info["trend_analysis"] = True
            temporal_info["type"] = "trend"
        
        if temporal_info["has_temporal"]:
            logger.info(f"🕐 Temporal query detected: type={temporal_info['type']}, years={temporal_info['years']}")
        
        return temporal_info
    
    def build_temporal_filters(self, temporal_info: Dict) -> Optional[Dict]:
        """
        Build metadata filters based on temporal information
        
        Args:
            temporal_info: Temporal information from detect_temporal_query
            
        Returns:
            Metadata filter dictionary for ChromaDB (single year only, ChromaDB doesn't support $in)
        """
        if not temporal_info.get("has_temporal"):
            return None
        
        filters = {}
        
        # Filter by years if specified
        if temporal_info.get("years"):
            # ChromaDB doesn't support $in operator or OR conditions
            # For single year, use it directly
            # For multiple years, we'll skip the year filter and handle in post-processing
            if len(temporal_info["years"]) == 1:
                # Ensure year is string (ChromaDB metadata should match stored type)
                year = temporal_info["years"][0]
                filters["year"] = str(year) if isinstance(year, int) else year
            # For multiple years, return None to search all and filter in post-processing
            # The calling code should handle this case
        
        return filters if filters else None
    
    def enhance_retrieval_for_temporal(self, temporal_info: Dict, top_k: int = 5) -> int:
        """
        Adjust top_k based on temporal requirements
        
        Args:
            temporal_info: Temporal information
            top_k: Default top_k
            
        Returns:
            Adjusted top_k value
        """
        # For comparisons, retrieve more documents
        if temporal_info.get("comparison") or temporal_info.get("trend_analysis"):
            return top_k * 2  # Double for comparisons
        elif len(temporal_info.get("years", [])) > 1:
            return top_k * len(temporal_info["years"])  # Per year
        
        return top_k
    
    def build_temporal_prompt_context(self, temporal_info: Dict, retrieved_docs: List[Dict]) -> str:
        """
        Build prompt context for temporal queries
        
        Args:
            temporal_info: Temporal information
            retrieved_docs: Retrieved documents
            
        Returns:
            Formatted prompt context string
        """
        if not temporal_info.get("has_temporal"):
            return ""
        
        context_parts = [
            "",
            "=== TEMPORAL CONTEXT ===",
        ]
        
        if temporal_info.get("comparison"):
            context_parts.append(
                f"This query requires comparing data across multiple time periods: {', '.join(map(str, temporal_info.get('years', [])))}"
            )
            context_parts.append("Please provide side-by-side comparisons showing differences and trends.")
        
        if temporal_info.get("trend_analysis"):
            context_parts.append(
                f"This query requires trend analysis. Analyze how metrics have changed over time."
            )
            context_parts.append("Show progression, rate of change, and key inflection points.")
        
        if temporal_info.get("years"):
            context_parts.append(
                f"Focus on data from years: {', '.join(map(str, temporal_info['years']))}"
            )
        
        # Group documents by year for temporal context
        docs_by_year = {}
        for doc in retrieved_docs:
            year = doc.get('metadata', {}).get('year', 'Unknown')
            # Normalize year to string for consistent sorting
            if year is not None and year != 'Unknown':
                year_str = str(year)
            else:
                year_str = 'Unknown'
            
            if year_str not in docs_by_year:
                docs_by_year[year_str] = []
            docs_by_year[year_str].append(doc)
        
        if docs_by_year:
            context_parts.append("")
            context_parts.append("Retrieved documents organized by year:")
            # Sort years: convert numeric years to int for proper sorting, keep 'Unknown' at end
            year_keys = []
            unknown_years = []
            for year_key in docs_by_year.keys():
                if year_key == 'Unknown':
                    unknown_years.append(year_key)
                else:
                    try:
                        # Try to convert to int for proper numeric sorting
                        year_keys.append((int(year_key), year_key))
                    except (ValueError, TypeError):
                        # If not numeric, keep as string
                        year_keys.append((0, year_key))
            
            # Sort numeric years in reverse (newest first), then add Unknown
            sorted_years = [y[1] for y in sorted(year_keys, reverse=True)] + unknown_years
            for year in sorted_years:
                context_parts.append(f"  - {year}: {len(docs_by_year[year])} document chunks")
        
        context_parts.append("")
        
        return "\n".join(context_parts)

