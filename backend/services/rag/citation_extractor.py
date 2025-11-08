"""
Citation Extractor and Verifier
Extracts citations from AI responses and verifies them against retrieved documents
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CitationExtractor:
    """Extracts and verifies citations from AI responses"""
    
    def __init__(self):
        """Initialize citation extractor"""
        # Patterns for citation detection
        self.citation_patterns = [
            # "According to [Type] ([Source])"
            re.compile(r'according\s+to\s+(?:the\s+)?([^(]+?)\s*\(([^)]+)\)', re.IGNORECASE),
            # "([Type] ([Source]))"
            re.compile(r'\(([^(]+?)\s*\(([^)]+)\)\)', re.IGNORECASE),
            # "[Source] shows/indicates/reports"
            re.compile(r'([A-Za-z0-9\s\-_]+\.pdf)\s+(?:shows|indicates|reports|states)', re.IGNORECASE),
            # "from [Type]"
            re.compile(r'from\s+(?:the\s+)?(?:([0-9]{4})\s+)?([A-Za-z0-9\s]+?)(?:\s+report|\s+document|\s+10-?k|\s+10-?q)?', re.IGNORECASE),
        ]
        
        # Document type patterns
        self.doc_type_patterns = {
            '10-K': re.compile(r'10\s*-?\s*k|10k', re.IGNORECASE),
            '10-Q': re.compile(r'10\s*-?\s*q|10q', re.IGNORECASE),
            'Impact Report': re.compile(r'impact\s+report', re.IGNORECASE),
            'Annual Report': re.compile(r'annual\s+report', re.IGNORECASE),
        }
        
        logger.info("📝 Citation Extractor initialized")
    
    def extract_citations(self, response_text: str) -> List[Dict]:
        """
        Extract citations from AI response text
        
        Args:
            response_text: AI-generated response text
            
        Returns:
            List of extracted citations with metadata
        """
        if not response_text or not isinstance(response_text, str):
            logger.warning("⚠️ Invalid response_text provided to extract_citations")
            return []
        
        citations = []
        
        # Try each citation pattern
        for pattern in self.citation_patterns:
            matches = pattern.finditer(response_text)
            for match in matches:
                citation = {
                    "text": match.group(0),
                    "type": None,
                    "source": None,
                    "year": None,
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                    "confidence": 0.5
                }
                
                # Extract components based on pattern
                groups = match.groups()
                if len(groups) >= 2:
                    citation["type"] = groups[0].strip() if groups[0] is not None else None
                    citation["source"] = groups[1].strip() if groups[1] is not None else None
                elif len(groups) == 1:
                    citation["source"] = groups[0].strip() if groups[0] is not None else None
                
                # Extract year from type or source
                type_str = citation.get("type") or ""
                source_str = citation.get("source") or ""
                year_match = re.search(r'\b(19|20)\d{2}\b', f"{type_str} {source_str}")
                if year_match:
                    citation["year"] = int(year_match.group(0))
                
                # Detect document type
                type_text = citation.get("type") or ""
                source_text = citation.get("source") or ""
                full_text = f"{type_text} {source_text}".lower()
                for doc_type, doc_pattern in self.doc_type_patterns.items():
                    if doc_pattern.search(full_text):
                        citation["type"] = doc_type
                        break
                
                citations.append(citation)
        
        # Remove duplicates
        unique_citations = []
        seen = set()
        for cit in citations:
            key = (cit.get("source"), cit.get("type"), cit.get("year"))
            if key not in seen:
                seen.add(key)
                unique_citations.append(cit)
        
        logger.info(f"📝 Extracted {len(unique_citations)} citations from response")
        return unique_citations
    
    def verify_citations(self, citations: List[Dict], retrieved_docs: List[Dict]) -> List[Dict]:
        """
        Verify citations against retrieved documents
        
        Args:
            citations: Extracted citations
            retrieved_docs: Documents retrieved for the query
            
        Returns:
            Verified citations with verification status
        """
        verified_citations = []
        
        # Create lookup map of retrieved documents
        doc_lookup = {}
        for doc in retrieved_docs:
            metadata = doc.get('metadata', {})
            source = metadata.get('source', '').lower()
            doc_type = metadata.get('document_type', '').lower()
            year = metadata.get('year')
            
            # Index by source
            doc_lookup[source] = doc
            
            # Index by type + year
            if doc_type and year:
                key = f"{doc_type}_{year}"
                if key not in doc_lookup:
                    doc_lookup[key] = doc
        
        # Verify each citation
        for citation in citations:
            verified = citation.copy()
            verified["verified"] = False
            verified["match_document"] = None
            verified["match_score"] = 0.0
            
            citation_source = citation.get("source", "").lower()
            citation_type = citation.get("type", "").lower()
            citation_year = citation.get("year")
            
            # Try to match by source filename
            if citation_source:
                # Normalize source (remove path, keep filename)
                source_file = citation_source.split('/')[-1] if '/' in citation_source else citation_source
                source_file = source_file.replace('.pdf', '')
                
                for doc_source, doc in doc_lookup.items():
                    doc_file = doc_source.lower().replace('.pdf', '')
                    if source_file in doc_file or doc_file in source_file:
                        verified["verified"] = True
                        verified["match_document"] = doc
                        verified["match_score"] = 0.9
                        break
            
            # Try to match by type and year
            if not verified["verified"] and citation_type and citation_year:
                key = f"{citation_type}_{citation_year}".lower()
                if key in doc_lookup:
                    verified["verified"] = True
                    verified["match_document"] = doc_lookup[key]
                    verified["match_score"] = 0.8
            
            # Try partial matching
            if not verified["verified"]:
                for doc_source, doc in doc_lookup.items():
                    metadata = doc.get('metadata', {})
                    doc_type = metadata.get('document_type', '').lower()
                    doc_year = metadata.get('year')
                    
                    # Match by type similarity
                    if citation_type and doc_type and citation_type in doc_type:
                        if not citation_year or citation_year == doc_year:
                            verified["verified"] = True
                            verified["match_document"] = doc
                            verified["match_score"] = 0.6
                            break
            
            verified_citations.append(verified)
        
        verified_count = sum(1 for c in verified_citations if c["verified"])
        logger.info(f"✅ Verified {verified_count}/{len(verified_citations)} citations")
        
        return verified_citations
    
    def format_citations_for_display(self, verified_citations: List[Dict]) -> Dict:
        """
        Format verified citations for frontend display
        
        Args:
            verified_citations: Verified citations
            
        Returns:
            Formatted citation data
        """
        verified = [c for c in verified_citations if c.get("verified")]
        unverified = [c for c in verified_citations if not c.get("verified")]
        
        return {
            "total_citations": len(verified_citations),
            "verified_count": len(verified),
            "unverified_count": len(unverified),
            "verified_citations": [
                {
                    "source": c.get("source"),
                    "type": c.get("type"),
                    "year": c.get("year"),
                    "confidence": c.get("match_score", 0),
                    "text": c.get("text")
                }
                for c in verified
            ],
            "unverified_citations": [
                {
                    "source": c.get("source"),
                    "type": c.get("type"),
                    "year": c.get("year"),
                    "text": c.get("text")
                }
                for c in unverified
            ]
        }

