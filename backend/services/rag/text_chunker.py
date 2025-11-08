"""
Text Chunker Service
Splits documents into semantic chunks for embedding
"""

import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class TextChunker:
    """Splits documents into chunks for embedding"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap size between chunks for context preservation
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"✂️ Text chunker initialized: chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_document(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split document into semantic chunks
        
        Args:
            text: Full document text
            metadata: Document metadata
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            logger.warning("⚠️ Empty text provided for chunking")
            return []
        
        logger.info(f"✂️ Chunking document: {len(text)} characters")
        
        # Strategy: Sentence-based chunking with overlap
        chunks = []
        
        # Split into sentences (handle various sentence endings)
        sentences = self._split_into_sentences(text)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence) + 1  # +1 for space
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = " ".join(current_chunk)
                chunk_metadata = self._create_chunk_metadata(metadata, len(chunks))
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata,
                    "chunk_index": len(chunks),
                    "character_count": len(chunk_text)
                })
                
                # Start new chunk with overlap (keep last N sentences)
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) + 1 for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_metadata = self._create_chunk_metadata(metadata, len(chunks))
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata,
                "chunk_index": len(chunks),
                "character_count": len(chunk_text)
            })
        
        logger.info(f"✅ Created {len(chunks)} chunks from document")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Pattern to match sentence endings (., !, ?) followed by space or newline
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        # Filter out empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """
        Get overlap sentences from end of chunk
        
        Args:
            sentences: Current chunk sentences
            
        Returns:
            Overlap sentences (last N sentences that fit in overlap size)
        """
        if not sentences:
            return []
        
        # Get sentences from the end that fit in overlap size
        overlap_sentences = []
        overlap_length = 0
        
        for sentence in reversed(sentences):
            sentence_length = len(sentence) + 1
            if overlap_length + sentence_length <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_length += sentence_length
            else:
                break
        
        return overlap_sentences
    
    def _create_chunk_metadata(self, base_metadata: Dict, chunk_index: int) -> Dict:
        """
        Create metadata for a chunk (enhanced for Phase 3)
        
        Args:
            base_metadata: Base document metadata
            chunk_index: Index of this chunk
            
        Returns:
            Chunk metadata dictionary (with no None values for ChromaDB compatibility)
        """
        # Build metadata, ensuring no None values
        metadata = {
            "source": base_metadata.get("filename", "unknown"),
            "document_type": base_metadata.get("document_type", "Report"),
            "chunk_index": chunk_index,
        }
        
        # Add year only if it's not None
        year = base_metadata.get("year")
        if year is not None:
            metadata["year"] = str(year) if not isinstance(year, str) else year
        
        # Add other metadata fields, skipping None values and excluded fields
        excluded_fields = ["filename", "file_path", "file_size"]
        for k, v in base_metadata.items():
            if k not in excluded_fields and v is not None:
                # Ensure value is a valid type for ChromaDB (bool, int, float, str)
                if isinstance(v, (bool, int, float, str)):
                    metadata[k] = v
                elif v is not None:
                    # Convert to string if possible
                    try:
                        metadata[k] = str(v)
                    except:
                        # Skip if can't convert
                        continue
        
        return metadata
    
    def chunk_by_sections(self, text: str, metadata: Dict, section_patterns: List[str] = None) -> List[Dict]:
        """
        Chunk document by sections (useful for structured documents like 10K)
        
        Args:
            text: Full document text
            metadata: Document metadata
            section_patterns: List of regex patterns to identify sections
            
        Returns:
            List of chunk dictionaries
        """
        if section_patterns is None:
            # Default section patterns for financial documents
            section_patterns = [
                r'^\s*(ITEM\s+\d+[\.:]?\s+)',  # ITEM 1, ITEM 2, etc.
                r'^\s*([A-Z][A-Z\s]{10,})',     # ALL CAPS headings
                r'^\s*(\d+\.\s+[A-Z])',         # Numbered sections
            ]
        
        # Find section boundaries
        sections = []
        current_section = {"title": "Introduction", "text": ""}
        
        lines = text.split('\n')
        
        for line in lines:
            # Check if line matches section pattern
            is_section_header = False
            for pattern in section_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    # Save previous section
                    if current_section["text"].strip():
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = {
                        "title": line.strip(),
                        "text": ""
                    }
                    is_section_header = True
                    break
            
            if not is_section_header:
                current_section["text"] += line + "\n"
        
        # Add final section
        if current_section["text"].strip():
            sections.append(current_section)
        
        # Chunk each section
        all_chunks = []
        for section_idx, section in enumerate(sections):
            section_metadata = {
                **metadata,
                "section_title": section["title"],
                "section_index": section_idx
            }
            
            # Use regular chunking for section content
            section_chunks = self.chunk_document(section["text"], section_metadata)
            all_chunks.extend(section_chunks)
        
        logger.info(f"✅ Created {len(all_chunks)} chunks from {len(sections)} sections")
        return all_chunks

