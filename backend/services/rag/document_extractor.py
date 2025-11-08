"""
Document Extractor Service
Extracts text and metadata from PDF documents
Now supports multi-modal extraction (text + tables)
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import pypdf

from .multimodal_extractor import MultiModalExtractor

logger = logging.getLogger(__name__)


class DocumentExtractor:
    """Extracts text and metadata from PDF documents"""
    
    def __init__(self, enable_tables: bool = True):
        """
        Initialize document extractor
        
        Args:
            enable_tables: Whether to enable table extraction
        """
        self.enable_tables = enable_tables
        self.multimodal_extractor = MultiModalExtractor() if enable_tables else None
        logger.info(f"📄 Document extractor initialized (tables: {enable_tables})")
    
    def extract_from_pdf(self, pdf_path: str, metadata: Dict) -> Dict:
        """
        Extract text, tables, and metadata from PDF (multi-modal extraction)
        
        Args:
            pdf_path: Path to PDF file
            metadata: Document metadata (from discovery service)
            
        Returns:
            Dictionary with extracted text, tables, pages, and enhanced metadata
        """
        file_path = Path(pdf_path)
        
        if not file_path.exists():
            logger.error(f"❌ PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            logger.info(f"📖 Extracting from: {file_path.name}")
            
            # Use multi-modal extractor if enabled
            if self.enable_tables and self.multimodal_extractor:
                try:
                    multimodal_result = self.multimodal_extractor.extract_all(pdf_path, metadata)
                    
                    # Extract text for backward compatibility
                    full_text = multimodal_result.get("combined_content", multimodal_result.get("text", ""))
                    tables = multimodal_result.get("tables", [])
                    
                    # Build pages_text from combined content (approximate)
                    pages_text = [full_text]  # Simplified for now
                    
                    # Build result with table information
                    result = {
                        "text": full_text,
                        "tables": tables,
                        "pages": pages_text,
                        "total_pages": metadata.get("total_pages", len(pages_text)),
                        "total_characters": len(full_text),
                        "total_words": len(full_text.split()),
                        "table_count": len(tables),
                        "has_tables": len(tables) > 0,
                        "metadata": {
                            **metadata,
                            **multimodal_result.get("metadata", {}),
                            "extraction_date": None
                        }
                    }
                    
                    logger.info(f"✅ Extracted {result['total_pages']} pages, {result['total_words']} words, "
                               f"{result['table_count']} tables from {file_path.name}")
                    
                    return result
                
                except Exception as e:
                    logger.warning(f"⚠️ Multi-modal extraction failed, falling back to text-only: {e}")
                    # Fall through to text-only extraction
            
            # Fallback to text-only extraction
            logger.info(f"📖 Extracting text from: {file_path.name}")
            
            # Extract text from PDF
            text_content = []
            pages_text = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                logger.info(f"📄 PDF has {total_pages} pages")
                
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    try:
                        page_text = page.extract_text()
                        pages_text.append(page_text)
                        text_content.append(page_text)
                        
                        if page_num % 50 == 0:
                            logger.info(f"  Processed {page_num}/{total_pages} pages...")
                    except Exception as e:
                        logger.warning(f"⚠️ Error extracting page {page_num}: {e}")
                        pages_text.append("")  # Empty page if extraction fails
            
            # Combine all text
            full_text = "\n\n".join(text_content)
            
            # Extract document metadata from PDF if available
            pdf_metadata = {}
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = pypdf.PdfReader(file)
                    if pdf_reader.metadata:
                        pdf_metadata = {
                            "title": pdf_reader.metadata.get("/Title", ""),
                            "author": pdf_reader.metadata.get("/Author", ""),
                            "creator": pdf_reader.metadata.get("/Creator", ""),
                            "producer": pdf_reader.metadata.get("/Producer", ""),
                            "creation_date": str(pdf_reader.metadata.get("/CreationDate", "")),
                            "modification_date": str(pdf_reader.metadata.get("/ModDate", ""))
                        }
            except Exception as e:
                logger.warning(f"⚠️ Could not extract PDF metadata: {e}")
            
            # Build result
            result = {
                "text": full_text,
                "pages": pages_text,
                "total_pages": total_pages,
                "total_characters": len(full_text),
                "total_words": len(full_text.split()),
                "metadata": {
                    **metadata,
                    **pdf_metadata,
                    "extraction_date": None  # Will be set by caller
                }
            }
            
            logger.info(f"✅ Extracted {result['total_pages']} pages, "
                       f"{result['total_words']} words from {file_path.name}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error extracting PDF {pdf_path}: {e}")
            raise Exception(f"Failed to extract PDF: {str(e)}")
    
    def extract_text_chunk(self, text: str, start_char: int, end_char: int) -> str:
        """
        Extract a specific chunk of text
        
        Args:
            text: Full text
            start_char: Start character position
            end_char: End character position
            
        Returns:
            Extracted text chunk
        """
        return text[start_char:end_char]

