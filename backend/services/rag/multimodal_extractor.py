"""
Multi-Modal Document Extractor
Extracts both text and tables from PDF documents
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path
import pypdf
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logging.warning("pdfplumber not available, table extraction will be limited")

logger = logging.getLogger(__name__)


class MultiModalExtractor:
    """Extracts text and tables from PDF documents"""
    
    def __init__(self):
        """Initialize multi-modal extractor"""
        self.use_pdfplumber = PDFPLUMBER_AVAILABLE
        logger.info(f"📊 Multi-Modal Extractor initialized (pdfplumber: {self.use_pdfplumber})")
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF (fallback to pypdf)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        text_content = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page in pdf_reader.pages:
                    try:
                        page_text = page.extract_text()
                        text_content.append(page_text)
                    except Exception as e:
                        logger.warning(f"⚠️ Error extracting text from page: {e}")
                        text_content.append("")
        except Exception as e:
            logger.error(f"❌ Error reading PDF: {e}")
            raise
        
        return "\n\n".join(text_content)
    
    def extract_tables(self, pdf_path: str) -> List[Dict]:
        """
        Extract tables from PDF using pdfplumber
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of extracted tables with metadata
        """
        if not self.use_pdfplumber:
            logger.warning("⚠️ pdfplumber not available, skipping table extraction")
            return []
        
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        page_tables = page.extract_tables()
                        
                        for table_idx, table in enumerate(page_tables):
                            if table and len(table) > 0:
                                # Convert table to text representation
                                table_text = self._table_to_text(table)
                                
                                tables.append({
                                    "page": page_num,
                                    "table_index": table_idx,
                                    "rows": len(table),
                                    "columns": len(table[0]) if table else 0,
                                    "table_data": table,
                                    "table_text": table_text,
                                    "metadata": {
                                        "source": Path(pdf_path).name,
                                        "page": page_num,
                                        "table_index": table_idx
                                    }
                                })
                        
                        if page_num % 50 == 0:
                            logger.info(f"  Processed {page_num}/{len(pdf.pages)} pages for tables...")
                    
                    except Exception as e:
                        logger.warning(f"⚠️ Error extracting tables from page {page_num}: {e}")
        
        except Exception as e:
            logger.error(f"❌ Error extracting tables from PDF: {e}")
            return []
        
        logger.info(f"📊 Extracted {len(tables)} tables from {Path(pdf_path).name}")
        return tables
    
    def _table_to_text(self, table: List[List]) -> str:
        """
        Convert table data to text representation
        
        Args:
            table: Table data (list of rows)
            
        Returns:
            Text representation of table
        """
        if not table:
            return ""
        
        text_lines = []
        
        # Header row (if exists)
        if table:
            header = [str(cell) if cell is not None else "" for cell in table[0]]
            text_lines.append(" | ".join(header))
            text_lines.append("-" * (sum(len(cell) for cell in header) + len(header) * 3))
        
        # Data rows
        for row in table[1:] if table else []:
            row_text = [str(cell) if cell is not None else "" for cell in row]
            text_lines.append(" | ".join(row_text))
        
        return "\n".join(text_lines)
    
    def extract_all(self, pdf_path: str, metadata: Dict) -> Dict:
        """
        Extract both text and tables from PDF
        
        Args:
            pdf_path: Path to PDF file
            metadata: Document metadata
            
        Returns:
            Dictionary with text, tables, and combined content
        """
        logger.info(f"📊 Multi-modal extraction from: {Path(pdf_path).name}")
        
        # Extract text
        text_content = self.extract_text(pdf_path)
        
        # Extract tables
        tables = self.extract_tables(pdf_path)
        
        # Combine text and tables
        combined_content = text_content
        
        if tables:
            combined_content += "\n\n=== TABLES ===\n\n"
            for table in tables:
                combined_content += f"\n[Table from page {table['page']}]\n"
                combined_content += table['table_text']
                combined_content += "\n\n"
        
        return {
            "text": text_content,
            "tables": tables,
            "combined_content": combined_content,
            "table_count": len(tables),
            "metadata": {
                **metadata,
                "has_tables": len(tables) > 0,
                "table_count": len(tables)
            }
        }


