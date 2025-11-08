"""
File Processor Service
Handles different file types (PDF, Excel, CSV, Images) and extracts content
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

logger = logging.getLogger(__name__)


class FileProcessor:
    """Process various file types and extract content"""
    
    def __init__(self, upload_dir: Optional[str] = None):
        """
        Initialize file processor
        
        Args:
            upload_dir: Directory to store uploaded files
        """
        self.upload_dir = Path(upload_dir) if upload_dir else Path("backend/data/uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 File processor initialized, upload directory: {self.upload_dir}")
    
    def process_file(self, file_path: str, filename: str, file_type: str) -> Dict:
        """
        Process a file based on its type
        
        Args:
            file_path: Path to uploaded file
            filename: Original filename
            file_type: MIME type or extension
            
        Returns:
            Dictionary with extracted content and metadata
        """
        file_ext = Path(filename).suffix.lower()
        
        logger.info(f"📄 Processing file: {filename} (type: {file_type}, ext: {file_ext})")
        
        try:
            if file_ext in ['.pdf']:
                return self._process_pdf(file_path, filename)
            elif file_ext in ['.xlsx', '.xls']:
                return self._process_excel(file_path, filename)
            elif file_ext in ['.csv']:
                return self._process_csv(file_path, filename)
            elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
                return self._process_image(file_path, filename)
            else:
                logger.warning(f"⚠️ Unsupported file type: {file_ext}")
                return {
                    "success": False,
                    "error": f"Unsupported file type: {file_ext}",
                    "text": "",
                    "metadata": {"filename": filename, "file_type": file_type}
                }
        
        except Exception as e:
            logger.error(f"❌ Error processing file {filename}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "metadata": {"filename": filename, "file_type": file_type}
            }
    
    def _process_pdf(self, file_path: str, filename: str) -> Dict:
        """Extract text and tables from PDF"""
        try:
            text_content = []
            tables = []
            
            if PDFPLUMBER_AVAILABLE:
                # Use pdfplumber for better table extraction
                with pdfplumber.open(file_path) as pdf:
                    total_pages = len(pdf.pages)
                    
                    for page_num, page in enumerate(pdf.pages, 1):
                        # Extract text
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(page_text)
                        
                        # Extract tables
                        page_tables = page.extract_tables()
                        for table_idx, table in enumerate(page_tables):
                            if table:
                                # Convert to text representation
                                table_text = self._table_to_text(table)
                                tables.append({
                                    "page": page_num,
                                    "table_index": table_idx,
                                    "data": table,
                                    "text": table_text
                                })
                                text_content.append(f"\n\n[Table {len(tables)} from page {page_num}]\n{table_text}\n\n")
            else:
                # Fallback to pypdf
                import pypdf
                with open(file_path, 'rb') as file:
                    pdf_reader = pypdf.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(page_text)
            
            full_text = "\n\n".join(text_content)
            
            return {
                "success": True,
                "text": full_text,
                "tables": tables,
                "metadata": {
                    "filename": filename,
                    "file_type": "application/pdf",
                    "pages": len(text_content),
                    "table_count": len(tables)
                }
            }
        
        except Exception as e:
            logger.error(f"❌ Error processing PDF: {e}")
            raise
    
    def _process_excel(self, file_path: str, filename: str) -> Dict:
        """Extract data from Excel files"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            sheets_data = {}
            all_text = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # Convert DataFrame to text representation
                sheet_text = f"\n\n[Sheet: {sheet_name}]\n"
                sheet_text += df.to_string(index=False)
                all_text.append(sheet_text)
                
                sheets_data[sheet_name] = {
                    "data": df.to_dict('records'),
                    "columns": df.columns.tolist(),
                    "rows": len(df),
                    "text": sheet_text
                }
            
            full_text = "\n\n".join(all_text)
            
            return {
                "success": True,
                "text": full_text,
                "sheets": sheets_data,
                "metadata": {
                    "filename": filename,
                    "file_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "sheet_count": len(sheets_data),
                    "sheets": list(sheets_data.keys())
                }
            }
        
        except Exception as e:
            logger.error(f"❌ Error processing Excel: {e}")
            raise
    
    def _process_csv(self, file_path: str, filename: str) -> Dict:
        """Extract data from CSV files"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not read CSV with any supported encoding")
            
            # Convert to text
            csv_text = f"[CSV Data from {filename}]\n"
            csv_text += df.to_string(index=False)
            
            return {
                "success": True,
                "text": csv_text,
                "data": df.to_dict('records'),
                "columns": df.columns.tolist(),
                "rows": len(df),
                "metadata": {
                    "filename": filename,
                    "file_type": "text/csv",
                    "row_count": len(df),
                    "column_count": len(df.columns)
                }
            }
        
        except Exception as e:
            logger.error(f"❌ Error processing CSV: {e}")
            raise
    
    def _process_image(self, file_path: str, filename: str) -> Dict:
        """Extract text from images using OCR"""
        if not TESSERACT_AVAILABLE:
            return {
                "success": False,
                "error": "OCR not available. Install pytesseract and tesseract-ocr",
                "text": "",
                "metadata": {"filename": filename, "file_type": "image"}
            }
        
        try:
            # Open image
            image = Image.open(file_path)
            
            # Extract text using OCR
            extracted_text = pytesseract.image_to_string(image)
            
            # Also get structured data if available
            try:
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            except:
                data = None
            
            return {
                "success": True,
                "text": extracted_text,
                "ocr_data": data,
                "metadata": {
                    "filename": filename,
                    "file_type": "image",
                    "image_size": image.size,
                    "image_format": image.format
                }
            }
        
        except Exception as e:
            logger.error(f"❌ Error processing image: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "metadata": {"filename": filename, "file_type": "image"}
            }
    
    def _table_to_text(self, table: List[List]) -> str:
        """Convert table data to text representation"""
        if not table or len(table) == 0:
            return ""
        
        text_lines = []
        
        # Header row
        if table:
            header = [str(cell) if cell is not None else "" for cell in table[0]]
            text_lines.append(" | ".join(header))
            text_lines.append("-" * (sum(len(cell) for cell in header) + len(header) * 3))
        
        # Data rows
        for row in table[1:] if len(table) > 1 else []:
            row_text = [str(cell) if cell is not None else "" for cell in row]
            text_lines.append(" | ".join(row_text))
        
        return "\n".join(text_lines)

