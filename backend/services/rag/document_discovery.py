"""
Document Discovery Service
Scans Tesla Knowledge Base folder and identifies documents to process
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class DocumentDiscovery:
    """Discovers and tracks documents in Tesla Knowledge Base folder"""
    
    def __init__(self, knowledge_base_path: str = None):
        """
        Initialize document discovery service
        
        Args:
            knowledge_base_path: Path to Tesla Knowledge Base folder
                                If None, uses default location relative to backend
        """
        if knowledge_base_path is None:
            # Default: Tesla Knowledge Base folder at project root
            backend_dir = Path(__file__).parent.parent.parent
            self.kb_path = backend_dir.parent / "Tesla Knowledge Base"
        else:
            self.kb_path = Path(knowledge_base_path)
        
        # Status file to track processed documents
        backend_dir = Path(__file__).parent.parent.parent
        self.status_file = backend_dir / "data" / "rag_processing_status.json"
        
        # Ensure data directory exists
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Document discovery initialized: {self.kb_path}")
        logger.info(f"📋 Status file: {self.status_file}")
    
    def scan_for_documents(self) -> List[Dict]:
        """
        Scan Tesla Knowledge Base folder for PDF documents
        
        Returns:
            List of document information dictionaries
        """
        if not self.kb_path.exists():
            logger.warning(f"⚠️ Knowledge Base folder not found: {self.kb_path}")
            return []
        
        documents = []
        
        # Scan for PDF files
        for file_path in self.kb_path.glob("*.pdf"):
            doc_info = self._extract_document_info(file_path)
            if doc_info:
                documents.append(doc_info)
        
        logger.info(f"📄 Found {len(documents)} PDF documents in {self.kb_path}")
        return documents
    
    def _extract_document_info(self, file_path: Path) -> Optional[Dict]:
        """
        Extract metadata from filename
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Document info dictionary or None if invalid
        """
        filename = file_path.name
        file_size = file_path.stat().st_size
        
        # Parse document type and year from filename
        doc_type, year = self._parse_filename(filename)
        
        if not doc_type:
            logger.warning(f"⚠️ Could not determine document type for: {filename}")
            return None
        
        return {
            "filename": filename,
            "file_path": str(file_path),
            "file_size": file_size,
            "document_type": doc_type,
            "year": year,
            "modified_date": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        }
    
    def _parse_filename(self, filename: str) -> tuple:
        """
        Parse document type and year from filename
        
        Examples:
            "10K Report 2024.pdf" -> ("10K", 2024)
            "2024-extended-version-tesla-impact-report.pdf" -> ("Impact Report", 2024)
            "Tesla Annual Reports 2023.pdf" -> ("Annual Report", 2023)
        
        Returns:
            Tuple of (document_type, year) or (None, None) if cannot parse
        """
        filename_lower = filename.lower()
        
        # Extract year (4 digits between 2000-2100)
        import re
        year_match = re.search(r'\b(20[0-2]\d)\b', filename)
        year = int(year_match.group(1)) if year_match else None
        
        # Determine document type
        doc_type = None
        
        if "10k" in filename_lower or "10-k" in filename_lower:
            doc_type = "10K"
        elif "impact" in filename_lower:
            doc_type = "Impact Report"
        elif "annual" in filename_lower:
            doc_type = "Annual Report"
        elif "nasdaq" in filename_lower:
            doc_type = "NASDAQ Report"
        elif "quarterly" in filename_lower or "10q" in filename_lower or "10-q" in filename_lower:
            doc_type = "10Q"
        else:
            # Default fallback
            doc_type = "Report"
        
        return doc_type, year
    
    def get_processed_files(self) -> Dict:
        """
        Load processing status from file
        
        Returns:
            Dictionary with processing status
        """
        if not self.status_file.exists():
            return {
                "processed_files": [],
                "last_scan_date": None,
                "total_processed": 0,
                "total_failed": 0,
                "total_pending": 0
            }
        
        try:
            with open(self.status_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Error loading status file: {e}")
            return {
                "processed_files": [],
                "last_scan_date": None,
                "total_processed": 0,
                "total_failed": 0,
                "total_pending": 0
            }
    
    def is_processed(self, filename: str) -> bool:
        """
        Check if document has been processed
        
        Args:
            filename: Document filename
            
        Returns:
            True if processed, False otherwise
        """
        status = self.get_processed_files()
        processed_filenames = {f["filename"] for f in status.get("processed_files", [])}
        return filename in processed_filenames
    
    def mark_processed(
        self, 
        filename: str, 
        chunks_count: int, 
        pages_count: Optional[int] = None,
        status: str = "completed"
    ):
        """
        Mark document as processed
        
        Args:
            filename: Document filename
            chunks_count: Number of chunks created
            pages_count: Number of pages (if available)
            status: Processing status (completed, failed, processing)
        """
        status_data = self.get_processed_files()
        
        # Find existing entry or create new
        processed_files = status_data.get("processed_files", [])
        
        # Remove existing entry if present
        processed_files = [f for f in processed_files if f["filename"] != filename]
        
        # Add new entry
        processed_files.append({
            "filename": filename,
            "status": status,
            "processed_date": datetime.now().isoformat(),
            "chunks_count": chunks_count,
            "pages_count": pages_count
        })
        
        status_data["processed_files"] = processed_files
        status_data["last_scan_date"] = datetime.now().isoformat()
        
        # Update counts
        status_data["total_processed"] = len([f for f in processed_files if f["status"] == "completed"])
        status_data["total_failed"] = len([f for f in processed_files if f["status"] == "failed"])
        
        # Save to file
        try:
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Marked {filename} as {status}")
        except Exception as e:
            logger.error(f"❌ Error saving status file: {e}")
    
    def get_unprocessed_documents(self) -> List[Dict]:
        """
        Get list of documents that haven't been processed
        
        Returns:
            List of unprocessed document info dictionaries
        """
        all_docs = self.scan_for_documents()
        processed_filenames = {
            f["filename"] 
            for f in self.get_processed_files().get("processed_files", [])
            if f.get("status") == "completed"
        }
        
        unprocessed = [
            doc for doc in all_docs 
            if doc["filename"] not in processed_filenames
        ]
        
        logger.info(f"📋 Found {len(unprocessed)} unprocessed documents")
        return unprocessed


