"""
RAG Service
Main orchestrator for Retrieval-Augmented Generation
Coordinates document processing, storage, and retrieval
"""

import logging
import asyncio
from typing import List, Dict, Optional
from datetime import datetime

from .document_discovery import DocumentDiscovery
from .document_extractor import DocumentExtractor
from .text_chunker import TextChunker
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from .temporal_reasoning import TemporalReasoningService
from .citation_extractor import CitationExtractor
from .cache_service import CacheService

logger = logging.getLogger(__name__)


class RAGService:
    """Main RAG service orchestrator"""
    
    def __init__(self):
        """Initialize RAG service with all components"""
        self.document_discovery = DocumentDiscovery()
        self.document_extractor = DocumentExtractor(enable_tables=True)  # Enable multi-modal
        self.text_chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        self.embedding_service = EmbeddingService(model="text-embedding-3-small")
        self.vector_store = VectorStore(collection_name="tesla_documents")
        
        # Phase 3: Advanced features
        self.temporal_reasoning = TemporalReasoningService()
        self.citation_extractor = CitationExtractor()
        self.cache_service = CacheService(use_redis=False)  # In-memory cache by default
        
        logger.info("🚀 RAG Service initialized with Phase 3 features")
    
    async def process_document_async(self, document_info: Dict) -> Dict:
        """
        Async version of process_document for parallel processing
        
        Args:
            document_info: Document information from discovery service
            
        Returns:
            Processing result dictionary
        """
        # Run synchronous process in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_document, document_info)
    
    def process_document(self, document_info: Dict) -> Dict:
        """
        Process a single document: extract → chunk → embed → store
        
        Args:
            document_info: Document information from discovery service
            
        Returns:
            Processing result dictionary
        """
        filename = document_info["filename"]
        file_path = document_info["file_path"]
        
        logger.info(f"📄 Processing document: {filename}")
        
        try:
            # Step 1: Mark as processing
            self.document_discovery.mark_processed(
                filename=filename,
                chunks_count=0,
                status="processing"
            )
            
            # Step 2: Extract text and tables from PDF (multi-modal)
            logger.info(f"  📖 Extracting text and tables...")
            extracted = self.document_extractor.extract_from_pdf(
                pdf_path=file_path,
                metadata=document_info
            )
            
            # Step 3: Chunk the text (include tables if extracted)
            logger.info(f"  ✂️ Chunking text...")
            text_to_chunk = extracted.get("text", "")
            
            # If tables were extracted, include table text in chunks
            tables = extracted.get("tables", [])
            if tables:
                table_text = "\n\n".join([table.get("table_text", "") for table in tables])
                text_to_chunk = text_to_chunk + "\n\n=== TABLES ===\n\n" + table_text
                logger.info(f"  📊 Including {len(tables)} tables in chunking")
            
            chunks = self.text_chunker.chunk_document(
                text=text_to_chunk,
                metadata=extracted["metadata"]
            )
            
            if not chunks:
                raise ValueError("No chunks created from document")
            
            # Step 4: Generate embeddings
            logger.info(f"  🤖 Generating embeddings for {len(chunks)} chunks...")
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_service.generate_embeddings_batch(
                texts=chunk_texts,
                batch_size=100
            )
            
            # Step 5: Prepare metadata
            chunk_metadatas = [chunk["metadata"] for chunk in chunks]
            chunk_ids = [
                f"{filename}_chunk_{chunk['chunk_index']}"
                for chunk in chunks
            ]
            
            # Step 6: Store in vector database
            logger.info(f"  📚 Storing in vector database...")
            self.vector_store.add_documents(
                chunks=chunk_texts,
                embeddings=embeddings,
                metadatas=chunk_metadatas,
                ids=chunk_ids
            )
            
            # Step 7: Mark as completed
            self.document_discovery.mark_processed(
                filename=filename,
                chunks_count=len(chunks),
                pages_count=extracted.get("total_pages"),
                status="completed"
            )
            
            result = {
                "success": True,
                "filename": filename,
                "chunks_count": len(chunks),
                "pages_count": extracted.get("total_pages"),
                "total_characters": extracted.get("total_characters"),
                "processing_date": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Successfully processed {filename}: {len(chunks)} chunks")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing {filename}: {e}")
            
            # Mark as failed
            try:
                self.document_discovery.mark_processed(
                    filename=filename,
                    chunks_count=0,
                    status="failed"
                )
            except:
                pass
            
            return {
                "success": False,
                "filename": filename,
                "error": str(e),
                "processing_date": datetime.now().isoformat()
            }
    
    async def process_all_documents_async(self, force_reprocess: bool = False) -> Dict:
        """
        Async version: Process all unprocessed documents in parallel
        
        Args:
            force_reprocess: If True, reprocess even completed documents
            
        Returns:
            Processing summary dictionary
        """
        logger.info("🔄 Processing all documents (async)...")
        
        if force_reprocess:
            documents = self.document_discovery.scan_for_documents()
        else:
            documents = self.document_discovery.get_unprocessed_documents()
        
        if not documents:
            logger.info("📋 No documents to process")
            return {
                "success": True,
                "total_documents": 0,
                "processed": 0,
                "failed": 0,
                "results": []
            }
        
        logger.info(f"📋 Found {len(documents)} documents to process")
        
        # Process documents in parallel (limit concurrency)
        semaphore = asyncio.Semaphore(3)  # Process 3 documents concurrently
        
        async def process_with_semaphore(doc_info):
            async with semaphore:
                return await self.process_document_async(doc_info)
        
        tasks = [process_with_semaphore(doc_info) for doc_info in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed = 0
        failed = 0
        final_results = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Error processing document {i}: {result}")
                final_results.append({
                    "success": False,
                    "filename": documents[i].get("filename", "Unknown"),
                    "error": str(result)
                })
                failed += 1
            else:
                final_results.append(result)
                if result.get("success"):
                    processed += 1
                else:
                    failed += 1
        
        summary = {
            "success": True,
            "total_documents": len(documents),
            "processed": processed,
            "failed": failed,
            "results": final_results
        }
        
        logger.info(f"✅ Async processing complete: {processed} succeeded, {failed} failed")
        return summary
    
    def process_all_documents(self, force_reprocess: bool = False) -> Dict:
        """
        Process all unprocessed documents (synchronous, with async fallback option)
        
        Args:
            force_reprocess: If True, reprocess even completed documents
            
        Returns:
            Processing summary dictionary
        """
        logger.info("🔄 Processing all documents...")
        
        if force_reprocess:
            documents = self.document_discovery.scan_for_documents()
        else:
            documents = self.document_discovery.get_unprocessed_documents()
        
        if not documents:
            logger.info("📋 No documents to process")
            return {
                "success": True,
                "total_documents": 0,
                "processed": 0,
                "failed": 0,
                "results": []
            }
        
        logger.info(f"📋 Found {len(documents)} documents to process")
        
        results = []
        processed = 0
        failed = 0
        
        for doc_info in documents:
            result = self.process_document(doc_info)
            results.append(result)
            
            if result["success"]:
                processed += 1
            else:
                failed += 1
        
        summary = {
            "success": True,
            "total_documents": len(documents),
            "processed": processed,
            "failed": failed,
            "results": results
        }
        
        logger.info(f"✅ Processing complete: {processed} succeeded, {failed} failed")
        return summary
    
    def retrieve_documents(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query with caching and temporal reasoning
        
        Args:
            query: User query text
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {"year": 2024})
            
        Returns:
            List of retrieved document chunks with metadata
        """
        try:
            # Check cache first
            cache_key = {"query": query, "top_k": top_k, "filters": filters}
            cached_results = self.cache_service.get("retrieval", cache_key)
            if cached_results:
                logger.info(f"💾 Cache hit for query")
                return cached_results
            
            # Detect temporal aspects of query
            temporal_info = self.temporal_reasoning.detect_temporal_query(query)
            
            # Build temporal filters if needed
            temporal_filters = self.temporal_reasoning.build_temporal_filters(temporal_info)
            if temporal_filters:
                # Merge with user-provided filters
                filters = {**(filters or {}), **temporal_filters}
                logger.info(f"🕐 Applied temporal filters: {filters}")
            
            # Adjust top_k for temporal queries
            adjusted_top_k = self.temporal_reasoning.enhance_retrieval_for_temporal(temporal_info, top_k)
            
            # Generate query embedding
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=adjusted_top_k,
                where=filters
            )
            
            # If temporal query, add temporal context to results
            if temporal_info.get("has_temporal"):
                for result in results:
                    result["temporal_info"] = temporal_info
            
            # Cache results
            self.cache_service.set("retrieval", cache_key, results, ttl=3600)
            
            logger.info(f"🔍 Retrieved {len(results)} documents for query (temporal: {temporal_info.get('has_temporal')})")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error retrieving documents: {e}")
            return []
    
    def extract_and_verify_citations(self, response_text: str, retrieved_docs: List[Dict]) -> Dict:
        """
        Extract citations from response and verify against retrieved documents
        
        Args:
            response_text: AI-generated response text
            retrieved_docs: Documents retrieved for the query
            
        Returns:
            Citation verification results
        """
        try:
            # Validate input
            if not response_text or not isinstance(response_text, str):
                logger.warning("⚠️ Invalid response_text for citation extraction")
                return {
                    "total_citations": 0,
                    "verified_count": 0,
                    "unverified_count": 0,
                    "verified_citations": [],
                    "unverified_citations": []
                }
            
            # Ensure retrieved_docs is a list
            if not retrieved_docs:
                retrieved_docs = []
            
            # Extract citations
            citations = self.citation_extractor.extract_citations(response_text)
            
            # Verify citations
            verified_citations = self.citation_extractor.verify_citations(citations, retrieved_docs)
            
            # Format for display
            formatted = self.citation_extractor.format_citations_for_display(verified_citations)
            
            logger.info(f"📝 Citations: {formatted['verified_count']}/{formatted['total_citations']} verified")
            return formatted
            
        except Exception as e:
            logger.error(f"❌ Error extracting/verifying citations: {e}", exc_info=True)
            return {
                "total_citations": 0,
                "verified_count": 0,
                "unverified_count": 0,
                "verified_citations": [],
                "unverified_citations": [],
                "error": str(e)
            }
    
    def get_status(self) -> Dict:
        """
        Get RAG service status
        
        Returns:
            Status dictionary with processing info and collection stats
        """
        try:
            # Get processing status
            processing_status = self.document_discovery.get_processed_files()
            
            # Get vector store info
            collection_info = self.vector_store.get_collection_info()
            
            # Get cache stats
            cache_stats = self.cache_service.get_cache_stats()
            
            return {
                "processing": processing_status,
                "vector_store": collection_info,
                "cache": cache_stats,
                "status": "ready",
                "phase3_features": {
                    "multi_modal": True,
                    "temporal_reasoning": True,
                    "citation_verification": True,
                    "caching": True,
                    "async_processing": True
                }
            }
        except Exception as e:
            logger.error(f"❌ Error getting status: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

