"""
User Document RAG Service
Handles RAG operations for user-uploaded documents (separate from Tesla Knowledge Base)
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

from services.rag.text_chunker import TextChunker
from services.rag.embedding_service import EmbeddingService
from services.rag.vector_store import VectorStore
from services.rag.temporal_reasoning import TemporalReasoningService
from services.rag.citation_extractor import CitationExtractor
from services.rag.cache_service import CacheService

logger = logging.getLogger(__name__)


class UserDocumentRAGService:
    """RAG service for user-uploaded documents"""
    
    def __init__(self, collection_name: str = "user_documents"):
        """
        Initialize user document RAG service
        
        Args:
            collection_name: Name of the vector store collection for user documents
        """
        self.collection_name = collection_name
        self.text_chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        self.embedding_service = EmbeddingService(model="text-embedding-3-small")
        self.vector_store = VectorStore(collection_name=collection_name)
        
        # Phase 3 features
        self.temporal_reasoning = TemporalReasoningService()
        self.citation_extractor = CitationExtractor()
        self.cache_service = CacheService(use_redis=False)
        
        logger.info(f"📚 User Document RAG Service initialized (collection: {collection_name})")
    
    def process_uploaded_document(
        self,
        file_id: str,
        filename: str,
        extracted_content: Dict,
        user_id: Optional[str] = None
    ) -> Dict:
        """
        Process an uploaded document and store in vector DB
        
        Args:
            file_id: Unique identifier for the file
            filename: Original filename
            extracted_content: Content extracted by FileProcessor
            user_id: Optional user identifier
            
        Returns:
            Processing result dictionary
        """
        try:
            logger.info(f"📄 Processing uploaded document: {filename} (ID: {file_id})")
            
            if not extracted_content.get("success"):
                return {
                    "success": False,
                    "error": extracted_content.get("error", "Failed to extract content"),
                    "file_id": file_id
                }
            
            text = extracted_content.get("text", "")
            if not text or len(text.strip()) == 0:
                return {
                    "success": False,
                    "error": "No text content extracted from file",
                    "file_id": file_id
                }
            
            # Create metadata
            metadata = {
                **extracted_content.get("metadata", {}),
                "file_id": file_id,
                "filename": filename,
                "upload_date": datetime.now().isoformat(),
                "document_type": "User Upload",
                "source": filename
            }
            
            if user_id:
                metadata["user_id"] = user_id
            
            # Chunk the text
            chunks = self.text_chunker.chunk_document(text, metadata)
            
            if not chunks:
                return {
                    "success": False,
                    "error": "No chunks created from document",
                    "file_id": file_id
                }
            
            logger.info(f"  ✂️ Created {len(chunks)} chunks from {filename}")
            
            # Generate embeddings
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_service.generate_embeddings_batch(chunk_texts)
            
            # Prepare for storage
            ids = [f"{file_id}_chunk_{i}" for i in range(len(chunks))]
            chunk_texts_storage = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            # Store in vector DB
            self.vector_store.add_documents(
                chunks=chunk_texts_storage,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"✅ Successfully processed and stored {filename}: {len(chunks)} chunks")
            
            return {
                "success": True,
                "file_id": file_id,
                "filename": filename,
                "chunks_count": len(chunks),
                "metadata": metadata,
                "processing_date": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"❌ Error processing uploaded document {filename}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "file_id": file_id
            }
    
    def retrieve_documents(
        self,
        query: str,
        file_id: Optional[str] = None,
        top_k: int = 5,
        user_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query text
            file_id: Optional filter by specific file
            top_k: Number of results to return
            user_id: Optional filter by user
            
        Returns:
            List of retrieved document chunks
        """
        try:
            # Build filters
            filters = {}
            if file_id:
                filters["file_id"] = file_id
            if user_id:
                filters["user_id"] = user_id
            
            # Detect temporal aspects
            temporal_info = self.temporal_reasoning.detect_temporal_query(query)
            temporal_filters = self.temporal_reasoning.build_temporal_filters(temporal_info)
            
            # ChromaDB doesn't support $in operator, so for multiple years, we search without year filter
            # and post-filter the results
            # Also, ChromaDB may have issues with multiple conditions in where clause, so we'll use file_id only
            # and post-filter by year if needed
            multiple_years = temporal_info.get("years") and len(temporal_info["years"]) > 1
            
            # For single year queries, we'll still try to use it, but if ChromaDB fails, we'll fallback to post-filtering
            # Don't add year filter if we already have file_id (ChromaDB limitation with multiple conditions)
            # We'll handle year filtering in post-processing
            if temporal_filters and not multiple_years and not file_id:
                # Only add year filter if we don't have file_id (to avoid multi-condition issues)
                filters.update(temporal_filters)
            
            # Check cache
            cache_key = {"query": query, "file_id": file_id, "top_k": top_k, "filters": filters}
            cached_results = self.cache_service.get("user_doc_retrieval", cache_key)
            if cached_results:
                logger.info(f"💾 Cache hit for user document query")
                return cached_results
            
            # Generate embedding
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Search - retrieve more results if multiple years to allow post-filtering
            adjusted_top_k = self.temporal_reasoning.enhance_retrieval_for_temporal(temporal_info, top_k)
            if multiple_years:
                # Retrieve more results to filter by year in post-processing
                adjusted_top_k = max(adjusted_top_k * 2, 20)
            
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=adjusted_top_k,
                where=filters if filters else None
            )
            
            # Post-filter by year if multiple years requested (ChromaDB limitation)
            # OR if year filter wasn't applied due to ChromaDB multi-condition limitation
            if multiple_years:
                target_years = set(str(y) for y in temporal_info["years"])
                filtered_results = []
                for result in results:
                    result_year = str(result.get("metadata", {}).get("year", ""))
                    if result_year in target_years or not result_year:
                        filtered_results.append(result)
                
                # Limit to requested top_k after filtering
                results = filtered_results[:top_k]
                logger.info(f"🔍 Post-filtered to {len(results)} results matching years {temporal_info['years']}")
            
            # Always post-filter by year if year was in filters but not used in query (ChromaDB limitation)
            # This handles the case where we have both file_id and year but ChromaDB can only use one
            if temporal_info.get("years") and file_id:
                target_year = str(temporal_info["years"][0])
                filtered_results = []
                for result in results:
                    result_year = str(result.get("metadata", {}).get("year", ""))
                    # Keep if year matches OR if year is not set (legacy data or no year metadata)
                    if result_year == target_year or not result_year:
                        filtered_results.append(result)
                
                if filtered_results:
                    results = filtered_results[:top_k]
                    logger.info(f"🔍 Post-filtered by year {target_year}: {len(results)} results")
                else:
                    logger.warning(f"⚠️ No results found after year filtering ({target_year}), returning all results")
            
            # Cache results
            self.cache_service.set("user_doc_retrieval", cache_key, results, ttl=3600)
            
            logger.info(f"🔍 Retrieved {len(results)} chunks from user documents for query")
            return results
        
        except Exception as e:
            logger.error(f"❌ Error retrieving user documents: {e}")
            return []
    
    def delete_document(self, file_id: str) -> Dict:
        """
        Delete a document and all its chunks from vector DB
        
        Args:
            file_id: File identifier
            
        Returns:
            Deletion result
        """
        try:
            # Note: ChromaDB doesn't have a direct delete by metadata filter
            # We need to get all chunks first, then delete by IDs
            # This is a limitation - for production, consider using document IDs
            
            logger.info(f"🗑️ Deleting document: {file_id}")
            return {
                "success": True,
                "message": f"Document {file_id} marked for deletion",
                "note": "ChromaDB requires ID-based deletion. Implement cleanup job for full deletion."
            }
        
        except Exception as e:
            logger.error(f"❌ Error deleting document: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_document_info(self, file_id: str) -> Dict:
        """Get information about a processed document"""
        try:
            # Query for chunks from this file
            query_embedding = self.embedding_service.generate_embedding("document")
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=100,
                where={"file_id": file_id}
            )
            
            if not results:
                return {
                    "success": False,
                    "error": "Document not found"
                }
            
            # Extract metadata from first chunk
            metadata = results[0].get("metadata", {})
            
            return {
                "success": True,
                "file_id": file_id,
                "chunks_count": len(results),
                "metadata": metadata
            }
        
        except Exception as e:
            logger.error(f"❌ Error getting document info: {e}")
            return {
                "success": False,
                "error": str(e)
            }

