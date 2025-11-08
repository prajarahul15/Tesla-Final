"""
Vector Store Service
Manages ChromaDB vector database for document embeddings
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
import chromadb

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages ChromaDB vector database"""
    
    def __init__(self, collection_name: str = "tesla_documents", persist_dir: str = None):
        """
        Initialize vector store
        
        Args:
            collection_name: Name of ChromaDB collection
            persist_dir: Directory to persist ChromaDB data
        """
        if persist_dir is None:
            # Default: backend/data/chroma_db
            backend_dir = Path(__file__).parent.parent.parent
            persist_dir = backend_dir / "data" / "chroma_db"
        
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client (NEW API)
        try:
            # Use PersistentClient for local persistent storage
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "description": "Tesla financial documents and reports"
                }
            )
            
            logger.info(f"📚 Vector store initialized: {collection_name}")
            logger.info(f"💾 Persist directory: {self.persist_dir}")
            
        except Exception as e:
            logger.error(f"❌ Error initializing ChromaDB: {e}")
            raise Exception(f"Failed to initialize vector store: {str(e)}")
    
    def _clean_metadata(self, metadata: Dict) -> Dict:
        """
        Clean metadata by removing None values and ensuring valid types
        
        ChromaDB only accepts: Bool, Int, Float, Str (not None)
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Cleaned metadata dictionary with no None values
        """
        cleaned = {}
        for key, value in metadata.items():
            # Skip None values
            if value is None:
                continue
            
            # Ensure value is one of ChromaDB's accepted types
            if isinstance(value, (bool, int, float, str)):
                cleaned[key] = value
            elif isinstance(value, (list, dict)):
                # Skip complex types (ChromaDB doesn't support them in metadata)
                continue
            else:
                # Convert to string if possible
                try:
                    cleaned[key] = str(value)
                except:
                    # Skip if can't convert
                    continue
        
        return cleaned
    
    def add_documents(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: Optional[List[str]] = None
    ):
        """
        Add documents to vector store
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: Optional list of document IDs (auto-generated if not provided)
        """
        if not chunks or not embeddings:
            logger.warning("⚠️ No chunks or embeddings provided")
            return
        
        if len(chunks) != len(embeddings) or len(chunks) != len(metadatas):
            raise ValueError("chunks, embeddings, and metadatas must have same length")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"chunk_{i}_{metadatas[i].get('source', 'unknown')}" 
                   for i in range(len(chunks))]
        
        # Clean metadata to remove None values and ensure valid types
        cleaned_metadatas = [self._clean_metadata(meta) for meta in metadatas]
        
        try:
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                ids=ids,
                metadatas=cleaned_metadatas
            )
            
            logger.info(f"✅ Added {len(chunks)} documents to vector store")
            
        except Exception as e:
            logger.error(f"❌ Error adding documents to vector store: {e}")
            raise Exception(f"Failed to add documents: {str(e)}")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Deprecated, use 'where' instead
            where: Metadata filter (e.g., {"year": 2024})
            
        Returns:
            List of search result dictionaries
        """
        try:
            # Use 'where' parameter for filtering (newer ChromaDB API)
            query_filter = where if where else filters
            
            # ChromaDB has limitations with multiple conditions
            # If we have multiple conditions, use only the first one and handle others in post-processing
            # This is a workaround for ChromaDB's "Expected where to have exactly one operator" error
            if query_filter and len(query_filter) > 1:
                logger.warning(f"⚠️ Multiple filter conditions detected ({list(query_filter.keys())}). Using first condition only. Others will be handled in post-processing.")
                # Use only the first condition (typically file_id is most important)
                first_key = list(query_filter.keys())[0]
                query_filter = {first_key: query_filter[first_key]}
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=query_filter if query_filter else None  # Metadata filtering
            )
            
            # Format results
            search_results = []
            if results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    search_results.append({
                        "text": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i],
                        "id": results['ids'][0][i]
                    })
            
            logger.info(f"🔍 Found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"❌ Error searching vector store: {e}")
            raise Exception(f"Failed to search vector store: {str(e)}")
    
    def delete_by_metadata(self, where: Dict):
        """
        Delete documents matching metadata filter
        
        Args:
            where: Metadata filter (e.g., {"source": "10K Report 2024.pdf"})
        """
        try:
            # ChromaDB delete with metadata filter
            self.collection.delete(where=where)
            logger.info(f"🗑️ Deleted documents matching filter: {where}")
            
        except Exception as e:
            logger.error(f"❌ Error deleting documents: {e}")
            raise Exception(f"Failed to delete documents: {str(e)}")
    
    def get_collection_count(self) -> int:
        """
        Get total number of documents in collection
        
        Returns:
            Number of documents
        """
        try:
            count = self.collection.count()
            return count
        except Exception as e:
            logger.error(f"❌ Error getting collection count: {e}")
            return 0
    
    def get_collection_info(self) -> Dict:
        """
        Get collection information
        
        Returns:
            Dictionary with collection metadata
        """
        try:
            count = self.collection.count()
            return {
                "name": self.collection.name,
                "count": count,
                "metadata": self.collection.metadata
            }
        except Exception as e:
            logger.error(f"❌ Error getting collection info: {e}")
            return {}

