"""
Embedding Service
Generates embeddings for text chunks using OpenAI
"""

import os
import logging
from typing import List, Optional
from openai import OpenAI
import time

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generates embeddings for text using OpenAI"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize embedding service
        
        Args:
            model: OpenAI embedding model to use
        """
        self.model = model
        self.client = None
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.client = OpenAI(api_key=api_key)
            logger.info(f"🤖 Embedding service initialized with model: {model}")
        else:
            logger.error("❌ OPENAI_API_KEY not found in environment")
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Rate limiting (OpenAI allows 3000 requests per minute)
        self.requests_per_minute = 3000
        self.request_times = []
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text chunk
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector (list of floats)
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            # Handle rate limiting
            self._wait_for_rate_limit()
            
            response = self.client.embeddings.create(
                model=self.model,
                input=text[:8000]  # Limit to 8000 tokens (safety limit)
            )
            
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Error generating embedding: {e}")
            raise Exception(f"Failed to generate embedding: {str(e)}")
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple text chunks in batches
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process per batch
            
        Returns:
            List of embedding vectors
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        embeddings = []
        total = len(texts)
        
        logger.info(f"🤖 Generating embeddings for {total} chunks (batch_size={batch_size})")
        
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total + batch_size - 1) // batch_size
            
            try:
                # Handle rate limiting
                self._wait_for_rate_limit()
                
                # Generate embeddings for batch
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                logger.info(f"  ✅ Batch {batch_num}/{total_batches}: {len(batch)} embeddings generated")
                
                # Small delay between batches to avoid rate limits
                if i + batch_size < total:
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"❌ Error generating batch {batch_num}: {e}")
                # Continue with next batch
                # Add empty embeddings for failed batch
                embeddings.extend([[] for _ in batch])
        
        logger.info(f"✅ Generated {len(embeddings)} embeddings total")
        return embeddings
    
    def _wait_for_rate_limit(self):
        """
        Wait if we're approaching rate limit
        Simple rate limiting implementation
        """
        now = time.time()
        
        # Remove request times older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # If we're close to rate limit, wait
        if len(self.request_times) >= self.requests_per_minute * 0.9:
            wait_time = 60 - (now - self.request_times[0])
            if wait_time > 0:
                logger.info(f"⏳ Rate limit approaching, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                # Clean up again after waiting
                self.request_times = [t for t in self.request_times if now + wait_time - t < 60]
        
        # Record this request
        self.request_times.append(now)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings from this model
        
        Returns:
            Embedding dimension
        """
        # text-embedding-3-small: 1536 dimensions
        # text-embedding-3-large: 3072 dimensions
        dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dimension_map.get(self.model, 1536)


