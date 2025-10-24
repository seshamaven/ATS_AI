"""
Enhanced Pinecone Manager with Dynamic Index Management
Handles Pinecone v3 SDK integration with comprehensive error handling and logging.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.client.exceptions import PineconeException

# Configure logging
logger = logging.getLogger(__name__)


class EnhancedPineconeManager:
    """
    Enhanced Pinecone manager with dynamic index management using Pinecone SDK v3.
    Handles index creation, validation, and error handling with comprehensive logging.
    """
    
    def __init__(self, api_key: str = None, index_name: str = None, dimension: int = None):
        """
        Initialize Pinecone manager with environment variables.
        
        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            index_name: Index name (defaults to PINECONE_INDEX_NAME env var)
            dimension: Vector dimension (defaults to PINECONE_DIMENSION env var)
        """
        # Get configuration from environment variables
        self.api_key = api_key or os.getenv('PINECONE_API_KEY')
        self.index_name = index_name or os.getenv('PINECONE_INDEX_NAME', 'ats-resumes')
        self.dimension = dimension or int(os.getenv('PINECONE_DIMENSION', '1536'))
        self.cloud = os.getenv('PINECONE_CLOUD', 'aws')
        self.region = os.getenv('PINECONE_REGION', 'us-east-1')
        
        # Validate required configuration
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        if not self.index_name:
            raise ValueError("PINECONE_INDEX_NAME environment variable is required")
        
        if not isinstance(self.dimension, int) or self.dimension <= 0:
            raise ValueError(f"PINECONE_DIMENSION must be a positive integer, got: {self.dimension}")
        
        # Initialize Pinecone client
        try:
            self.pc = Pinecone(api_key=self.api_key)
            logger.info("Successfully initialized Pinecone client")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {e}")
            raise
        
        self.index = None
        self._index_initialized = False
    
    def get_or_create_index(self):
        """
        Check if index exists and create if necessary.
        Returns initialized Pinecone index client.
        """
        try:
            logger.info("Checking for existing Pinecone index...")
            
            # List all existing indexes
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
            
            logger.info(f"Found {len(index_names)} existing indexes: {index_names}")
            
            if self.index_name in index_names:
                logger.info(f"Using existing index: {self.index_name}")
                self.index = self.pc.Index(self.index_name)
                
                # Verify index dimensions match
                index_stats = self.index.describe_index_stats()
                existing_dimension = index_stats.dimension
                
                if existing_dimension != self.dimension:
                    error_msg = (
                        f"Index dimension mismatch! "
                        f"Existing index '{self.index_name}' has dimension {existing_dimension}, "
                        f"but expected dimension is {self.dimension}. "
                        f"Please update PINECONE_DIMENSION environment variable or use a different index name."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                logger.info(f"Successfully connected to existing index '{self.index_name}' with dimension {existing_dimension}")
                
            else:
                logger.info("Creating new index since none found...")
                self._create_new_index()
            
            self._index_initialized = True
            return self.index
            
        except PineconeException as e:
            error_msg = f"Pinecone API error: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during index management: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def _create_new_index(self):
        """Create a new Pinecone index with specified configuration."""
        try:
            logger.info(f"Creating new index '{self.index_name}' with dimension {self.dimension}")
            
            # Create index with serverless specification
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud=self.cloud,
                    region=self.region
                )
            )
            
            # Wait for index to be ready
            logger.info("Waiting for index to be ready...")
            self.pc.describe_index(self.index_name)
            
            # Initialize index client
            self.index = self.pc.Index(self.index_name)
            
            logger.info(f"Successfully created and initialized index '{self.index_name}' with dimension {self.dimension}")
            
        except PineconeException as e:
            if "already exists" in str(e).lower():
                logger.warning(f"Index '{self.index_name}' already exists, connecting to it...")
                self.index = self.pc.Index(self.index_name)
            else:
                error_msg = f"Failed to create Pinecone index: {e}"
                logger.error(error_msg)
                raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error creating index: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def upsert_vectors(self, vectors: List[Dict[str, Any]]):
        """
        Insert vectors into Pinecone index.
        
        Args:
            vectors: List of vector dictionaries with 'id', 'values', and 'metadata'
        """
        if not self._index_initialized:
            raise RuntimeError("Index not initialized. Call get_or_create_index() first.")
        
        if not vectors:
            logger.warning("No vectors provided for upsert operation")
            return
        
        try:
            logger.info(f"Upserting {len(vectors)} vectors to Pinecone index '{self.index_name}'")
            
            # Validate vector dimensions
            for i, vector in enumerate(vectors):
                if 'values' not in vector:
                    raise ValueError(f"Vector {i} missing 'values' field")
                
                vector_dim = len(vector['values'])
                if vector_dim != self.dimension:
                    raise ValueError(
                        f"Vector {i} has dimension {vector_dim}, expected {self.dimension}"
                    )
            
            # Perform upsert operation
            self.index.upsert(vectors=vectors)
            logger.info(f"Successfully upserted {len(vectors)} vectors to Pinecone")
            
        except PineconeException as e:
            error_msg = f"Pinecone upsert error: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during vector upsert: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def query_vectors(self, query_vector: List[float], top_k: int = 10, 
                     include_metadata: bool = True, filter: Dict[str, Any] = None):
        """
        Query vectors from Pinecone index.
        
        Args:
            query_vector: Query vector for similarity search
            top_k: Number of results to return
            include_metadata: Whether to include metadata in results
            filter: Optional metadata filter
            
        Returns:
            Query results from Pinecone
        """
        if not self._index_initialized:
            raise RuntimeError("Index not initialized. Call get_or_create_index() first.")
        
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query vector has dimension {len(query_vector)}, expected {self.dimension}"
            )
        
        try:
            logger.info(f"Querying Pinecone index '{self.index_name}' with top_k={top_k}")
            
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=include_metadata,
                filter=filter
            )
            
            logger.info(f"Query returned {len(results.matches)} results")
            return results
            
        except PineconeException as e:
            error_msg = f"Pinecone query error: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during vector query: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def get_index_stats(self):
        """Get statistics about the current index."""
        if not self._index_initialized:
            raise RuntimeError("Index not initialized. Call get_or_create_index() first.")
        
        try:
            stats = self.index.describe_index_stats()
            logger.info(f"Index '{self.index_name}' stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            raise
    
    def delete_index(self, confirm: bool = False):
        """
        Delete the Pinecone index.
        
        Args:
            confirm: Must be True to actually delete the index
        """
        if not confirm:
            raise ValueError("Index deletion requires confirm=True")
        
        try:
            logger.warning(f"Deleting Pinecone index '{self.index_name}'")
            self.pc.delete_index(self.index_name)
            logger.info(f"Successfully deleted index '{self.index_name}'")
            self._index_initialized = False
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            raise


def create_pinecone_manager() -> EnhancedPineconeManager:
    """
    Factory function to create a Pinecone manager with environment variables.
    
    Returns:
        Initialized EnhancedPineconeManager instance
    """
    try:
        manager = EnhancedPineconeManager()
        manager.get_or_create_index()
        return manager
    except Exception as e:
        logger.error(f"Failed to create Pinecone manager: {e}")
        raise


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Test Pinecone manager creation
        print("Testing Enhanced Pinecone Manager...")
        manager = create_pinecone_manager()
        
        # Get index stats
        stats = manager.get_index_stats()
        print(f"Index stats: {stats}")
        
        print("✅ Pinecone manager test completed successfully!")
        
    except Exception as e:
        print(f"❌ Pinecone manager test failed: {e}")
