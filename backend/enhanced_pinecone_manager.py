"""
Enhanced Pinecone Manager with Dynamic Index Management
Handles Pinecone v3 SDK integration with comprehensive error handling and logging.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec

# Try to import PineconeException, fallback to generic Exception if not available
try:
    from pinecone.exceptions import PineconeException
except ImportError:
    try:
        from pinecone.core.client.exceptions import PineconeException
    except ImportError:
        # Fallback: use generic Exception
        PineconeException = Exception

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
                # Index doesn't exist - check if we've hit the limit
                logger.warning(f"Index '{self.index_name}' not found. Checking available indexes...")
                
                # Try to use an existing index as fallback
                if index_names:
                    # Find an index with matching dimension, or use the first one
                    fallback_index = None
                    for idx_name in index_names:
                        try:
                            temp_index = self.pc.Index(idx_name)
                            stats = temp_index.describe_index_stats()
                            if stats.dimension == self.dimension:
                                fallback_index = idx_name
                                break
                        except:
                            continue
                    
                    # If no matching dimension found, use the first index
                    if not fallback_index:
                        fallback_index = index_names[0]
                        logger.warning(
                            f"⚠️  No index found with dimension {self.dimension}.\n"
                            f"   Using '{fallback_index}' (may have different dimension)."
                        )
                    
                    logger.warning(
                        f"⚠️  Index '{self.index_name}' does not exist.\n"
                        f"   You have {len(index_names)} existing index(es): {index_names}\n"
                        f"   Using fallback index: '{fallback_index}'\n"
                        f"   💡 Tip: Set PINECONE_INDEX_NAME environment variable to '{fallback_index}' to avoid this warning."
                    )
                    self.index_name = fallback_index
                    self.index = self.pc.Index(fallback_index)
                    
                    # Verify dimension (warn but don't fail)
                    try:
                        index_stats = self.index.describe_index_stats()
                        existing_dimension = index_stats.dimension
                        if existing_dimension != self.dimension:
                            logger.warning(
                                f"⚠️  Dimension mismatch: Index '{fallback_index}' has dimension {existing_dimension}, "
                                f"but expected {self.dimension}. This may cause issues."
                            )
                        else:
                            logger.info(f"✅ Connected to fallback index '{fallback_index}' with matching dimension {existing_dimension}")
                    except:
                        logger.info(f"Connected to fallback index: {fallback_index}")
                else:
                    # No indexes exist - try to create one, but handle limit errors gracefully
                    logger.info(f"No indexes found. Attempting to create '{self.index_name}'...")
                    try:
                        self._create_new_index()
                    except Exception as create_error:
                        error_str = str(create_error).lower()
                        # Check if it's a 403/limit error
                        if "403" in str(create_error) or "forbidden" in error_str or "max serverless indexes" in error_str:
                            # Try to get the default index name
                            try:
                                from ats_config import ATSConfig
                                default_index = ATSConfig.PINECONE_INDEX_NAME
                            except:
                                default_index = 'ats-resumes'
                            
                            error_msg = (
                                f"Failed to create Pinecone index '{self.index_name}': {create_error}\n"
                                f"⚠️  You've reached the Pinecone free tier limit of 5 indexes.\n"
                                f"💡 Solutions:\n"
                                f"   1. Delete unused indexes from your Pinecone dashboard\n"
                                f"   2. Use an existing index by setting PINECONE_INDEX_NAME to one of your existing indexes\n"
                                f"   3. Upgrade your Pinecone plan for more indexes\n"
                                f"   4. The system uses namespaces, so you only need ONE index for all profile types."
                            )
                            logger.error(error_msg)
                            raise Exception(error_msg)
                        else:
                            # Re-raise other errors
                            raise
            
            self._index_initialized = True
            return self.index
            
        except Exception as e:
            error_msg = f"Pinecone API error: {e}"
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
            
        except Exception as e:
            error_str = str(e).lower()
            if "already exists" in error_str or "duplicate" in error_str:
                logger.warning(f"Index '{self.index_name}' already exists, connecting to it...")
                self.index = self.pc.Index(self.index_name)
            else:
                error_msg = f"Failed to create Pinecone index: {e}"
                logger.error(error_msg)
                raise Exception(error_msg)
    
    def upsert_vectors(self, vectors: List[Dict[str, Any]], namespace: str = None):
        """
        Insert vectors into Pinecone index with optional namespace support.
        
        Args:
            vectors: List of vector dictionaries with 'id', 'values', and 'metadata'
            namespace: Optional namespace name (e.g., 'java', 'python', 'business-analyst')
                      If None, vectors are inserted into default namespace
                      If provided, all vectors are inserted into that namespace
        """
        if not self._index_initialized:
            raise RuntimeError("Index not initialized. Call get_or_create_index() first.")
        
        if not vectors:
            logger.warning("No vectors provided for upsert operation")
            return
        
        try:
            namespace_info = f" namespace '{namespace}'" if namespace else " (default namespace)"
            logger.info(f"Upserting {len(vectors)} vectors to Pinecone index '{self.index_name}'{namespace_info}")
            
            # Prepare vectors with cleaned metadata
            prepared_vectors = []
            for i, vector in enumerate(vectors):
                if 'values' not in vector:
                    raise ValueError(f"Vector {i} missing 'values' field")
                
                vector_dim = len(vector['values'])
                if vector_dim != self.dimension:
                    raise ValueError(
                        f"Vector {i} has dimension {vector_dim}, expected {self.dimension}"
                    )
                
                # Prepare metadata to handle NULL values
                prepared_vector = {
                    'id': vector['id'],
                    'values': vector['values']
                }
                
                if 'metadata' in vector:
                    prepared_vector['metadata'] = self.prepare_metadata(vector['metadata'])
                else:
                    prepared_vector['metadata'] = {}
                
                prepared_vectors.append(prepared_vector)
            
            # Perform upsert operation with namespace
            if namespace:
                self.index.upsert(vectors=prepared_vectors, namespace=namespace)
                logger.info(f"Successfully upserted {len(prepared_vectors)} vectors to namespace '{namespace}'")
            else:
                self.index.upsert(vectors=prepared_vectors)
                logger.info(f"Successfully upserted {len(prepared_vectors)} vectors to default namespace")
            
        except Exception as e:
            error_msg = f"Pinecone upsert error: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def query_vectors(self, query_vector: List[float], top_k: int = 10, 
                     include_metadata: bool = True, filter: Dict[str, Any] = None,
                     namespace: str = None):
        """
        Query vectors from Pinecone index with optional namespace support.
        
        Args:
            query_vector: Query vector for similarity search
            top_k: Number of results to return
            include_metadata: Whether to include metadata in results
            filter: Optional metadata filter
            namespace: Optional namespace name to query (e.g., 'java', 'python')
                      If None, queries default namespace
                      If provided, queries only that namespace
            
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
            namespace_info = f" namespace '{namespace}'" if namespace else " (default namespace)"
            logger.info(f"Querying Pinecone index '{self.index_name}'{namespace_info} with top_k={top_k}")
            
            # Build query parameters
            query_params = {
                'vector': query_vector,
                'top_k': top_k,
                'include_metadata': include_metadata
            }
            
            # Add namespace if provided
            if namespace:
                query_params['namespace'] = namespace
            
            # Add filter if provided
            if filter:
                query_params['filter'] = filter
            
            results = self.index.query(**query_params)
            
            logger.info(f"Query returned {len(results.matches)} results")
            return results
            
        except Exception as e:
            error_msg = f"Pinecone query error: {e}"
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
    
    def prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare metadata for Pinecone by handling NULL values.
        Converts NULL values to appropriate default values.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Cleaned metadata dictionary safe for Pinecone
        """
        cleaned_metadata = {}
        
        for key, value in metadata.items():
            if value is None:
                # Convert NULL values to appropriate defaults
                if key in ['name', 'email', 'domain', 'education', 'file_type', 'source', 'profile_type']:
                    cleaned_metadata[key] = 'Unknown'
                elif key in ['primary_skills', 'secondary_skills', 'all_skills']:
                    cleaned_metadata[key] = 'No skills'
                elif key in ['current_location', 'preferred_locations']:
                    cleaned_metadata[key] = 'Unknown'
                elif key in ['current_company', 'current_designation']:
                    cleaned_metadata[key] = 'Unknown'
                elif key in ['notice_period', 'expected_salary', 'current_salary']:
                    cleaned_metadata[key] = 'Not specified'
                else:
                    # For other fields, use 'Unknown' as default
                    cleaned_metadata[key] = 'Unknown'
            elif isinstance(value, (str, int, float, bool)):
                # Valid Pinecone types - keep as is
                cleaned_metadata[key] = value
            elif isinstance(value, list):
                # Lists are allowed if they contain strings
                if all(isinstance(item, str) for item in value):
                    cleaned_metadata[key] = value
                else:
                    # Convert mixed lists to strings
                    cleaned_metadata[key] = ', '.join(str(item) for item in value)
            else:
                # Convert other types to strings
                cleaned_metadata[key] = str(value)
        
        return cleaned_metadata
    
    @staticmethod
    def create_dynamic_index(index_name: str, 
                            api_key: str = None,
                            dimension: int = 1536,
                            metric: str = 'cosine',
                            cloud: str = 'aws',
                            region: str = 'us-east-1') -> bool:
        """
        Static method to create a Pinecone index dynamically.
        
        Args:
            index_name: Name of the index to create (e.g., 'java', 'python')
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            dimension: Vector dimension (default: 1536)
            metric: Similarity metric ('cosine', 'euclidean', 'dotproduct')
            cloud: Cloud provider ('aws', 'gcp', 'azure')
            region: Region for the index
            
        Returns:
            True if index was created successfully, False if it already exists
        """
        try:
            # Get API key
            api_key = api_key or os.getenv('PINECONE_API_KEY')
            if not api_key:
                raise ValueError("PINECONE_API_KEY is required")
            
            # Initialize Pinecone client
            pc = Pinecone(api_key=api_key)
            
            # Validate and normalize index name
            index_name = index_name.lower().strip()
            if not index_name.replace('-', '').replace('_', '').isalnum():
                raise ValueError("Index name must contain only lowercase letters, numbers, hyphens, and underscores")
            
            logger.info(f"Checking if index '{index_name}' already exists...")
            
            # List all existing indexes
            existing_indexes = pc.list_indexes()
            existing_names = [idx.name for idx in existing_indexes]
            
            if index_name in existing_names:
                logger.warning(f"Index '{index_name}' already exists")
                return False
            
            logger.info(f"Creating new index '{index_name}' with dimension {dimension}...")
            
            # Create index with serverless specification
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )
            
            logger.info(f"✅ Successfully created index '{index_name}'")
            return True
            
        except Exception as e:
            error_str = str(e).lower()
            if "already exists" in error_str or "duplicate" in error_str:
                logger.warning(f"Index '{index_name}' already exists")
                return False
            else:
                error_msg = f"Failed to create Pinecone index '{index_name}': {e}"
                logger.error(error_msg)
                raise Exception(error_msg)


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
