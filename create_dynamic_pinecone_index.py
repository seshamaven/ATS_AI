"""
Dynamic Pinecone Index Creator
Creates Pinecone indexes dynamically based on profile types or custom names.
"""

import os
import logging
from typing import Optional, Dict, Any
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

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.debug("Loaded environment variables from .env file")
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass

from ats_config import ATSConfig


class DynamicPineconeIndexCreator:
    """
    Utility class for creating Pinecone indexes dynamically.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize with Pinecone API key.
        
        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
        """
        # Try multiple sources for API key
        self.api_key = (
            api_key or 
            os.getenv('PINECONE_API_KEY') or 
            os.getenv('ATS_PINECONE_API_KEY') or
            (ATSConfig.PINECONE_API_KEY if hasattr(ATSConfig, 'PINECONE_API_KEY') else None)
        )
        
        if not self.api_key:
            error_msg = """
❌ PINECONE_API_KEY is required but not found!

To fix this, you can:

1. Set environment variable (Windows PowerShell):
   $env:PINECONE_API_KEY="your-api-key-here"

2. Set environment variable (Windows CMD):
   set PINECONE_API_KEY=your-api-key-here

3. Set environment variable (Linux/Mac):
   export PINECONE_API_KEY=your-api-key-here

4. Create a .env file in the backend directory with:
   PINECONE_API_KEY=your-api-key-here

5. Pass API key directly when creating the object:
   creator = DynamicPineconeIndexCreator(api_key="your-api-key-here")

Get your API key from: https://app.pinecone.io/
"""
            logger.error(error_msg)
            raise ValueError("PINECONE_API_KEY is required. See error message above for setup instructions.")
        
        # Initialize Pinecone client
        try:
            self.pc = Pinecone(api_key=self.api_key)
            logger.info("Successfully initialized Pinecone client")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {e}")
            raise
    
    def create_index(self, 
                    index_name: str,
                    dimension: int = 1536,
                    metric: str = 'cosine',
                    cloud: str = 'aws',
                    region: str = 'us-east-1') -> bool:
        """
        Create a new Pinecone index dynamically.
        
        Args:
            index_name: Name of the index to create (e.g., 'java', 'python', 'ats-java-resumes')
            dimension: Vector dimension (default: 1536 for OpenAI embeddings)
            metric: Similarity metric ('cosine', 'euclidean', 'dotproduct')
            cloud: Cloud provider ('aws', 'gcp', 'azure')
            region: Region for the index
            
        Returns:
            True if index was created successfully, False if it already exists
        """
        try:
            # Validate index name
            if not index_name or not isinstance(index_name, str):
                raise ValueError("Index name must be a non-empty string")
            
            # Pinecone index names must be lowercase and alphanumeric with hyphens
            index_name = index_name.lower().strip()
            if not index_name.replace('-', '').replace('_', '').isalnum():
                raise ValueError("Index name must contain only lowercase letters, numbers, hyphens, and underscores")
            
            logger.info(f"Checking if index '{index_name}' already exists...")
            
            # List all existing indexes
            existing_indexes = self.pc.list_indexes()
            existing_names = [idx.name for idx in existing_indexes]
            
            if index_name in existing_names:
                logger.warning(f"Index '{index_name}' already exists")
                logger.info(f"Index details: {self.pc.describe_index(index_name)}")
                return False
            
            logger.info(f"Creating new index '{index_name}' with dimension {dimension}...")
            
            # Create index with serverless specification
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )
            
            logger.info(f"✅ Successfully created index '{index_name}'")
            logger.info(f"   Dimension: {dimension}")
            logger.info(f"   Metric: {metric}")
            logger.info(f"   Cloud: {cloud}")
            logger.info(f"   Region: {region}")
            
            # Wait a moment and verify the index
            import time
            time.sleep(2)  # Give Pinecone a moment to initialize
            
            index_info = self.pc.describe_index(index_name)
            logger.info(f"Index status: {index_info.status}")
            
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
    
    def create_profile_type_index(self, profile_type: str, dimension: int = 1536) -> bool:
        """
        Create an index for a specific profile type (e.g., 'java', 'python', '.net').
        
        Args:
            profile_type: Profile type name (e.g., 'java', 'python', 'dotnet')
            dimension: Vector dimension (default: 1536)
            
        Returns:
            True if index was created successfully
        """
        # Normalize profile type name for index
        normalized_name = profile_type.lower().strip()
        
        # Replace special characters
        normalized_name = normalized_name.replace('.', '').replace(' ', '-').replace('/', '-')
        
        # Create index name with prefix
        index_name = f"ats-{normalized_name}-resumes"
        
        logger.info(f"Creating index for profile type: {profile_type} -> {index_name}")
        
        return self.create_index(index_name, dimension=dimension)
    
    def list_all_indexes(self) -> list:
        """
        List all existing Pinecone indexes.
        
        Returns:
            List of index names
        """
        try:
            indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in indexes]
            logger.info(f"Found {len(index_names)} indexes: {index_names}")
            return index_names
        except Exception as e:
            logger.error(f"Error listing indexes: {e}")
            return []
    
    def delete_index(self, index_name: str) -> bool:
        """
        Delete a Pinecone index.
        
        Args:
            index_name: Name of the index to delete
            
        Returns:
            True if index was deleted successfully
        """
        try:
            logger.warning(f"Deleting index '{index_name}'...")
            self.pc.delete_index(index_name)
            logger.info(f"✅ Successfully deleted index '{index_name}'")
            return True
        except Exception as e:
            error_str = str(e).lower()
            if "not found" in error_str or "does not exist" in error_str:
                logger.warning(f"Index '{index_name}' does not exist")
                return False
            else:
                error_msg = f"Failed to delete index '{index_name}': {e}"
                logger.error(error_msg)
                raise Exception(error_msg)
    
    def get_index_info(self, index_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific index.
        
        Args:
            index_name: Name of the index
            
        Returns:
            Dictionary with index information or None if not found
        """
        try:
            index_info = self.pc.describe_index(index_name)
            return {
                'name': index_info.name,
                'dimension': index_info.dimension,
                'metric': index_info.metric,
                'status': index_info.status,
                'host': index_info.host if hasattr(index_info, 'host') else None,
            }
        except Exception as e:
            error_str = str(e).lower()
            if "not found" in error_str or "does not exist" in error_str:
                logger.warning(f"Index '{index_name}' does not exist")
                return None
            else:
                logger.error(f"Error getting index info: {e}")
                return None


def create_java_index():
    """Create a Pinecone index specifically for Java profiles."""
    try:
        creator = DynamicPineconeIndexCreator()
        
        # Create index named 'java' or 'ats-java-resumes'
        index_name = 'java'
        dimension = ATSConfig.EMBEDDING_DIMENSION if hasattr(ATSConfig, 'EMBEDDING_DIMENSION') else 1536
        
        logger.info("=" * 60)
        logger.info("Creating Pinecone index for Java profiles")
        logger.info("=" * 60)
        
        success = creator.create_index(
            index_name=index_name,
            dimension=dimension,
            metric='cosine',
            cloud='aws',
            region='us-east-1'
        )
        
        if success:
            logger.info(f"✅ Successfully created index '{index_name}'")
        else:
            logger.info(f"ℹ️  Index '{index_name}' already exists")
        
        # Get index information
        info = creator.get_index_info(index_name)
        if info:
            logger.info(f"Index information: {info}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Failed to create Java index: {e}")
        raise


if __name__ == "__main__":
    """
    Main execution - creates a 'java' index as an example.
    """
    import sys
    
    try:
        if len(sys.argv) > 1:
            # Custom index name provided
            index_name = sys.argv[1]
            creator = DynamicPineconeIndexCreator()
            
            dimension = int(sys.argv[2]) if len(sys.argv) > 2 else 1536
            
            logger.info(f"Creating index: {index_name}")
            creator.create_index(index_name, dimension=dimension)
        else:
            # Default: create Java index
            logger.info("Creating default Java index...")
            create_java_index()
            
            # Also list all indexes
            creator = DynamicPineconeIndexCreator()
            logger.info("\n" + "=" * 60)
            logger.info("All existing indexes:")
            logger.info("=" * 60)
            indexes = creator.list_all_indexes()
            for idx in indexes:
                info = creator.get_index_info(idx)
                if info:
                    logger.info(f"  - {idx}: {info}")
        
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        sys.exit(1)

