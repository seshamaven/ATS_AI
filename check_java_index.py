"""
Check if 'java' index exists in Pinecone.
Run: python check_java_index.py
"""

import os
import sys
import logging
from pinecone import Pinecone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_java_index():
    """Check if 'java' index exists in Pinecone."""
    try:
        # Get API key from environment or command line
        api_key = os.getenv('PINECONE_API_KEY') or os.getenv('ATS_PINECONE_API_KEY')
        
        if not api_key:
            # Try to get from command line argument
            if len(sys.argv) > 1:
                api_key = sys.argv[1]
            else:
                logger.error("❌ PINECONE_API_KEY not found!")
                logger.error("   Set it as: $env:PINECONE_API_KEY='your-key'")
                logger.error("   Or pass as: python check_java_index.py YOUR_API_KEY")
                return False
        
        logger.info("=" * 60)
        logger.info("Checking for 'java' index in Pinecone")
        logger.info("=" * 60)
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=api_key)
        logger.info("✅ Connected to Pinecone")
        
        # List all indexes
        indexes = pc.list_indexes()
        index_names = [idx.name for idx in indexes]
        
        logger.info(f"\nFound {len(index_names)} total indexes:")
        for idx_name in index_names:
            marker = "👉" if idx_name == "java" else "  "
            logger.info(f"{marker} {idx_name}")
        
        # Check specifically for 'java' index
        if "java" in index_names:
            logger.info("\n" + "=" * 60)
            logger.info("✅ 'java' index EXISTS in Pinecone!")
            logger.info("=" * 60)
            
            # Get detailed information
            try:
                index_info = pc.describe_index("java")
                logger.info("\nIndex Details:")
                logger.info(f"  Name: {index_info.name}")
                logger.info(f"  Dimension: {index_info.dimension}")
                logger.info(f"  Metric: {index_info.metric}")
                logger.info(f"  Status: {index_info.status}")
                if hasattr(index_info, 'host'):
                    logger.info(f"  Host: {index_info.host}")
                
                # Get stats if index is ready
                if index_info.status.get('ready', False) if isinstance(index_info.status, dict) else True:
                    try:
                        index = pc.Index("java")
                        stats = index.describe_index_stats()
                        logger.info(f"\nIndex Statistics:")
                        logger.info(f"  Total Vectors: {stats.total_vector_count}")
                        logger.info(f"  Namespaces: {len(stats.namespaces) if hasattr(stats, 'namespaces') else 'N/A'}")
                    except Exception as e:
                        logger.warning(f"  Could not get index stats: {e}")
                
            except Exception as e:
                logger.warning(f"Could not get detailed info: {e}")
            
            return True
        else:
            logger.info("\n" + "=" * 60)
            logger.info("❌ 'java' index DOES NOT EXIST in Pinecone")
            logger.info("=" * 60)
            logger.info("\n💡 To create it, run:")
            logger.info("   python create_java_index.py --api-key YOUR_API_KEY")
            return False
            
    except Exception as e:
        logger.error(f"\n❌ Error checking index: {e}")
        logger.error("\nPossible issues:")
        logger.error("  1. Invalid API key")
        logger.error("  2. No internet connection")
        logger.error("  3. Pinecone service unavailable")
        return False


if __name__ == "__main__":
    success = check_java_index()
    sys.exit(0 if success else 1)

