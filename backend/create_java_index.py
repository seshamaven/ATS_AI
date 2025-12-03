"""
Simple script to create a Pinecone index named 'java'.
Run this script to create the index: python create_java_index.py

You can also pass the API key as an argument:
    python create_java_index.py --api-key YOUR_API_KEY
"""

import logging
import sys
import argparse
from create_dynamic_pinecone_index import DynamicPineconeIndexCreator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Create a Pinecone index named 'java'."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create a Pinecone index named "java"')
    parser.add_argument('--api-key', type=str, help='Pinecone API key (optional if set as environment variable)')
    args = parser.parse_args()
    
    try:
        logger.info("=" * 60)
        logger.info("Creating Pinecone Index: 'java'")
        logger.info("=" * 60)
        
        # Initialize the index creator with API key from command line or environment
        api_key = args.api_key if args.api_key else None
        creator = DynamicPineconeIndexCreator(api_key=api_key)
        
        # Create the 'java' index with default dimension (1536)
        success = creator.create_index(
            index_name='java',
            dimension=1536,
            metric='cosine',
            cloud='aws',
            region='us-east-1'
        )
        
        if success:
            logger.info("✅ Successfully created 'java' index!")
        else:
            logger.info("ℹ️  'java' index already exists")
        
        # Get and display index information
        info = creator.get_index_info('java')
        if info:
            logger.info("\nIndex Information:")
            logger.info(f"  Name: {info['name']}")
            logger.info(f"  Dimension: {info['dimension']}")
            logger.info(f"  Metric: {info['metric']}")
            logger.info(f"  Status: {info['status']}")
        
        # List all indexes
        logger.info("\n" + "=" * 60)
        logger.info("All existing indexes:")
        logger.info("=" * 60)
        indexes = creator.list_all_indexes()
        for idx in indexes:
            logger.info(f"  - {idx}")
        
        logger.info("\n✅ Script completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to create index: {e}")
        raise


if __name__ == "__main__":
    main()

