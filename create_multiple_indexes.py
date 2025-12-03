"""
Create multiple Pinecone indexes for different profile types.
Run: python create_multiple_indexes.py
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


def create_all_indexes(api_key: str = None):
    """Create all required indexes."""
    
    # List of indexes to create
    indexes_to_create = [
        'python',
        'dotnet',  # .net (normalized)
        'business-analyst',
        'sql',
        'project-manager',
        'others'
    ]
    
    logger.info("=" * 60)
    logger.info("Creating Multiple Pinecone Indexes")
    logger.info("=" * 60)
    logger.info(f"Indexes to create: {', '.join(indexes_to_create)}")
    logger.info("")
    
    try:
        # Initialize the index creator
        creator = DynamicPineconeIndexCreator(api_key=api_key)
        
        results = {}
        
        for index_name in indexes_to_create:
            logger.info("-" * 60)
            logger.info(f"Creating index: '{index_name}'")
            logger.info("-" * 60)
            
            try:
                success = creator.create_index(
                    index_name=index_name,
                    dimension=1536,
                    metric='cosine',
                    cloud='aws',
                    region='us-east-1'
                )
                
                if success:
                    logger.info(f"✅ Successfully created '{index_name}' index!")
                    results[index_name] = 'created'
                else:
                    logger.info(f"ℹ️  '{index_name}' index already exists")
                    results[index_name] = 'exists'
                
                # Get index information
                info = creator.get_index_info(index_name)
                if info:
                    logger.info(f"   Dimension: {info['dimension']}")
                    logger.info(f"   Metric: {info['metric']}")
                    logger.info(f"   Status: {info['status']}")
                
            except Exception as e:
                logger.error(f"❌ Failed to create '{index_name}' index: {e}")
                results[index_name] = 'failed'
            
            logger.info("")
        
        # Summary
        logger.info("=" * 60)
        logger.info("Summary")
        logger.info("=" * 60)
        
        created = [name for name, status in results.items() if status == 'created']
        exists = [name for name, status in results.items() if status == 'exists']
        failed = [name for name, status in results.items() if status == 'failed']
        
        if created:
            logger.info(f"\n✅ Created ({len(created)}): {', '.join(created)}")
        if exists:
            logger.info(f"\nℹ️  Already exists ({len(exists)}): {', '.join(exists)}")
        if failed:
            logger.info(f"\n❌ Failed ({len(failed)}): {', '.join(failed)}")
        
        logger.info("\n" + "=" * 60)
        logger.info("All existing indexes in your Pinecone account:")
        logger.info("=" * 60)
        all_indexes = creator.list_all_indexes()
        for idx in all_indexes:
            logger.info(f"  - {idx}")
        
        logger.info("\n✅ Script completed!")
        
        return len(failed) == 0
        
    except Exception as e:
        logger.error(f"❌ Script failed: {e}")
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create multiple Pinecone indexes')
    parser.add_argument('--api-key', type=str, help='Pinecone API key (optional if set as environment variable)')
    args = parser.parse_args()
    
    try:
        create_all_indexes(api_key=args.api_key)
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

