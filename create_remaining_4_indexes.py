"""
Direct script to create the 4 remaining indexes.
This script creates them with explicit normalized names.

Run: python create_remaining_4_indexes.py --api-key YOUR_API_KEY
"""

import os
import sys
import logging
import time
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_api_key(api_key_arg=None):
    """Get API key from various sources."""
    api_key = (
        api_key_arg or
        os.getenv('PINECONE_API_KEY') or
        os.getenv('ATS_PINECONE_API_KEY')
    )
    
    if not api_key:
        logger.error("❌ PINECONE_API_KEY is required!")
        logger.error("   Set it as: $env:PINECONE_API_KEY='your-key'")
        logger.error("   Or pass as: python create_remaining_4_indexes.py --api-key your-key")
        sys.exit(1)
    
    return api_key


def create_index(pc, index_name, dimension=1536):
    """Create a single index."""
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Creating index: '{index_name}'")
        logger.info(f"{'='*60}")
        
        # Check if index exists
        existing_indexes = pc.list_indexes()
        existing_names = [idx.name for idx in existing_indexes]
        
        if index_name in existing_names:
            logger.info(f"ℹ️  Index '{index_name}' already exists!")
            return True
        
        # Create the index
        logger.info(f"Creating new index '{index_name}' (dimension: {dimension})...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        
        # Wait for index to be ready
        logger.info("Waiting for index to initialize...")
        time.sleep(3)
        
        # Verify creation
        index_info = pc.describe_index(index_name)
        logger.info(f"✅ Successfully created '{index_name}' index!")
        logger.info(f"   Status: {index_info.status}")
        logger.info(f"   Dimension: {index_info.dimension}")
        
        return True
        
    except Exception as e:
        error_str = str(e).lower()
        if "already exists" in error_str or "duplicate" in error_str:
            logger.info(f"ℹ️  Index '{index_name}' already exists")
            return True
        else:
            logger.error(f"❌ Failed to create '{index_name}': {e}")
            return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create 4 remaining indexes')
    parser.add_argument('--api-key', type=str, help='Pinecone API key')
    args = parser.parse_args()
    
    # The 4 indexes to create (with exact normalized names)
    indexes_to_create = [
        'business-analyst',
        'project-manager',
        'sql',
        'others'
    ]
    
    logger.info("=" * 60)
    logger.info("Creating 4 Remaining Indexes")
    logger.info("=" * 60)
    logger.info(f"Indexes: {', '.join(indexes_to_create)}")
    
    try:
        # Get API key
        api_key = get_api_key(args.api_key)
        
        # Initialize Pinecone
        logger.info("\nConnecting to Pinecone...")
        pc = Pinecone(api_key=api_key)
        logger.info("✅ Connected to Pinecone")
        
        # Create each index
        results = {}
        for index_name in indexes_to_create:
            success = create_index(pc, index_name)
            results[index_name] = success
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Summary")
        logger.info("=" * 60)
        
        created = [name for name, success in results.items() if success]
        failed = [name for name, success in results.items() if not success]
        
        if created:
            logger.info(f"\n✅ Successfully created/exists ({len(created)}):")
            for name in created:
                logger.info(f"   - {name}")
        
        if failed:
            logger.error(f"\n❌ Failed ({len(failed)}):")
            for name in failed:
                logger.error(f"   - {name}")
        
        # List all indexes
        logger.info("\n" + "=" * 60)
        logger.info("All indexes in your Pinecone account:")
        logger.info("=" * 60)
        all_indexes = pc.list_indexes()
        index_names = [idx.name for idx in all_indexes]
        
        # Check which ones we have
        required = ['java', 'python', 'dotnet', 'business-analyst', 'project-manager', 'sql', 'others']
        for idx in sorted(index_names):
            marker = "✅" if idx in required else "  "
            logger.info(f"{marker} {idx}")
        
        # Final check
        missing = [idx for idx in required if idx not in index_names]
        if missing:
            logger.warning(f"\n⚠️  Still missing: {', '.join(missing)}")
        else:
            logger.info(f"\n✅ All 7 required indexes are now created!")
            logger.info(f"   Total indexes: {len(index_names)}")
        
    except Exception as e:
        logger.error(f"\n❌ Script failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

