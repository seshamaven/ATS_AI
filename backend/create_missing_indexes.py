"""
Create the 4 missing indexes: Business Analyst, Project Manager, SQL, and Others.
Run: python create_missing_indexes.py --api-key YOUR_API_KEY
"""

import logging
import sys
import argparse
from create_all_pinecone_indexes import PineconeIndexCreator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Create the 4 missing indexes."""
    parser = argparse.ArgumentParser(description='Create missing Pinecone indexes')
    parser.add_argument('--api-key', type=str, help='Pinecone API key')
    args = parser.parse_args()
    
    # The 4 missing indexes
    missing_indexes = [
        'Business Analyst',      # Will normalize to 'business-analyst'
        'Project Manager',        # Will normalize to 'project-manager'
        'SQL',                    # Will normalize to 'sql'
        'Others'                  # Will normalize to 'others'
    ]
    
    logger.info("=" * 60)
    logger.info("Creating Missing Indexes")
    logger.info("=" * 60)
    logger.info(f"Indexes to create: {', '.join(missing_indexes)}")
    logger.info("")
    
    try:
        creator = PineconeIndexCreator(api_key=args.api_key)
        
        results = {}
        
        for index_name in missing_indexes:
            logger.info("-" * 60)
            logger.info(f"Processing: '{index_name}'")
            logger.info("-" * 60)
            
            # Check what it will normalize to
            normalized = creator.normalize_index_name(index_name)
            logger.info(f"Normalized name: '{normalized}'")
            
            # Check if already exists
            if creator.check_index_exists(index_name):
                logger.info(f"ℹ️  Index '{normalized}' already exists!")
                results[index_name] = {'status': 'exists', 'success': True}
            else:
                # Create the index
                result = creator.create_index(index_name)
                results[index_name] = result
            
            logger.info("")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("Summary")
        logger.info("=" * 60)
        
        created = [name for name, result in results.items() if result.get('status') == 'created']
        exists = [name for name, result in results.items() if result.get('status') == 'exists']
        failed = [name for name, result in results.items() if not result.get('success', False)]
        
        if created:
            logger.info(f"\n✅ Created ({len(created)}):")
            for name in created:
                normalized = creator.normalize_index_name(name)
                logger.info(f"   {name} -> {normalized}")
        
        if exists:
            logger.info(f"\nℹ️  Already exists ({len(exists)}):")
            for name in exists:
                normalized = creator.normalize_index_name(name)
                logger.info(f"   {name} -> {normalized}")
        
        if failed:
            logger.info(f"\n❌ Failed ({len(failed)}):")
            for name in failed:
                normalized = creator.normalize_index_name(name)
                error = results[name].get('error', 'Unknown error')
                logger.info(f"   {name} -> {normalized}: {error}")
        
        # List all indexes to verify
        logger.info("\n" + "=" * 60)
        logger.info("All indexes in your Pinecone account:")
        logger.info("=" * 60)
        all_indexes = creator.list_all_indexes()
        for idx in all_indexes:
            marker = "✅" if idx in ['business-analyst', 'project-manager', 'sql', 'others'] else "  "
            logger.info(f"{marker} {idx}")
        
        logger.info(f"\nTotal indexes: {len(all_indexes)}")
        
        # Check specifically for the 4 we need
        required = ['business-analyst', 'project-manager', 'sql', 'others']
        missing = [idx for idx in required if idx not in all_indexes]
        
        if missing:
            logger.warning(f"\n⚠️  Still missing: {', '.join(missing)}")
        else:
            logger.info("\n✅ All required indexes are now created!")
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

