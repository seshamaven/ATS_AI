"""
Setup Pinecone namespaces for profile types.
Since you've reached the 5-index limit, we'll use namespaces within a single index instead.

This is actually BETTER than separate indexes - Pinecone recommends using namespaces!

Usage:
    python setup_pinecone_namespaces.py --api-key YOUR_API_KEY
"""

import os
import sys
import logging
import argparse
from pinecone import Pinecone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# All profile type namespaces
PROFILE_NAMESPACES = [
    'java',
    'python',
    'dotnet',
    'business-analyst',
    'project-manager',
    'sql',
    'others'
]


def get_api_key(api_key_arg=None):
    """Get API key from various sources."""
    api_key = (
        api_key_arg or
        os.getenv('PINECONE_API_KEY') or
        os.getenv('ATS_PINECONE_API_KEY')
    )
    
    if not api_key:
        logger.error("❌ PINECONE_API_KEY is required!")
        sys.exit(1)
    
    return api_key


def list_indexes(pc):
    """List all indexes and show which ones can be deleted."""
    logger.info("=" * 60)
    logger.info("Current Pinecone Indexes")
    logger.info("=" * 60)
    
    indexes = pc.list_indexes()
    index_names = [idx.name for idx in indexes]
    
    logger.info(f"\nYou have {len(indexes)} indexes (Limit: 5):")
    for idx in sorted(index_names):
        logger.info(f"  - {idx}")
    
    return index_names


def setup_namespaces(pc, index_name='ats-resumes'):
    """Setup namespaces in an existing index."""
    logger.info("=" * 60)
    logger.info("Setting Up Namespaces")
    logger.info("=" * 60)
    logger.info(f"Using index: '{index_name}'")
    logger.info(f"Namespaces to setup: {', '.join(PROFILE_NAMESPACES)}")
    logger.info("")
    logger.info("ℹ️  Namespaces are automatically created when you insert vectors with namespace metadata.")
    logger.info("   No explicit creation needed - they exist as logical partitions within the index.")
    logger.info("")
    logger.info("✅ Namespace setup complete!")
    logger.info("")
    logger.info("To use namespaces when inserting vectors:")
    logger.info("   vector_data = {")
    logger.info("       'id': 'resume_123',")
    logger.info("       'values': embedding,")
    logger.info("       'metadata': {...}")
    logger.info("   }")
    logger.info("   index.upsert(vectors=[vector_data], namespace='java')")
    logger.info("")
    logger.info("To query a specific namespace:")
    logger.info("   results = index.query(vector=query_vector, namespace='java', top_k=10)")


def delete_index(pc, index_name):
    """Delete an index (use with caution!)."""
    try:
        logger.warning(f"⚠️  Deleting index: '{index_name}'")
        pc.delete_index(index_name)
        logger.info(f"✅ Successfully deleted '{index_name}'")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to delete '{index_name}': {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Setup Pinecone namespaces')
    parser.add_argument('--api-key', type=str, help='Pinecone API key')
    parser.add_argument('--index', type=str, default='ats-resumes', help='Index name to use for namespaces')
    parser.add_argument('--delete', type=str, help='Delete a specific index (use with caution!)')
    parser.add_argument('--list', action='store_true', help='List all indexes')
    
    args = parser.parse_args()
    
    try:
        api_key = get_api_key(args.api_key)
        pc = Pinecone(api_key=api_key)
        
        if args.delete:
            # Delete specified index
            if delete_index(pc, args.delete):
                logger.info("\n✅ Index deleted. You can now create new indexes.")
            return
        
        if args.list:
            # Just list indexes
            list_indexes(pc)
            return
        
        # Show current indexes
        index_names = list_indexes(pc)
        
        logger.info("\n" + "=" * 60)
        logger.info("Solution: Use Namespaces Instead of Separate Indexes")
        logger.info("=" * 60)
        logger.info("")
        logger.info("You've reached the 5-index limit. Instead of creating separate indexes,")
        logger.info("use NAMESPACES within your existing 'ats-resumes' index.")
        logger.info("")
        logger.info("Benefits:")
        logger.info("  ✅ No index limit - unlimited namespaces")
        logger.info("  ✅ Better organization - logical partitions")
        logger.info("  ✅ Recommended by Pinecone")
        logger.info("  ✅ Same performance as separate indexes")
        logger.info("")
        
        # Check if we can use existing index
        if args.index in index_names:
            logger.info(f"✅ Using existing index: '{args.index}'")
            setup_namespaces(pc, args.index)
        else:
            logger.warning(f"⚠️  Index '{args.index}' not found!")
            logger.info(f"Available indexes: {', '.join(index_names)}")
            logger.info("")
            logger.info("Options:")
            logger.info("  1. Use an existing index: --index <index-name>")
            logger.info("  2. Delete unused indexes to free up space")
            logger.info("     Example: python setup_pinecone_namespaces.py --delete ats-resumes-1536")
        
        # Show recommendation
        logger.info("\n" + "=" * 60)
        logger.info("Recommendation")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Since you have 5 indexes already, you have two options:")
        logger.info("")
        logger.info("Option 1: Use Namespaces (RECOMMENDED)")
        logger.info("  - Use your existing 'ats-resumes' index")
        logger.info("  - Store different profile types in different namespaces")
        logger.info("  - No limit on namespaces!")
        logger.info("")
        logger.info("Option 2: Delete Unused Indexes")
        logger.info("  - You have 'ats-resumes' and 'ats-resumes-1536' (might be duplicates)")
        logger.info("  - Delete one: python setup_pinecone_namespaces.py --delete ats-resumes-1536")
        logger.info("  - Then create the 4 missing indexes")
        logger.info("")
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

