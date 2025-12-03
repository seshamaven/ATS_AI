"""
Create the main Pinecone index for ATS system.
With namespace implementation, only ONE index is needed.
Namespaces will be created automatically when resumes are indexed.
"""

import os
import sys
import logging
from pinecone import Pinecone, ServerlessSpec

# Try to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_main_index(api_key: str = None, index_name: str = 'ats-resumes', dimension: int = 1536):
    """
    Create the main Pinecone index for the ATS system.
    
    Args:
        api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
        index_name: Index name (defaults to 'ats-resumes')
        dimension: Vector dimension (defaults to 1536)
    """
    # Get API key
    if not api_key:
        api_key = os.getenv('PINECONE_API_KEY') or os.getenv('ATS_PINECONE_API_KEY')
    
    if not api_key:
        logger.error("❌ PINECONE_API_KEY is required!")
        logger.error("   Set it as environment variable or pass as --api-key argument")
        logger.error("   Example: python create_main_index.py --api-key YOUR_API_KEY")
        sys.exit(1)
    
    try:
        logger.info(f"🔌 Connecting to Pinecone...")
        pc = Pinecone(api_key=api_key)
        
        # Check existing indexes
        existing_indexes = pc.list_indexes()
        existing_names = [idx.name for idx in existing_indexes]
        
        logger.info(f"\n📋 Found {len(existing_names)} existing index(es):")
        for name in sorted(existing_names):
            logger.info(f"   - {name}")
        
        # Check if main index already exists
        if index_name in existing_names:
            logger.info(f"\n✅ Index '{index_name}' already exists!")
            
            # Get index details
            index_info = pc.describe_index(index_name)
            logger.info(f"\n📊 Index Details:")
            logger.info(f"   Name: {index_info.name}")
            logger.info(f"   Dimension: {index_info.dimension}")
            logger.info(f"   Metric: {index_info.metric}")
            logger.info(f"   Status: {index_info.status.get('ready', 'Unknown')}")
            
            # Check namespaces
            try:
                index = pc.Index(index_name)
                stats = index.describe_index_stats()
                if hasattr(stats, 'namespaces') and stats.namespaces:
                    logger.info(f"\n📁 Namespaces in index:")
                    for ns_name, ns_stats in stats.namespaces.items():
                        vector_count = ns_stats.get('vector_count', 0)
                        logger.info(f"   - '{ns_name}': {vector_count} vectors")
                else:
                    logger.info(f"\n📁 No namespaces yet (will be created automatically when resumes are indexed)")
            except Exception as e:
                logger.warning(f"Could not fetch namespace stats: {e}")
            
            logger.info(f"\n✨ Ready to use! Namespaces will be created automatically when you index resumes.")
            return True
        
        # Create new index
        logger.info(f"\n🔨 Creating new index '{index_name}'...")
        logger.info(f"   Dimension: {dimension}")
        logger.info(f"   Metric: cosine")
        logger.info(f"   Spec: serverless")
        
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        
        logger.info(f"\n✅ Successfully created index '{index_name}'!")
        logger.info(f"\n📝 Next Steps:")
        logger.info(f"   1. The index is ready to use")
        logger.info(f"   2. Namespaces will be created automatically when you process resumes")
        logger.info(f"   3. Resumes will be organized by profile_type in different namespaces")
        logger.info(f"   4. Example namespaces: 'java', 'python', 'dotnet', 'business-analyst', etc.")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"\n❌ Failed to create index: {error_msg}")
        
        # Check for specific errors
        if "403" in error_msg or "FORBIDDEN" in error_msg:
            if "max serverless indexes" in error_msg.lower():
                logger.error("\n⚠️  You've reached the Pinecone free tier limit of 5 indexes.")
                logger.error("   Solution: Delete unused indexes or use namespaces (already implemented!)")
                logger.error("   With namespaces, you only need ONE index for all profile types.")
            else:
                logger.error("   Check your API key permissions.")
        elif "401" in error_msg or "UNAUTHORIZED" in error_msg:
            logger.error("   Invalid API key. Please check your PINECONE_API_KEY.")
        
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create the main Pinecone index for ATS system')
    parser.add_argument('--api-key', type=str, help='Pinecone API key (or set PINECONE_API_KEY env var)')
    parser.add_argument('--index-name', type=str, default='ats-resumes', help='Index name (default: ats-resumes)')
    parser.add_argument('--dimension', type=int, default=1536, help='Vector dimension (default: 1536)')
    
    args = parser.parse_args()
    
    success = create_main_index(
        api_key=args.api_key,
        index_name=args.index_name,
        dimension=args.dimension
    )
    
    sys.exit(0 if success else 1)

