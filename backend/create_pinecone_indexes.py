"""
Single file to create all Pinecone indexes.
This is the ONLY file you need for creating indexes.

Usage:
    # Create all indexes
    python create_pinecone_indexes.py
    
    # Create specific index
    python create_pinecone_indexes.py --index python
    
    # Create multiple indexes
    python create_pinecone_indexes.py --index python --index dotnet
    
    # With API key
    python create_pinecone_indexes.py --api-key YOUR_API_KEY
    
    # List all indexes
    python create_pinecone_indexes.py --list
"""

import os
import sys
import logging
import time
import argparse
from typing import List, Optional
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try to import ATSConfig
try:
    from ats_config import ATSConfig
    HAS_ATS_CONFIG = True
except ImportError:
    HAS_ATS_CONFIG = False


# All default indexes to create
ALL_INDEXES = [
    'java',
    'python',
    'dotnet',
    'business-analyst',
    'project-manager',
    'sql',
    'others'
]


class PineconeIndexManager:
    """Single class to handle all Pinecone index operations."""
    
    def __init__(self, api_key: str = None):
        """Initialize with API key from various sources."""
        self.api_key = (
            api_key or 
            os.getenv('PINECONE_API_KEY') or 
            os.getenv('ATS_PINECONE_API_KEY') or
            (ATSConfig.PINECONE_API_KEY if HAS_ATS_CONFIG and hasattr(ATSConfig, 'PINECONE_API_KEY') else None)
        )
        
        if not self.api_key:
            error_msg = """
❌ PINECONE_API_KEY is required!

To fix:
1. Set environment variable: $env:PINECONE_API_KEY="your-key"
2. Or pass as argument: --api-key your-key
3. Or create .env file with: PINECONE_API_KEY=your-key

Get your API key from: https://app.pinecone.io/
"""
            logger.error(error_msg)
            raise ValueError("PINECONE_API_KEY is required")
        
        # Initialize Pinecone client
        try:
            self.pc = Pinecone(api_key=self.api_key)
            logger.info("✅ Connected to Pinecone")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Pinecone: {e}")
            raise
    
    def normalize_index_name(self, name: str) -> str:
        """Normalize index name for Pinecone requirements."""
        name = name.lower().strip()
        name = name.replace(' ', '-').replace('.', '').replace('/', '-').replace('_', '-')
        name = ''.join(c if c.isalnum() or c == '-' else '' for c in name)
        while '--' in name:
            name = name.replace('--', '-')
        return name.strip('-')
    
    def create_index(self, 
                    index_name: str,
                    dimension: int = 1536,
                    metric: str = 'cosine',
                    cloud: str = 'aws',
                    region: str = 'us-east-1') -> dict:
        """Create a single index."""
        normalized_name = self.normalize_index_name(index_name)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Creating: '{index_name}' -> '{normalized_name}'")
        logger.info(f"{'='*60}")
        
        try:
            # Check if exists
            existing_indexes = self.pc.list_indexes()
            existing_names = [idx.name for idx in existing_indexes]
            
            if normalized_name in existing_names:
                logger.info(f"ℹ️  Index '{normalized_name}' already exists")
                return {'name': normalized_name, 'status': 'exists', 'success': True}
            
            # Create index
            logger.info(f"Creating index '{normalized_name}' (dimension: {dimension})...")
            self.pc.create_index(
                name=normalized_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
            
            time.sleep(2)  # Wait for initialization
            
            index_info = self.pc.describe_index(normalized_name)
            logger.info(f"✅ Successfully created '{normalized_name}'!")
            logger.info(f"   Status: {index_info.status}")
            
            return {'name': normalized_name, 'status': 'created', 'success': True}
            
        except Exception as e:
            error_str = str(e).lower()
            error_msg = str(e)
            
            if "already exists" in error_str or "duplicate" in error_str:
                logger.info(f"ℹ️  Index '{normalized_name}' already exists")
                return {'name': normalized_name, 'status': 'exists', 'success': True}
            elif "403" in error_msg or "forbidden" in error_str or "max serverless indexes" in error_str:
                logger.error(f"❌ Failed: Reached Pinecone index limit (5 indexes)")
                logger.error("")
                logger.error("💡 Solutions:")
                logger.error("   1. Use NAMESPACES instead (RECOMMENDED):")
                logger.error("      - Use existing 'ats-resumes' index with namespaces")
                logger.error("      - Run: python setup_pinecone_namespaces.py --api-key YOUR_KEY")
                logger.error("")
                logger.error("   2. Delete unused indexes:")
                logger.error("      - You have: ats-resumes, ats-resumes-1536 (might be duplicate)")
                logger.error("      - Delete one: python setup_pinecone_namespaces.py --delete ats-resumes-1536")
                logger.error("")
                logger.error("   3. Upgrade Pinecone plan for more indexes")
                logger.error("      - Visit: https://app.pinecone.io/")
                return {'name': normalized_name, 'status': 'failed', 'success': False, 'error': 'Index limit reached'}
            else:
                logger.error(f"❌ Failed: {e}")
                return {'name': normalized_name, 'status': 'failed', 'success': False, 'error': str(e)}
    
    def create_all_indexes(self, indexes: List[str] = None) -> dict:
        """Create multiple indexes."""
        if indexes is None:
            indexes = ALL_INDEXES
        
        logger.info("=" * 60)
        logger.info("Creating Pinecone Indexes")
        logger.info("=" * 60)
        logger.info(f"Total: {len(indexes)} indexes")
        logger.info(f"Indexes: {', '.join(indexes)}")
        
        results = {}
        for index_name in indexes:
            result = self.create_index(index_name)
            results[index_name] = result
        
        self.print_summary(results)
        return results
    
    def print_summary(self, results: dict):
        """Print summary of results."""
        logger.info("\n" + "=" * 60)
        logger.info("Summary")
        logger.info("=" * 60)
        
        created = [name for name, r in results.items() if r['status'] == 'created']
        exists = [name for name, r in results.items() if r['status'] == 'exists']
        failed = [name for name, r in results.items() if r['status'] == 'failed']
        
        if created:
            logger.info(f"\n✅ Created ({len(created)}): {', '.join(created)}")
        if exists:
            logger.info(f"\nℹ️  Already exists ({len(exists)}): {', '.join(exists)}")
        if failed:
            logger.error(f"\n❌ Failed ({len(failed)}): {', '.join(failed)}")
            for name in failed:
                error = results[name].get('error', 'Unknown')
                logger.error(f"   {name}: {error}")
        
        # List all indexes
        logger.info("\n" + "=" * 60)
        logger.info("All indexes in your Pinecone account:")
        logger.info("=" * 60)
        try:
            all_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in all_indexes]
            required = ['java', 'python', 'dotnet', 'business-analyst', 'project-manager', 'sql', 'others']
            
            for idx in sorted(index_names):
                marker = "✅" if idx in required else "  "
                logger.info(f"{marker} {idx}")
            
            missing = [idx for idx in required if idx not in index_names]
            if missing:
                logger.warning(f"\n⚠️  Missing: {', '.join(missing)}")
            else:
                logger.info(f"\n✅ All required indexes exist! (Total: {len(index_names)})")
        except Exception as e:
            logger.warning(f"Could not list indexes: {e}")
    
    def list_all_indexes(self) -> List[str]:
        """List all existing indexes."""
        try:
            indexes = self.pc.list_indexes()
            return [idx.name for idx in indexes]
        except Exception as e:
            logger.error(f"Error: {e}")
            return []


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Create Pinecone indexes - Single file for all operations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create all default indexes
  python create_pinecone_indexes.py
  
  # Create specific index
  python create_pinecone_indexes.py --index python
  
  # Create multiple indexes
  python create_pinecone_indexes.py --index python --index dotnet --index sql
  
  # With API key
  python create_pinecone_indexes.py --api-key YOUR_API_KEY
  
  # List all indexes
  python create_pinecone_indexes.py --list
        """
    )
    
    parser.add_argument('--api-key', type=str, help='Pinecone API key')
    parser.add_argument('--index', action='append', dest='indexes', help='Specific index(es) to create')
    parser.add_argument('--list', action='store_true', help='List all indexes and exit')
    
    args = parser.parse_args()
    
    try:
        manager = PineconeIndexManager(api_key=args.api_key)
        
        if args.list:
            logger.info("=" * 60)
            logger.info("All Pinecone Indexes")
            logger.info("=" * 60)
            indexes = manager.list_all_indexes()
            if indexes:
                for idx in sorted(indexes):
                    logger.info(f"  - {idx}")
                logger.info(f"\nTotal: {len(indexes)} indexes")
            else:
                logger.info("No indexes found")
            return
        
        if args.indexes:
            results = manager.create_all_indexes(indexes=args.indexes)
        else:
            results = manager.create_all_indexes()
        
        failed = sum(1 for r in results.values() if not r['success'])
        sys.exit(0 if failed == 0 else 1)
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

