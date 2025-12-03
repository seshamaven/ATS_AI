"""
Single file to create all Pinecone indexes.
Supports creating individual indexes or all indexes at once.

Usage:
    # Create all indexes
    python create_all_pinecone_indexes.py
    
    # Delete all existing indexes and create new ones
    python create_all_pinecone_indexes.py --delete-all
    
    # Create specific index
    python create_all_pinecone_indexes.py --index python
    
    # Create multiple specific indexes
    python create_all_pinecone_indexes.py --index python --index dotnet
    
    # With API key
    python create_all_pinecone_indexes.py --api-key YOUR_API_KEY
"""

import os
import sys
import logging
import time
from typing import List, Optional
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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


# All indexes to create
ALL_INDEXES = [
    'java',
    '.net',
    'python',
    'Business Analyst',
    'Others'
]


class PineconeIndexCreator:
    """Single class to handle all Pinecone index operations."""
    
    def __init__(self, api_key: str = None):
        """Initialize with API key."""
        # Try multiple sources for API key
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
        # Convert to lowercase
        name = name.lower().strip()
        
        # Special handling for .net -> dotnet
        if name == '.net' or name == 'net':
            return 'dotnet'
        
        # Replace spaces and special characters with hyphens
        name = name.replace(' ', '-').replace('.', '').replace('/', '-')
        name = name.replace('_', '-')
        
        # Remove any remaining invalid characters
        name = ''.join(c if c.isalnum() or c == '-' else '' for c in name)
        
        # Remove multiple consecutive hyphens
        while '--' in name:
            name = name.replace('--', '-')
        
        # Remove leading/trailing hyphens
        name = name.strip('-')
        
        return name
    
    def create_index(self, 
                    index_name: str,
                    dimension: int = 1536,
                    metric: str = 'cosine',
                    cloud: str = 'aws',
                    region: str = 'us-east-1') -> dict:
        """
        Create a single index.
        
        Returns:
            dict with status: 'created', 'exists', or 'failed'
        """
        # Normalize index name
        normalized_name = self.normalize_index_name(index_name)
        
        logger.info(f"Creating index: '{index_name}' -> '{normalized_name}'")
        
        try:
            # Check if index already exists
            existing_indexes = self.pc.list_indexes()
            existing_names = [idx.name for idx in existing_indexes]
            
            if normalized_name in existing_names:
                logger.info(f"ℹ️  Index '{normalized_name}' already exists")
                return {
                    'index_name': normalized_name,
                    'original_name': index_name,
                    'status': 'exists',
                    'success': True
                }
            
            # Create the index
            logger.info(f"Creating new index '{normalized_name}' (dimension: {dimension})...")
            self.pc.create_index(
                name=normalized_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
            
            # Wait a moment for index to initialize
            time.sleep(2)
            
            # Get index info
            index_info = self.pc.describe_index(normalized_name)
            
            logger.info(f"✅ Successfully created '{normalized_name}' index!")
            logger.info(f"   Status: {index_info.status}")
            
            return {
                'index_name': normalized_name,
                'original_name': index_name,
                'status': 'created',
                'success': True,
                'dimension': dimension,
                'metric': metric
            }
            
        except Exception as e:
            error_str = str(e).lower()
            if "already exists" in error_str or "duplicate" in error_str:
                logger.warning(f"Index '{normalized_name}' already exists")
                return {
                    'index_name': normalized_name,
                    'original_name': index_name,
                    'status': 'exists',
                    'success': True
                }
            else:
                logger.error(f"❌ Failed to create '{normalized_name}': {e}")
                return {
                    'index_name': normalized_name,
                    'original_name': index_name,
                    'status': 'failed',
                    'success': False,
                    'error': str(e)
                }
    
    def create_all_indexes(self, indexes: List[str] = None) -> dict:
        """
        Create multiple indexes.
        
        Args:
            indexes: List of index names (defaults to ALL_INDEXES)
        
        Returns:
            dict with results for each index
        """
        if indexes is None:
            indexes = ALL_INDEXES
        
        logger.info("=" * 60)
        logger.info("Creating Multiple Pinecone Indexes")
        logger.info("=" * 60)
        logger.info(f"Total indexes to create: {len(indexes)}")
        logger.info(f"Indexes: {', '.join(indexes)}")
        logger.info("")
        
        results = {}
        
        for index_name in indexes:
            logger.info("-" * 60)
            result = self.create_index(index_name)
            results[index_name] = result
            logger.info("")
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: dict):
        """Print summary of index creation results."""
        logger.info("=" * 60)
        logger.info("Summary")
        logger.info("=" * 60)
        
        created = [name for name, result in results.items() if result['status'] == 'created']
        exists = [name for name, result in results.items() if result['status'] == 'exists']
        failed = [name for name, result in results.items() if result['status'] == 'failed']
        
        if created:
            logger.info(f"\n✅ Created ({len(created)}):")
            for name in created:
                normalized = results[name]['index_name']
                logger.info(f"   - {name} -> {normalized}")
        
        if exists:
            logger.info(f"\nℹ️  Already exists ({len(exists)}):")
            for name in exists:
                normalized = results[name]['index_name']
                logger.info(f"   - {name} -> {normalized}")
        
        if failed:
            logger.info(f"\n❌ Failed ({len(failed)}):")
            for name in failed:
                normalized = results[name]['index_name']
                error = results[name].get('error', 'Unknown error')
                logger.info(f"   - {name} -> {normalized}: {error}")
        
        # List all indexes
        logger.info("\n" + "=" * 60)
        logger.info("All indexes in your Pinecone account:")
        logger.info("=" * 60)
        try:
            all_indexes = self.pc.list_indexes()
            for idx in all_indexes:
                logger.info(f"  - {idx.name}")
        except Exception as e:
            logger.warning(f"Could not list indexes: {e}")
    
    def list_all_indexes(self) -> List[str]:
        """List all existing indexes."""
        try:
            indexes = self.pc.list_indexes()
            return [idx.name for idx in indexes]
        except Exception as e:
            logger.error(f"Error listing indexes: {e}")
            return []
    
    def check_index_exists(self, index_name: str) -> bool:
        """Check if an index exists."""
        normalized = self.normalize_index_name(index_name)
        all_indexes = self.list_all_indexes()
        return normalized in all_indexes
    
    def delete_index(self, index_name: str) -> dict:
        """
        Delete a single index.
        
        Args:
            index_name: Name of the index to delete (will be normalized)
        
        Returns:
            dict with status: 'deleted', 'not_found', or 'failed'
        """
        normalized_name = self.normalize_index_name(index_name)
        
        logger.info(f"Deleting index: '{index_name}' -> '{normalized_name}'")
        
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            existing_names = [idx.name for idx in existing_indexes]
            
            if normalized_name not in existing_names:
                logger.info(f"ℹ️  Index '{normalized_name}' does not exist")
                return {
                    'index_name': normalized_name,
                    'original_name': index_name,
                    'status': 'not_found',
                    'success': True
                }
            
            # Delete the index
            logger.info(f"Deleting index '{normalized_name}'...")
            self.pc.delete_index(normalized_name)
            
            # Wait a moment for deletion to complete
            time.sleep(2)
            
            logger.info(f"✅ Successfully deleted '{normalized_name}' index!")
            
            return {
                'index_name': normalized_name,
                'original_name': index_name,
                'status': 'deleted',
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to delete '{normalized_name}': {e}")
            return {
                'index_name': normalized_name,
                'original_name': index_name,
                'status': 'failed',
                'success': False,
                'error': str(e)
            }
    
    def delete_all_indexes(self) -> dict:
        """
        Delete all existing indexes.
        
        Returns:
            dict with results for each deleted index
        """
        logger.info("=" * 60)
        logger.info("Deleting All Existing Pinecone Indexes")
        logger.info("=" * 60)
        
        try:
            existing_indexes = self.pc.list_indexes()
            existing_names = [idx.name for idx in existing_indexes]
            
            if not existing_names:
                logger.info("No indexes found to delete")
                return {}
            
            logger.info(f"Found {len(existing_names)} indexes to delete: {', '.join(existing_names)}")
            logger.info("")
            
            results = {}
            
            for index_name in existing_names:
                logger.info("-" * 60)
                result = self.delete_index(index_name)
                results[index_name] = result
                logger.info("")
            
            # Print summary
            deleted = [name for name, result in results.items() if result['status'] == 'deleted']
            not_found = [name for name, result in results.items() if result['status'] == 'not_found']
            failed = [name for name, result in results.items() if result['status'] == 'failed']
            
            logger.info("=" * 60)
            logger.info("Deletion Summary")
            logger.info("=" * 60)
            
            if deleted:
                logger.info(f"\n✅ Deleted ({len(deleted)}):")
                for name in deleted:
                    logger.info(f"   - {name}")
            
            if not_found:
                logger.info(f"\nℹ️  Not found ({len(not_found)}):")
                for name in not_found:
                    logger.info(f"   - {name}")
            
            if failed:
                logger.info(f"\n❌ Failed ({len(failed)}):")
                for name in failed:
                    error = results[name].get('error', 'Unknown error')
                    logger.info(f"   - {name}: {error}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error listing/deleting indexes: {e}")
            return {}


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create Pinecone indexes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create all default indexes
  python create_all_pinecone_indexes.py
  
  # Delete all existing indexes and create new ones
  python create_all_pinecone_indexes.py --delete-all
  
  # Create specific index
  python create_all_pinecone_indexes.py --index python
  
  # Create multiple specific indexes
  python create_all_pinecone_indexes.py --index python --index dotnet
  
  # With API key
  python create_all_pinecone_indexes.py --api-key YOUR_API_KEY
        """
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='Pinecone API key (optional if set as environment variable)'
    )
    
    parser.add_argument(
        '--index',
        action='append',
        dest='indexes',
        help='Specific index name(s) to create (can be used multiple times). If not specified, creates all default indexes.'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all existing indexes and exit'
    )
    
    parser.add_argument(
        '--delete-all',
        action='store_true',
        help='Delete all existing indexes before creating new ones'
    )
    
    args = parser.parse_args()
    
    try:
        creator = PineconeIndexCreator(api_key=args.api_key)
        
        # If --list flag, just list indexes and exit
        if args.list:
            logger.info("=" * 60)
            logger.info("All Pinecone Indexes")
            logger.info("=" * 60)
            indexes = creator.list_all_indexes()
            if indexes:
                for idx in indexes:
                    logger.info(f"  - {idx}")
                logger.info(f"\nTotal: {len(indexes)} indexes")
            else:
                logger.info("No indexes found")
            return
        
        # Delete all existing indexes if requested
        if args.delete_all:
            logger.info("")
            logger.info("⚠️  WARNING: This will delete ALL existing indexes!")
            logger.info("")
            deletion_results = creator.delete_all_indexes()
            
            # Wait a bit for deletions to complete
            if deletion_results:
                logger.info("Waiting 5 seconds for deletions to complete...")
                time.sleep(5)
                logger.info("")
        
        # Create indexes
        if args.indexes:
            # Create specific indexes
            results = creator.create_all_indexes(indexes=args.indexes)
        else:
            # Create all default indexes
            results = creator.create_all_indexes()
        
        # Check if all succeeded
        failed_count = sum(1 for r in results.values() if not r['success'])
        if failed_count == 0:
            logger.info("\n✅ All operations completed successfully!")
            sys.exit(0)
        else:
            logger.warning(f"\n⚠️  {failed_count} operation(s) failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

