"""
Pinecone Index Data Cleanup Script
Clears all vectors from the Pinecone index for fresh data loading.
"""

import logging
from pinecone import Pinecone
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_pinecone_index():
    """
    Clean up Pinecone index by deleting all vectors.
    """
    try:
        # Validate configuration
        if not Config.validate_config():
            logger.error("Configuration validation failed. Please check your environment variables.")
            return False
        
        # Print configuration (hiding sensitive data)
        Config.print_config(hide_sensitive=True)
        
        # Initialize Pinecone
        logger.info("Initializing Pinecone connection...")
        pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        
        # Check if index exists
        if Config.PINECONE_INDEX_NAME not in pc.list_indexes().names():
            logger.error(f"Index '{Config.PINECONE_INDEX_NAME}' not found!")
            logger.info("Available indexes:")
            for index_name in pc.list_indexes().names():
                logger.info(f"  - {index_name}")
            return False
        
        # Connect to index
        logger.info(f"Connecting to index: {Config.PINECONE_INDEX_NAME}")
        index = pc.Index(Config.PINECONE_INDEX_NAME)
        
        # Get index stats before cleanup
        try:
            stats = index.describe_index_stats()
            total_vectors = stats.total_vector_count
            logger.info(f"Index contains {total_vectors} vectors before cleanup")
        except Exception as e:
            logger.warning(f"Could not get index stats: {e}")
            total_vectors = "unknown"
        
        # Confirm cleanup
        print("\n" + "="*60)
        print("⚠️  WARNING: This will delete ALL vectors from the index!")
        print(f"Index: {Config.PINECONE_INDEX_NAME}")
        print(f"Vectors to delete: {total_vectors}")
        print("="*60)
        
        confirm = input("\nAre you sure you want to proceed? (yes/no): ").lower().strip()
        
        if confirm not in ['yes', 'y']:
            logger.info("Cleanup cancelled by user")
            return False
        
        # Delete all vectors
        logger.info("Starting vector deletion...")
        index.delete(delete_all=True)
        
        # Verify cleanup
        try:
            stats_after = index.describe_index_stats()
            remaining_vectors = stats_after.total_vector_count
            logger.info(f"Index now contains {remaining_vectors} vectors after cleanup")
            
            if remaining_vectors == 0:
                logger.info("✅ Cleanup completed successfully!")
                return True
            else:
                logger.warning(f"⚠️  Cleanup may not have completed fully. {remaining_vectors} vectors remain.")
                return False
                
        except Exception as e:
            logger.warning(f"Could not verify cleanup: {e}")
            logger.info("Cleanup operation completed, but verification failed")
            return True
            
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return False

def list_pinecone_indexes():
    """
    List all available Pinecone indexes.
    """
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        
        # List indexes
        indexes = pc.list_indexes()
        
        print("\n" + "="*50)
        print("Available Pinecone Indexes:")
        print("="*50)
        
        if not indexes.names():
            print("No indexes found.")
        else:
            for i, index_name in enumerate(indexes.names(), 1):
                print(f"{i}. {index_name}")
                
                # Get index details
                try:
                    index = pc.Index(index_name)
                    stats = index.describe_index_stats()
                    print(f"   - Vectors: {stats.total_vector_count}")
                    print(f"   - Dimension: {stats.dimension}")
                except Exception as e:
                    print(f"   - Error getting stats: {e}")
                print()
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error listing indexes: {e}")

def get_index_stats():
    """
    Get detailed statistics for the configured index.
    """
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        
        # Check if index exists
        if Config.PINECONE_INDEX_NAME not in pc.list_indexes().names():
            logger.error(f"Index '{Config.PINECONE_INDEX_NAME}' not found!")
            return False
        
        # Connect to index
        index = pc.Index(Config.PINECONE_INDEX_NAME)
        
        # Get stats
        stats = index.describe_index_stats()
        
        print("\n" + "="*50)
        print(f"Index Statistics: {Config.PINECONE_INDEX_NAME}")
        print("="*50)
        print(f"Total Vectors: {stats.total_vector_count}")
        print(f"Dimension: {stats.dimension}")
        
        if hasattr(stats, 'namespaces') and stats.namespaces:
            print("\nNamespaces:")
            for namespace, ns_stats in stats.namespaces.items():
                print(f"  - {namespace}: {ns_stats.vector_count} vectors")
        
        print("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"Error getting index stats: {e}")
        return False

def main():
    """
    Main function with menu options.
    """
    print("\n" + "="*60)
    print("Pinecone Index Cleanup Tool")
    print("="*60)
    print("1. List all indexes")
    print("2. Get index statistics")
    print("3. Cleanup index (delete all vectors)")
    print("4. Exit")
    print("="*60)
    
    while True:
        try:
            choice = input("\nSelect an option (1-4): ").strip()
            
            if choice == '1':
                list_pinecone_indexes()
            elif choice == '2':
                get_index_stats()
            elif choice == '3':
                success = cleanup_pinecone_index()
                if success:
                    print("\n✅ Index cleanup completed successfully!")
                else:
                    print("\n❌ Index cleanup failed!")
            elif choice == '4':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
