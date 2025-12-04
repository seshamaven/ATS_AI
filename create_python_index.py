"""Create Python index - Run: python create_python_index.py"""

import logging
import sys
import argparse
from create_dynamic_pinecone_index import DynamicPineconeIndexCreator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', type=str, help='Pinecone API key')
    args = parser.parse_args()
    
    try:
        logger.info("Creating 'python' index...")
        creator = DynamicPineconeIndexCreator(api_key=args.api_key)
        success = creator.create_index('python', dimension=1536)
        logger.info("✅ Success!" if success else "ℹ️  Already exists")
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

