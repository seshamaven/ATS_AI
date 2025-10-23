"""
Flask API for embedding regulations data into Pinecone vector database.
Handles text chunking, OpenAI embeddings, and Pinecone storage using DataPipeline.
"""

import logging
from typing import List, Dict, Any
from flask import Flask, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from datetime import datetime
from config import Config
from datapipeline import create_data_pipeline
from token_tracker import get_token_tracker, log_embedding_tokens
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Validate configuration
if not Config.validate_config():
    logger.error("Configuration validation failed. Please check your environment variables.")
    exit(1)

# Print configuration (hiding sensitive data)
Config.print_config(hide_sensitive=True)

# Initialize OpenAI embeddings based on configuration
def get_openai_embeddings():
    """Get OpenAI embeddings client (Azure or regular) based on configuration."""
    if Config.AZURE_OPENAI_ENDPOINT:
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            azure_deployment=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
            openai_api_key=Config.AZURE_OPENAI_API_KEY,
            openai_api_version=Config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT
        )
    else:
        return OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)

embeddings = get_openai_embeddings()


class TextProcessor:
    """Handles text processing and chunking."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        chunk_size = chunk_size or Config.CHUNK_SIZE
        chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize token counter for embedding model
        try:
            self.token_encoder = tiktoken.encoding_for_model("text-embedding-ada-002")
        except Exception:
            # Fallback to cl100k_base encoding
            self.token_encoder = tiktoken.get_encoding("cl100k_base")
    
    def chunk_document(self, document: str) -> List[str]:
        """Split document into chunks."""
        return self.text_splitter.split_text(document)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the embedding model's tokenizer."""
        try:
            return len(self.token_encoder.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}, using character-based estimation")
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4


class PineconeManager:
    """Handles Pinecone vector database operations."""
    
    def __init__(self, api_key: str = None, index_name: str = None):
        self.pc = Pinecone(api_key=api_key or Config.PINECONE_API_KEY)
        self.index_name = index_name or Config.PINECONE_INDEX_NAME
        self.index = None
    
    def get_or_create_index(self):
        """Get existing index or create new one."""
        try:
            # Check if index exists
            if self.index_name in self.pc.list_indexes().names():
                self.index = self.pc.Index(self.index_name)
                logger.info(f"Connected to existing Pinecone index: {self.index_name}")
            else:
                # Create new index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=Config.EMBEDDING_DIMENSION,
                    metric='cosine',
                    spec={
                        'serverless': {
                            'cloud': Config.PINECONE_CLOUD,
                            'region': Config.PINECONE_REGION
                        }
                    }
                )
                self.index = self.pc.Index(self.index_name)
                logger.info(f"Created new Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error with Pinecone index: {e}")
            raise
    
    def upsert_vectors(self, vectors: List[Dict[str, Any]]):
        """Insert vectors into Pinecone."""
        logger.info(vectors)
        try:
            self.index.upsert(vectors=vectors)
            logger.info(f"Successfully upserted {len(vectors)} vectors to Pinecone")
        except Exception as e:
            logger.error(f"Error upserting vectors to Pinecone: {e}")
            raise


@app.route('/embed', methods=['POST'])
def embed_data():
    """
    Endpoint to fetch data from MySQL, create embeddings, and store in Pinecone.
    """
    try:
        logger.info("Starting embedding process")
        
        # Initialize components
        data_pipeline = create_data_pipeline()
        text_processor = TextProcessor()
        pinecone_manager = PineconeManager()
        
        # Connect to database
        if not data_pipeline.connect():
            return jsonify({'error': 'Failed to connect to database'}), 500
        
        try:
            # Get processed regulations from data pipeline
            processed_regulations = data_pipeline.get_processed_regulations()
            
            if not processed_regulations:
                return jsonify({'message': 'No regulations found in database'}), 200
            
            # Get or create Pinecone index
            pinecone_manager.get_or_create_index()
            
            # Process each regulation
            all_vectors = []
            processed_count = 0
            total_tokens_used = 0
            total_cost = 0.0
            
            # Initialize token tracker
            token_tracker = get_token_tracker()
            
            for processed_reg in processed_regulations:
                try:
                    document = processed_reg['document']
                    metadata = processed_reg['metadata']
                    row_id = processed_reg['row_id']
                    
                    if not document.strip():
                        logger.warning(f"Skipping empty document for row {row_id}")
                        continue
                    
                    # Chunk document
                    chunks = text_processor.chunk_document(document)
                    logger.info(f"Chunks: {chunks}")
                    # Create embeddings for each chunk
                    for i, chunk in enumerate(chunks):
                        try:
                            # Count tokens for this chunk
                            chunk_tokens = text_processor.count_tokens(chunk)
                            
                            # Generate embedding
                            embedding = embeddings.embed_query(chunk)
                            logger.info(f"Embedding: {embedding}")
                            
                            # Log token usage for embedding
                            log_embedding_tokens(
                                model_name="text-embedding-ada-002",
                                input_tokens=chunk_tokens,
                                text_content=chunk[:100],  # Log first 100 chars for reference
                                operation_type="embedding",
                                metadata={
                                    "row_id": row_id,
                                    "chunk_index": i,
                                    "total_chunks": len(chunks),
                                    "regulation": metadata.get('regulation', '')[:100]
                                }
                            )
                            
                            total_tokens_used += chunk_tokens
                            # Create comprehensive vector metadata
                            vector_id = f"reg_{row_id}_chunk_{i}"
                            vector_metadata = {
                                'row_id': row_id,
                                'chunk_index': i,
                                'total_chunks': len(chunks),
                                'chunk_text': chunk[:500],  # Store first 500 chars for reference
                                # Core regulation fields (matching reglibrary table)
                                'regulation': metadata.get('regulation', ''),
                                'summary': metadata.get('summary', ''),
                                'regulator': metadata.get('regulator', ''),
                                'industry': metadata.get('industry', ''),
                                'sub_industry': metadata.get('sub_industry', ''),
                                'activity_class': metadata.get('activity_class', ''),
                                # Task classification
                                'task_category': metadata.get('task_category', ''),
                                'task_subcategory': metadata.get('task_subcategory', ''),
                                # Regulation details
                                'reg_number': metadata.get('reg_number', ''),
                                'reg_date': metadata.get('reg_date', ''),
                                'reg_category': metadata.get('reg_category', ''),
                                'reg_subject': metadata.get('reg_subject', ''),
                                # Compliance details
                                'due_date': metadata.get('due_date', ''),
                                'frequency': metadata.get('frequency', ''),
                                # Risk and control
                                'risk_category': metadata.get('risk_category', ''),
                                'control_nature': metadata.get('control_nature', ''),
                                'department': metadata.get('department', ''),
                                # Additional fields
                                'sourced_from': metadata.get('sourced_from', ''),
                                'prev_reg': metadata.get('prev_reg', ''),
                                'action_items_description': metadata.get('action_items_description', ''),
                                'action_items_names': metadata.get('action_items_names', ''),
                                'date_created': metadata.get('date_created', ''),
                                'date_modified': metadata.get('date_modified', ''),
                                'effective_date': metadata.get('effective_date', ''),
                                'end_date': metadata.get('end_date', ''),
                                'risk_rating': metadata.get('risk_rating', ''),
                                'active': metadata.get('active', '')
                            }
                            
                            # Remove empty values to optimize storage
                            vector_metadata = {k: v for k, v in vector_metadata.items() if v and str(v).strip()}
                            
                            all_vectors.append({
                                'id': vector_id,
                                'values': embedding,
                                'metadata': vector_metadata
                            })
                            
                        except Exception as e:
                            logger.error(f"Error creating embedding for chunk {i} of row {row_id}: {e}")
                            continue
                    
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing regulation {processed_reg.get('row_id', 'unknown')}: {e}")
                    continue
            
            # Batch upsert to Pinecone
            if all_vectors:
                # Pinecone supports batch upserts up to 100 vectors
                batch_size = 100
                for i in range(0, len(all_vectors), batch_size):
                    batch = all_vectors[i:i + batch_size]
                    pinecone_manager.upsert_vectors(batch)
            
            return jsonify({
                'message': 'Embedding process completed successfully',
                'processed_regulations': processed_count,
                'total_vectors_created': len(all_vectors),
                'token_usage': {
                    'total_tokens': total_tokens_used,
                    'total_cost_usd': round(total_cost, 6),
                    'model_used': 'text-embedding-ada-002',
                    'avg_tokens_per_chunk': round(total_tokens_used / len(all_vectors), 2) if all_vectors else 0
                },
                'timestamp': datetime.now().isoformat()
            }), 200
            
        finally:
            data_pipeline.disconnect()
            
    except Exception as e:
        logger.error(f"Error in embed endpoint: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=Config.EMBED_API_PORT, debug=Config.FLASK_DEBUG)
