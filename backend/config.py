"""
Configuration management for the Regulation Library API.
Handles environment variables and application settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any

# Get the directory where this config file is located
config_dir = Path(__file__).parent.absolute()
# Try to load .env from backend directory first, then root directory
env_path_backend = config_dir / '.env'
env_path_root = config_dir.parent / '.env'

# Load environment variables from .env file
# Try backend/.env first, then root/.env
if env_path_backend.exists():
    load_dotenv(env_path_backend)
elif env_path_root.exists():
    load_dotenv(env_path_root)
else:
    # Fallback to default behavior (current directory)
    load_dotenv()


class Config:
    """Application configuration class."""
    
    # Database Configuration - Support both Railway and local formats
    MYSQL_HOST = os.getenv('MYSQLHOST', os.getenv('MYSQL_HOST', 'localhost'))
    MYSQL_USER = os.getenv('MYSQLUSER', os.getenv('MYSQL_USER', 'root'))
    MYSQL_PASSWORD = os.getenv('MYSQLPASSWORD', os.getenv('MYSQL_PASSWORD', 'Reset@123'))
    MYSQL_DATABASE = os.getenv('MYSQLDATABASE', os.getenv('MYSQL_DATABASE', 'reglib'))
    MYSQL_PORT = int(os.getenv('MYSQLPORT', os.getenv('MYSQL_PORT', '3306')))
    
    # Ollama Configuration (Preferred - Local LLM)
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen3:4b')
    USE_OLLAMA = os.getenv('USE_OLLAMA', 'True').lower() == 'true'
    
    # Azure OpenAI Configuration (Fallback)
    AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY', os.getenv('OPENAI_API_KEY'))
    AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT', '')
    AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o')
    AZURE_OPENAI_MODEL = os.getenv('AZURE_OPENAI_MODEL', 'gpt-4o')
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-ada-002')
    
    # OpenAI Configuration (Alternative Fallback)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    OPENAI_EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002')
    
    # Pinecone Configuration
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'reglibpinekey')
    PINECONE_CLOUD = os.getenv('PINECONE_CLOUD', 'aws')
    PINECONE_REGION = os.getenv('PINECONE_REGION', 'us-east-1')
    
    # Flask Configuration
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    EMBED_API_PORT = int(os.getenv('EMBED_API_PORT', '5000'))
    CHAT_API_PORT = int(os.getenv('CHAT_API_PORT', '5001'))
    
    # SQLAlchemy Configuration
    SQLALCHEMY_DATABASE_URI = (
        f"mysql+mysqlclient://{os.getenv('MYSQLUSER', os.getenv('MYSQL_USER', 'root'))}:"
        f"{os.getenv('MYSQLPASSWORD', os.getenv('MYSQL_PASSWORD', 'Reset@123'))}@"
        f"{os.getenv('MYSQLHOST', os.getenv('MYSQL_HOST', 'localhost'))}:"
        f"{os.getenv('MYSQLPORT', os.getenv('MYSQL_PORT', '3306'))}/"
        f"{os.getenv('MYSQLDATABASE', os.getenv('MYSQL_DATABASE', 'reglib'))}"
    )
    
    # Application Configuration
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))
    TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', '5'))
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
    EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '1536'))
    
    @classmethod
    def get_mysql_config(cls) -> Dict[str, Any]:
        """Get MySQL configuration dictionary."""
        return {
            'host': cls.MYSQL_HOST,
            'user': cls.MYSQL_USER,
            'password': cls.MYSQL_PASSWORD,
            'database': cls.MYSQL_DATABASE,
            'port': cls.MYSQL_PORT
        }
    
    @classmethod
    def get_sqlalchemy_uri(cls) -> str:
        """Get SQLAlchemy database URI for Flask."""
        return (
            f"mysql+mysqlclient://{cls.MYSQL_USER}:{cls.MYSQL_PASSWORD}@"
            f"{cls.MYSQL_HOST}:{cls.MYSQL_PORT}/{cls.MYSQL_DATABASE}"
        )
    
    @classmethod
    def get_pinecone_config(cls) -> Dict[str, Any]:
        """Get Pinecone configuration dictionary."""
        return {
            'api_key': cls.PINECONE_API_KEY,
            'index_name': cls.PINECONE_INDEX_NAME,
            'cloud': cls.PINECONE_CLOUD,
            'region': cls.PINECONE_REGION,
            'dimension': cls.EMBEDDING_DIMENSION
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present."""
        required_vars = [
            'MYSQL_HOST',
            'MYSQL_USER',
            'MYSQL_PASSWORD',
            'MYSQL_DATABASE'
        ]
        
        # Check for Ollama, Azure OpenAI, or regular OpenAI
        has_ollama = cls.USE_OLLAMA
        has_azure = cls.AZURE_OPENAI_API_KEY and cls.AZURE_OPENAI_ENDPOINT
        has_openai = cls.OPENAI_API_KEY
        
        if not (has_ollama or has_azure or has_openai):
            print("Missing LLM configuration: Need either USE_OLLAMA=True, AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT, or OPENAI_API_KEY")
            return False
        
        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"Missing required environment variables: {', '.join(missing_vars)}")
            return False
        
        return True
    
    @classmethod
    def print_config(cls, hide_sensitive: bool = True):
        """Print current configuration (optionally hiding sensitive data)."""
        print("=" * 50)
        print("Configuration Summary")
        print("=" * 50)
        
        config_items = [
            ('MySQL Host', cls.MYSQL_HOST),
            ('MySQL User', cls.MYSQL_USER),
            ('MySQL Password', '***' if hide_sensitive else cls.MYSQL_PASSWORD),
            ('MySQL Database', cls.MYSQL_DATABASE),
            ('MySQL Port', cls.MYSQL_PORT),
            ('Use Ollama', cls.USE_OLLAMA),
            ('Ollama Base URL', cls.OLLAMA_BASE_URL if cls.USE_OLLAMA else 'Not configured'),
            ('Ollama Model', cls.OLLAMA_MODEL if cls.USE_OLLAMA else 'Not configured'),
            ('Azure OpenAI Endpoint', cls.AZURE_OPENAI_ENDPOINT or 'Not configured'),
            ('Azure OpenAI API Key', '***' if hide_sensitive and cls.AZURE_OPENAI_API_KEY else 'Not configured'),
            ('OpenAI API Key', '***' if hide_sensitive and cls.OPENAI_API_KEY else 'Not configured'),
            ('Embedding Model', cls.OLLAMA_MODEL if cls.USE_OLLAMA else (cls.AZURE_OPENAI_MODEL or cls.OPENAI_EMBEDDING_MODEL)),
            ('Pinecone API Key', '***' if hide_sensitive else cls.PINECONE_API_KEY),
            ('Pinecone Index', cls.PINECONE_INDEX_NAME),
            ('Embed API Port', cls.EMBED_API_PORT),
            ('Chat API Port', cls.CHAT_API_PORT),
            ('Chunk Size', cls.CHUNK_SIZE),
            ('Chunk Overlap', cls.CHUNK_OVERLAP),
            ('Top K Results', cls.TOP_K_RESULTS),
            ('Similarity Threshold', cls.SIMILARITY_THRESHOLD),
        ]
        
        for key, value in config_items:
            print(f"{key:20}: {value}")
        
        print("=" * 50)


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    MYSQL_DATABASE = 'reglib_test'


# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
