"""
Configuration management for the ATS (Application Tracking System).
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


class ATSConfig:
    """ATS-specific configuration class."""
    
    # Database Configuration - Support both Railway and local formats


    MYSQL_HOST = os.getenv('MYSQLHOST','localhost')
    MYSQL_USER = os.getenv('MYSQLUSER','root')
    MYSQL_PASSWORD = os.getenv('MYSQLPASSWORD','root')
    MYSQL_DATABASE = os.getenv('MYSQLDATABASE','ats_db')
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306)) 

    
    # Ollama Configuration (Local LLM - Disabled by default, use OpenAI instead)
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen3:4b')
    USE_OLLAMA = os.getenv('USE_OLLAMA', 'False').lower() == 'true'
    
    # Azure OpenAI Configuration (Fallback)
    AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY', os.getenv('OPENAI_API_KEY'))
    AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT', '')
    AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o')
    AZURE_OPENAI_MODEL = os.getenv('AZURE_OPENAI_MODEL', 'gpt-4o')
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-ada-002')
    
    # OpenAI Configuration (Preferred - Direct OpenAI API)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')  # Default to gpt-4o for better performance
    OPENAI_EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002')
    
    # Pinecone Configuration (Optional - for vector search at scale)
    PINECONE_API_KEY = os.getenv('ATS_PINECONE_API_KEY', os.getenv('PINECONE_API_KEY'))
    PINECONE_INDEX_NAME = os.getenv('ATS_PINECONE_INDEX_NAME', 'ats-resumes')
    PINECONE_CLOUD = os.getenv('ATS_PINECONE_CLOUD', 'aws')
    PINECONE_REGION = os.getenv('ATS_PINECONE_REGION', 'us-east-1')
    USE_PINECONE = os.getenv('USE_PINECONE', 'False').lower() == 'true'
    
    # Flask Configuration
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    # Railway uses PORT environment variable, fallback to ATS_API_PORT
    ATS_API_PORT = int(os.getenv('PORT', os.getenv('ATS_API_PORT', '5002')))
    
    # SQLAlchemy Configuration
    SQLALCHEMY_DATABASE_URI = (
        f"mysql+mysqlclient://{os.getenv('MYSQLUSER', os.getenv('ATS_MYSQL_USER', 'root'))}:"
        f"{os.getenv('MYSQLPASSWORD', os.getenv('ATS_MYSQL_PASSWORD', 'root'))}@"
        f"{os.getenv('MYSQLHOST', os.getenv('ATS_MYSQL_HOST', 'localhost'))}:"
        f"{os.getenv('MYSQLPORT', os.getenv('ATS_MYSQL_PORT', '3306'))}/"
        f"{os.getenv('MYSQLDATABASE', os.getenv('ATS_MYSQL_DATABASE', 'ats_db'))}"
    )
    
    # File Upload Configuration
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '10'))
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')
    
     # Embedding Configuration
    # NOTE: Pinecone index 'ats-resumes' is configured with 1536 dimensions,
    # so the default EMBEDDING_DIMENSION must also be 1536 to match.
    EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '1536'))
    BATCH_EMBEDDING_SIZE = int(os.getenv('BATCH_EMBEDDING_SIZE', '100'))
    
    # Ranking Algorithm Configuration
    RANKING_WEIGHTS = {
        'skills': float(os.getenv('RANKING_WEIGHT_SKILLS', '0.4')),
        'experience': float(os.getenv('RANKING_WEIGHT_EXPERIENCE', '0.3')),
        'domain': float(os.getenv('RANKING_WEIGHT_DOMAIN', '0.2')),
        'education': float(os.getenv('RANKING_WEIGHT_EDUCATION', '0.1'))
    }
    
    # Score normalization thresholds
    EXPERIENCE_MATCH_THRESHOLD_HIGH = float(os.getenv('EXP_MATCH_HIGH', '0.9'))
    EXPERIENCE_MATCH_THRESHOLD_MEDIUM = float(os.getenv('EXP_MATCH_MEDIUM', '0.7'))
    DOMAIN_MATCH_THRESHOLD_HIGH = float(os.getenv('DOMAIN_MATCH_HIGH', '0.85'))
    DOMAIN_MATCH_THRESHOLD_MEDIUM = float(os.getenv('DOMAIN_MATCH_MEDIUM', '0.65'))
    
    # Resume Parsing Configuration
    NLP_MODEL = os.getenv('NLP_MODEL', 'en_core_web_sm')  # spaCy model
    
    # MySQL Unix Socket (for local connections on Linux)
    MYSQL_UNIX_SOCKET = os.getenv('MYSQL_UNIX_SOCKET', '/var/run/mysqld/mysqld.sock')
    
    @classmethod
    def get_mysql_config(cls) -> Dict[str, Any]:
        """Get MySQL configuration dictionary."""
        config = {
            'host': cls.MYSQL_HOST,
            'user': cls.MYSQL_USER,
            'password': cls.MYSQL_PASSWORD,
            'database': cls.MYSQL_DATABASE,
            'port': cls.MYSQL_PORT
        }
        # Add unix_socket for local Linux connections
        if cls.MYSQL_HOST == 'localhost' and os.path.exists(cls.MYSQL_UNIX_SOCKET):
            config['unix_socket'] = cls.MYSQL_UNIX_SOCKET
        return config
    
    @classmethod
    def get_sqlalchemy_uri(cls) -> str:
        """Get SQLAlchemy database URI for Flask."""
        return (
            f"mysql+mysqlclient://{cls.MYSQL_USER}:{cls.MYSQL_PASSWORD}@"
            f"{cls.MYSQL_HOST}:{cls.MYSQL_PORT}/{cls.MYSQL_DATABASE}"
        )
    
    @classmethod
    def get_azure_openai_config(cls) -> Dict[str, Any]:
        """Get Azure OpenAI configuration dictionary."""
        return {
            'api_key': cls.AZURE_OPENAI_API_KEY,
            'endpoint': cls.AZURE_OPENAI_ENDPOINT,
            'api_version': cls.AZURE_OPENAI_API_VERSION,
            'deployment_name': cls.AZURE_OPENAI_DEPLOYMENT_NAME,
            'model': cls.AZURE_OPENAI_MODEL
        }
    
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
        
        # Validate ranking weights sum to 1.0
        total_weight = sum(cls.RANKING_WEIGHTS.values())
        if abs(total_weight - 1.0) > 0.01:
            print(f"Warning: Ranking weights sum to {total_weight}, expected 1.0")
        
        return True
    
    @classmethod
    def print_config(cls, hide_sensitive: bool = True):
        """Print current configuration (optionally hiding sensitive data)."""
        print("=" * 60)
        print("ATS Configuration Summary")
        print("=" * 60)
        
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
            ('Use Pinecone', cls.USE_PINECONE),
            ('Pinecone Index', cls.PINECONE_INDEX_NAME if cls.USE_PINECONE else 'Not enabled'),
            ('ATS API Port', cls.ATS_API_PORT),
            ('Max File Size (MB)', cls.MAX_FILE_SIZE_MB),
            ('Upload Folder', cls.UPLOAD_FOLDER),
            ('Ranking Weights', cls.RANKING_WEIGHTS),
        ]
        
        for key, value in config_items:
            print(f"{key:30}: {value}")
        
        print("=" * 60)


class DevelopmentATSConfig(ATSConfig):
    """Development configuration."""
    DEBUG = True
    TESTING = False


class ProductionATSConfig(ATSConfig):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    FLASK_DEBUG = False


class TestingATSConfig(ATSConfig):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    MYSQL_DATABASE = 'ats_db_test'


# Configuration mapping
ats_config = {
    'development': DevelopmentATSConfig,
    'production': ProductionATSConfig,
    'testing': TestingATSConfig,
    'default': DevelopmentATSConfig
}

