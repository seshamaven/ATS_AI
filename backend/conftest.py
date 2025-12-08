"""
Pytest configuration and fixtures for ATS API tests.
"""
import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from flask import Flask
import base64

# Import the Flask app
from ats_api import app as flask_app


@pytest.fixture
def app():
    """Create Flask application instance for testing."""
    flask_app.config['TESTING'] = True
    flask_app.config['WTF_CSRF_ENABLED'] = False
    return flask_app


@pytest.fixture
def client(app):
    """Create Flask test client."""
    return app.test_client()


@pytest.fixture
def temp_upload_dir():
    """Create temporary upload directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_database():
    """Mock database connection and operations."""
    with patch('ats_api.create_ats_database') as mock_db:
        db_instance = MagicMock()
        db_instance.cursor = MagicMock()
        db_instance.insert_resume = MagicMock(return_value=12345)
        db_instance.get_resume_by_id = MagicMock(return_value={
            'candidate_id': 12345,
            'name': 'John Doe',
            'email': 'john.doe@example.com',
            'phone': '123-456-7890',
            'total_experience': 5.0,
            'primary_skills': 'Python, Java, SQL',
            'domain': 'Information Technology',
            'education': 'Bachelor of Science',
            'profile_type': 'Python',
            'role_type': 'Software Engineer',
            'subrole_type': 'Backend',
            'sub_profile_type': 'Java',
            'current_designation': 'Senior Developer',
            'current_location': 'New York',
            'status': 'active'
        })
        db_instance.insert_or_update_profile_scores = MagicMock()
        db_instance.update_resume = MagicMock()
        db_instance.last_error = None
        db_instance.last_error_code = None
        
        # Context manager support
        mock_db.return_value.__enter__ = MagicMock(return_value=db_instance)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        
        yield db_instance


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    with patch('ats_api.embedding_service') as mock_service:
        mock_service.generate_embedding = MagicMock(
            return_value=[0.1] * 1536  # 1536-dimensional embedding
        )
        yield mock_service


@pytest.fixture
def mock_resume_parser():
    """Mock resume parser."""
    with patch('ats_api.resume_parser') as mock_parser:
        mock_parser.parse_resume = MagicMock(return_value={
            'name': 'John Doe',
            'email': 'john.doe@example.com',
            'phone': '123-456-7890',
            'total_experience': 5.0,
            'primary_skills': 'Python, Java, SQL',
            'secondary_skills': 'Docker, Kubernetes',
            'all_skills': 'Python, Java, SQL, Docker, Kubernetes',
            'domain': 'Information Technology',
            'education': 'Bachelor of Science',
            'profile_type': 'Python',
            'role_type': 'Software Engineer',
            'subrole_type': 'Backend',
            'sub_profile_type': 'Java',
            'current_designation': 'Senior Developer',
            'current_location': 'New York',
            'resume_text': 'John Doe\nSenior Developer\n5 years of experience in Python and Java...',
            'resume_summary': 'Experienced software developer with expertise in Python and Java'
        })
        yield mock_parser


@pytest.fixture
def mock_profile_scores():
    """Mock profile type scores."""
    with patch('profile_type_utils.get_all_profile_type_scores') as mock_scores:
        mock_scores.return_value = {
            'Python': 85.5,
            'Java': 72.3,
            '.Net': 15.2,
            'JavaScript': 10.5
        }
        yield mock_scores


@pytest.fixture
def mock_second_highest_profile():
    """Mock second highest profile type."""
    with patch('profile_type_utils.get_second_highest_profile_type') as mock_func:
        mock_func.return_value = 'Java'
        yield mock_func


@pytest.fixture
def mock_pinecone():
    """Mock Pinecone operations."""
    with patch('ats_api.EnhancedPineconeManager') as mock_manager_class:
        mock_manager = MagicMock()
        mock_manager.get_or_create_index = MagicMock()
        mock_manager.upsert_vectors = MagicMock()
        mock_manager_class.return_value = mock_manager
        yield mock_manager


@pytest.fixture
def sample_pdf_content():
    """Sample PDF file content (binary)."""
    # This is a minimal valid PDF structure
    return b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\nxref\n0 1\ntrailer\n<<\n/Root 1 0 R\n>>\n%%EOF'


@pytest.fixture
def sample_docx_content():
    """Sample DOCX file content (binary)."""
    # Minimal DOCX structure (ZIP-based format)
    # For testing, we'll use a simple binary string
    return b'PK\x03\x04\x14\x00\x00\x00\x08\x00'


@pytest.fixture
def sample_pdf_base64(sample_pdf_content):
    """Sample PDF file content as base64."""
    return base64.b64encode(sample_pdf_content).decode('utf-8')


@pytest.fixture
def mock_file_operations(temp_upload_dir):
    """Mock file operations to use temp directory."""
    with patch('ats_api.ATSConfig') as mock_config:
        mock_config.UPLOAD_FOLDER = temp_upload_dir
        mock_config.MAX_FILE_SIZE_MB = 10
        mock_config.ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
        mock_config.USE_PINECONE = False
        mock_config.PINECONE_API_KEY = None
        mock_config.EMBEDDING_DIMENSION = 1536
        yield mock_config

