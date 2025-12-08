"""
Pytest test cases for resume upload endpoints in ats_api.py.

Tests cover:
- /api/processResume (multipart file upload)
- /api/processResumeBase64 (JSON with base64 content)
- Success cases and error cases
- Response format validation
"""
import pytest
import json
import base64
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from io import BytesIO


class TestProcessResume:
    """Test cases for /api/processResume endpoint."""
    
    def test_upload_pdf_success(
        self, client, mock_database, mock_embedding_service, 
        mock_resume_parser, mock_profile_scores, mock_second_highest_profile,
        sample_pdf_content, temp_upload_dir
    ):
        """Test successful PDF resume upload."""
        # Mock file operations
        with patch('ats_api.ATSConfig') as mock_config:
            mock_config.UPLOAD_FOLDER = temp_upload_dir
            mock_config.MAX_FILE_SIZE_MB = 10
            mock_config.ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
            mock_config.USE_PINECONE = False
            mock_config.PINECONE_API_KEY = None
            
            # Create a test PDF file
            test_file = BytesIO(sample_pdf_content)
            test_file.name = 'test_resume.pdf'
            
            # Make the request
            response = client.post(
                '/api/processResume',
                data={'file': (test_file, 'test_resume.pdf')},
                content_type='multipart/form-data'
            )
            
            # Assertions
            assert response.status_code == 200
            data = json.loads(response.data)
            
            # Verify response structure
            assert data['status'] == 'success'
            assert data['message'] == 'Resume processed successfully'
            assert 'candidate_id' in data
            assert data['candidate_id'] == 12345
            assert data['candidate_name'] == 'John Doe'
            assert data['email'] == 'john.doe@example.com'
            assert data['total_experience'] == 5.0
            assert data['primary_skills'] == 'Python, Java, SQL'
            assert data['profile_type'] == 'Python'
            assert data['role_type'] == 'Software Engineer'
            assert data['subrole_type'] == 'Backend'
            assert data['sub_profile_type'] == 'Java'
            assert 'timestamp' in data
            
            # Verify database was called
            mock_database.insert_resume.assert_called_once()
            mock_database.insert_or_update_profile_scores.assert_called_once()
            mock_database.update_resume.assert_called()
            
            # Verify embedding was generated
            mock_embedding_service.generate_embedding.assert_called_once()
    
    def test_upload_docx_success(
        self, client, mock_database, mock_embedding_service,
        mock_resume_parser, mock_profile_scores, mock_second_highest_profile,
        sample_docx_content, temp_upload_dir
    ):
        """Test successful DOCX resume upload."""
        with patch('ats_api.ATSConfig') as mock_config:
            mock_config.UPLOAD_FOLDER = temp_upload_dir
            mock_config.MAX_FILE_SIZE_MB = 10
            mock_config.ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
            mock_config.USE_PINECONE = False
            
            test_file = BytesIO(sample_docx_content)
            test_file.name = 'test_resume.docx'
            
            response = client.post(
                '/api/processResume',
                data={'file': (test_file, 'test_resume.docx')},
                content_type='multipart/form-data'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['status'] == 'success'
            assert data['candidate_id'] == 12345
    
    def test_upload_with_pinecone_indexing(
        self, client, mock_database, mock_embedding_service,
        mock_resume_parser, mock_profile_scores, mock_second_highest_profile,
        mock_pinecone, sample_pdf_content, temp_upload_dir
    ):
        """Test resume upload with Pinecone indexing enabled."""
        with patch('ats_api.ATSConfig') as mock_config:
            mock_config.UPLOAD_FOLDER = temp_upload_dir
            mock_config.MAX_FILE_SIZE_MB = 10
            mock_config.ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
            mock_config.USE_PINECONE = True
            mock_config.PINECONE_API_KEY = 'test-api-key'
            mock_config.EMBEDDING_DIMENSION = 1536
            mock_config.PINECONE_INDEX_NAME = 'test-index'
            
            with patch('ats_api.get_index_name_from_profile_type', return_value='python'):
                test_file = BytesIO(sample_pdf_content)
                test_file.name = 'test_resume.pdf'
                
                response = client.post(
                    '/api/processResume',
                    data={'file': (test_file, 'test_resume.pdf')},
                    content_type='multipart/form-data'
                )
                
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['status'] == 'success'
                assert data['pinecone_indexed'] == True
                assert data.get('pinecone_error') is None
                
                # Verify Pinecone was called
                mock_pinecone.get_or_create_index.assert_called_once()
                mock_pinecone.upsert_vectors.assert_called_once()
    
    def test_upload_no_file(self, client):
        """Test upload without file."""
        response = client.post(
            '/api/processResume',
            data={},
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No file part' in data['error']
    
    def test_upload_empty_filename(self, client):
        """Test upload with empty filename."""
        test_file = BytesIO(b'content')
        test_file.name = ''
        
        response = client.post(
            '/api/processResume',
            data={'file': (test_file, '')},
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No file selected' in data['error']
    
    def test_upload_invalid_file_type(self, client):
        """Test upload with invalid file extension."""
        test_file = BytesIO(b'content')
        test_file.name = 'test.txt'
        
        response = client.post(
            '/api/processResume',
            data={'file': (test_file, 'test.txt')},
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Invalid file type' in data['error']
    
    def test_upload_database_error(
        self, client, mock_database, mock_embedding_service,
        mock_resume_parser, sample_pdf_content, temp_upload_dir
    ):
        """Test upload when database insert fails."""
        with patch('ats_api.ATSConfig') as mock_config:
            mock_config.UPLOAD_FOLDER = temp_upload_dir
            mock_config.MAX_FILE_SIZE_MB = 10
            mock_config.ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
            mock_config.USE_PINECONE = False
            
            # Mock database to return None (insert failure)
            mock_database.insert_resume.return_value = None
            mock_database.last_error = "Database connection failed"
            mock_database.last_error_code = 2002
            
            test_file = BytesIO(sample_pdf_content)
            test_file.name = 'test_resume.pdf'
            
            response = client.post(
                '/api/processResume',
                data={'file': (test_file, 'test_resume.pdf')},
                content_type='multipart/form-data'
            )
            
            assert response.status_code == 500
            data = json.loads(response.data)
            assert 'error' in data
            assert 'Failed to store resume' in data['error']
    
    def test_upload_embedding_generation_error(
        self, client, mock_database, mock_embedding_service,
        mock_resume_parser, sample_pdf_content, temp_upload_dir
    ):
        """Test upload when embedding generation fails."""
        with patch('ats_api.ATSConfig') as mock_config:
            mock_config.UPLOAD_FOLDER = temp_upload_dir
            mock_config.MAX_FILE_SIZE_MB = 10
            mock_config.ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
            mock_config.USE_PINECONE = False
            
            # Mock embedding service to raise exception
            mock_embedding_service.generate_embedding.side_effect = Exception("Embedding API error")
            
            test_file = BytesIO(sample_pdf_content)
            test_file.name = 'test_resume.pdf'
            
            response = client.post(
                '/api/processResume',
                data={'file': (test_file, 'test_resume.pdf')},
                content_type='multipart/form-data'
            )
            
            assert response.status_code == 500
            data = json.loads(response.data)
            assert data['status'] == 'error'


class TestProcessResumeBase64:
    """Test cases for /api/processResumeBase64 endpoint."""
    
    def test_upload_base64_pdf_success(
        self, client, mock_database, mock_embedding_service,
        mock_resume_parser, mock_profile_scores, mock_second_highest_profile,
        sample_pdf_base64, temp_upload_dir
    ):
        """Test successful base64 PDF upload."""
        with patch('ats_api.ATSConfig') as mock_config:
            mock_config.UPLOAD_FOLDER = temp_upload_dir
            mock_config.MAX_FILE_SIZE_MB = 10
            mock_config.ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
            mock_config.USE_PINECONE = False
            
            payload = {
                'filename': 'test_resume.pdf',
                'fileBase64': sample_pdf_base64
            }
            
            response = client.post(
                '/api/processResumeBase64',
                data=json.dumps(payload),
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            # Verify response structure
            assert data['status'] == 'success'
            assert data['message'] == 'Resume processed successfully'
            assert 'candidate_id' in data
            assert data['candidate_id'] == 12345
            assert data['candidate_name'] == 'John Doe'
            assert data['email'] == 'john.doe@example.com'
            assert 'timestamp' in data
            
            # Verify database was called
            mock_database.insert_resume.assert_called_once()
            mock_embedding_service.generate_embedding.assert_called_once()
    
    def test_upload_base64_docx_success(
        self, client, mock_database, mock_embedding_service,
        mock_resume_parser, mock_profile_scores, mock_second_highest_profile,
        temp_upload_dir
    ):
        """Test successful base64 DOCX upload."""
        with patch('ats_api.ATSConfig') as mock_config:
            mock_config.UPLOAD_FOLDER = temp_upload_dir
            mock_config.MAX_FILE_SIZE_MB = 10
            mock_config.ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
            mock_config.USE_PINECONE = False
            
            # Create base64 encoded DOCX content
            docx_content = b'PK\x03\x04\x14\x00\x00\x00\x08\x00'
            docx_base64 = base64.b64encode(docx_content).decode('utf-8')
            
            payload = {
                'filename': 'test_resume.docx',
                'fileBase64': docx_base64
            }
            
            response = client.post(
                '/api/processResumeBase64',
                data=json.dumps(payload),
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['status'] == 'success'
    
    def test_upload_base64_with_pinecone(
        self, client, mock_database, mock_embedding_service,
        mock_resume_parser, mock_profile_scores, mock_second_highest_profile,
        mock_pinecone, sample_pdf_base64, temp_upload_dir
    ):
        """Test base64 upload with Pinecone indexing."""
        with patch('ats_api.ATSConfig') as mock_config:
            mock_config.UPLOAD_FOLDER = temp_upload_dir
            mock_config.MAX_FILE_SIZE_MB = 10
            mock_config.ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
            mock_config.USE_PINECONE = True
            mock_config.PINECONE_API_KEY = 'test-api-key'
            mock_config.EMBEDDING_DIMENSION = 1536
            
            with patch('ats_api.get_index_name_from_profile_type', return_value='python'):
                payload = {
                    'filename': 'test_resume.pdf',
                    'fileBase64': sample_pdf_base64
                }
                
                response = client.post(
                    '/api/processResumeBase64',
                    data=json.dumps(payload),
                    content_type='application/json'
                )
                
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['pinecone_indexed'] == True
                mock_pinecone.upsert_vectors.assert_called_once()
    
    def test_upload_base64_not_json(self, client):
        """Test base64 upload with non-JSON request."""
        response = client.post(
            '/api/processResumeBase64',
            data='not json',
            content_type='text/plain'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Request must be JSON' in data['error']
    
    def test_upload_base64_missing_filename(self, client):
        """Test base64 upload without filename."""
        payload = {
            'fileBase64': base64.b64encode(b'content').decode('utf-8')
        }
        
        response = client.post(
            '/api/processResumeBase64',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'filename and fileBase64 are required' in data['error']
    
    def test_upload_base64_missing_filebase64(self, client):
        """Test base64 upload without fileBase64."""
        payload = {
            'filename': 'test.pdf'
        }
        
        response = client.post(
            '/api/processResumeBase64',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'filename and fileBase64 are required' in data['error']
    
    def test_upload_base64_invalid_base64(self, client):
        """Test base64 upload with invalid base64 data."""
        payload = {
            'filename': 'test.pdf',
            'fileBase64': 'invalid-base64!!!'
        }
        
        response = client.post(
            '/api/processResumeBase64',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Invalid base64' in data['error']
    
    def test_upload_base64_invalid_file_type(self, client):
        """Test base64 upload with invalid file extension."""
        payload = {
            'filename': 'test.txt',
            'fileBase64': base64.b64encode(b'content').decode('utf-8')
        }
        
        response = client.post(
            '/api/processResumeBase64',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Invalid file type' in data['error']
    
    def test_upload_base64_database_error(
        self, client, mock_database, mock_embedding_service,
        mock_resume_parser, sample_pdf_base64, temp_upload_dir
    ):
        """Test base64 upload when database insert fails."""
        with patch('ats_api.ATSConfig') as mock_config:
            mock_config.UPLOAD_FOLDER = temp_upload_dir
            mock_config.MAX_FILE_SIZE_MB = 10
            mock_config.ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
            mock_config.USE_PINECONE = False
            
            mock_database.insert_resume.return_value = None
            
            payload = {
                'filename': 'test_resume.pdf',
                'fileBase64': sample_pdf_base64
            }
            
            response = client.post(
                '/api/processResumeBase64',
                data=json.dumps(payload),
                content_type='application/json'
            )
            
            assert response.status_code == 500
            data = json.loads(response.data)
            assert 'error' in data


class TestResumeUploadIntegration:
    """Integration tests for resume upload flow."""
    
    def test_full_upload_flow_with_all_components(
        self, client, mock_database, mock_embedding_service,
        mock_resume_parser, mock_profile_scores, mock_second_highest_profile,
        mock_pinecone, sample_pdf_content, temp_upload_dir
    ):
        """Test complete upload flow with all components."""
        with patch('ats_api.ATSConfig') as mock_config:
            mock_config.UPLOAD_FOLDER = temp_upload_dir
            mock_config.MAX_FILE_SIZE_MB = 10
            mock_config.ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
            mock_config.USE_PINECONE = True
            mock_config.PINECONE_API_KEY = 'test-api-key'
            mock_config.EMBEDDING_DIMENSION = 1536
            mock_config.PINECONE_INDEX_NAME = 'test-index'
            
            with patch('ats_api.get_index_name_from_profile_type', return_value='python'):
                test_file = BytesIO(sample_pdf_content)
                test_file.name = 'test_resume.pdf'
                
                response = client.post(
                    '/api/processResume',
                    data={'file': (test_file, 'test_resume.pdf')},
                    content_type='multipart/form-data'
                )
                
                # Verify all components were called
                assert response.status_code == 200
                mock_resume_parser.parse_resume.assert_called_once()
                mock_embedding_service.generate_embedding.assert_called_once()
                mock_database.insert_resume.assert_called_once()
                mock_database.insert_or_update_profile_scores.assert_called_once()
                mock_database.update_resume.assert_called()
                mock_pinecone.get_or_create_index.assert_called_once()
                mock_pinecone.upsert_vectors.assert_called_once()
    
    def test_response_format_completeness(
        self, client, mock_database, mock_embedding_service,
        mock_resume_parser, mock_profile_scores, mock_second_highest_profile,
        sample_pdf_content, temp_upload_dir
    ):
        """Test that response contains all expected fields."""
        with patch('ats_api.ATSConfig') as mock_config:
            mock_config.UPLOAD_FOLDER = temp_upload_dir
            mock_config.MAX_FILE_SIZE_MB = 10
            mock_config.ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
            mock_config.USE_PINECONE = False
            
            test_file = BytesIO(sample_pdf_content)
            test_file.name = 'test_resume.pdf'
            
            response = client.post(
                '/api/processResume',
                data={'file': (test_file, 'test_resume.pdf')},
                content_type='multipart/form-data'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            # Verify all expected fields are present
            required_fields = [
                'status', 'message', 'candidate_id', 'candidate_name',
                'email', 'total_experience', 'primary_skills', 'domain',
                'education', 'profile_type', 'role_type', 'subrole_type',
                'sub_profile_type', 'embedding_dimensions', 'pinecone_indexed',
                'timestamp'
            ]
            
            for field in required_fields:
                assert field in data, f"Missing field: {field}"
            
            # Verify field types
            assert isinstance(data['candidate_id'], int)
            assert isinstance(data['candidate_name'], str)
            assert isinstance(data['total_experience'], (int, float))
            assert isinstance(data['pinecone_indexed'], bool)
            assert isinstance(data['timestamp'], str)
    
    def test_profile_scores_calculation(
        self, client, mock_database, mock_embedding_service,
        mock_resume_parser, mock_profile_scores, mock_second_highest_profile,
        sample_pdf_content, temp_upload_dir
    ):
        """Test that profile scores are calculated and stored."""
        with patch('ats_api.ATSConfig') as mock_config:
            mock_config.UPLOAD_FOLDER = temp_upload_dir
            mock_config.MAX_FILE_SIZE_MB = 10
            mock_config.ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
            mock_config.USE_PINECONE = False
            
            test_file = BytesIO(sample_pdf_content)
            test_file.name = 'test_resume.pdf'
            
            response = client.post(
                '/api/processResume',
                data={'file': (test_file, 'test_resume.pdf')},
                content_type='multipart/form-data'
            )
            
            assert response.status_code == 200
            
            # Verify profile scores were calculated
            mock_profile_scores.assert_called_once()
            
            # Verify profile scores were stored
            mock_database.insert_or_update_profile_scores.assert_called_once()
            
            # Verify sub_profile_type was set
            mock_database.update_resume.assert_called()
            update_calls = mock_database.update_resume.call_args_list
            sub_profile_calls = [call for call in update_calls if 'sub_profile_type' in str(call)]
            assert len(sub_profile_calls) > 0, "sub_profile_type should be updated"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

