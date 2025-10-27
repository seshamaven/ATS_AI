"""
Flask API for ATS (Application Tracking System).
Provides endpoints for resume processing and candidate ranking.

STATELESS ARCHITECTURE:
This API is designed to be completely stateless. Each request is processed
independently with no data or state from one request influencing another.

Key Design Principles:
1. Database connections are fresh per request using context managers
2. Ranking engine creates new instances per request  
3. Embedding generation is independent per request
4. Resume parsing is independent per request
5. No global state stores request data
6. Thread-safe components (OpenAI clients, NLP models)

This ensures complete isolation between users' requests - no skill data, resume
interpretation, or ranking results from one user influence another user's query.
"""

import os
import logging
import json
import time
from typing import Dict, List, Any
from datetime import datetime
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import traceback

# OpenAI / Azure OpenAI imports
try:
    from openai import OpenAI, AzureOpenAI
except ImportError:
    pass

from ats_config import ATSConfig
from ats_database import create_ats_database
from resume_parser import ResumeParser
from ranking_engine import create_ranking_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = ATSConfig.MAX_FILE_SIZE_MB * 1024 * 1024

# Validate configuration
if not ATSConfig.validate_config():
    logger.error("Configuration validation failed. Please check your environment variables.")
    exit(1)

# Print configuration
ATSConfig.print_config(hide_sensitive=True)

# Create upload folder if not exists
os.makedirs(ATSConfig.UPLOAD_FOLDER, exist_ok=True)


class EmbeddingService:
    """
    Service to generate embeddings using Azure OpenAI or OpenAI.
    
    STATELESS DESIGN: This class is stateless and thread-safe.
    It holds only immutable configuration (API keys, model names).
    Each generate_embedding() call is independent and doesn't cache results.
    Safe to use as a global singleton across concurrent requests.
    """
    
    def __init__(self):
        # Initialize embedding service based on configuration
        # NOTE: Only immutable configuration stored here (safe for global instance)
        self.use_azure = bool(ATSConfig.AZURE_OPENAI_ENDPOINT)
        
        if self.use_azure:
            logger.info("Using Azure OpenAI for embeddings")
            self.client = AzureOpenAI(
                api_key=ATSConfig.AZURE_OPENAI_API_KEY,
                api_version=ATSConfig.AZURE_OPENAI_API_VERSION,
                azure_endpoint=ATSConfig.AZURE_OPENAI_ENDPOINT
            )
            self.model = ATSConfig.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        else:
            logger.info("Using OpenAI for embeddings")
            self.client = OpenAI(api_key=ATSConfig.OPENAI_API_KEY)
            self.model = ATSConfig.OPENAI_EMBEDDING_MODEL
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for given text.
        
        Args:
            text: Input text
        
        Returns:
            1536-dimension embedding vector
        """
        try:
            # Truncate text if too long (max 8191 tokens for ada-002)
            if len(text) > 30000:
                text = text[:30000]
            
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            
            embedding = response.data[0].embedding
            logger.info(f"Generated embedding with {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise


# Initialize services as global singletons
# SAFETY: These are stateless - they only hold configuration and thread-safe clients.
# No request data is stored in these instances, ensuring complete request isolation.
embedding_service = EmbeddingService()
resume_parser = ResumeParser()


def allowed_file(filename: str) -> bool:
    # Check if file extension is allowed
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ATSConfig.ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def root():
    # Root endpoint for Railway health checks
    return jsonify({
        'status': 'healthy',
        'service': 'ATS API',
        'message': 'Application Tracking System API is running',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }), 200


@app.route('/health', methods=['GET'])
def health_check():
    # Health check endpoint
    try:
        # Test database connection with timeout
        db_status = 'unknown'
        stats = {}
        
        try:
            with create_ats_database() as db:
                stats = db.get_statistics()
                db_status = 'connected'
        except Exception as db_error:
            logger.warning(f"Database connection failed during health check: {db_error}")
            db_status = 'disconnected'
            stats = {'error': str(db_error)}
        
        # Return healthy status even if DB is down (for Railway deployment)
        return jsonify({
            'status': 'healthy',
            'service': 'ATS API',
            'timestamp': datetime.now().isoformat(),
            'database': db_status,
            'statistics': stats,
            'version': '1.0.0'
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503


@app.route('/api/processResume', methods=['POST'])
def process_resume():
    """
    Process and store resume with metadata and embeddings.
    
    Accepts: PDF or DOCX file
    Returns: JSON with candidate_id and status
    
    STATELESS GUARANTEE: This endpoint processes each resume independently.
    No data from one user's resume influences another user's processing.
    Each request uses a fresh database connection and generates new embeddings.
    """
    try:
                # Validate file in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in request'}), 400
        
        file = request.files.get('file')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed types: {", ".join(ATSConfig.ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join(ATSConfig.UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        logger.info(f"Processing resume: {filename}")
        
        try:
            # Extract file type
            file_type = filename.rsplit('.', 1)[1].lower()
            
            # Parse resume
            parsed_data = resume_parser.parse_resume(file_path, file_type)
            
            # Generate embedding from resume text
            logger.info("Generating embedding for resume...")
            resume_embedding = embedding_service.generate_embedding(parsed_data['resume_text'])
            
            # Store in database (without embedding - stored in Pinecone only)
            with create_ats_database() as db:
                candidate_id = db.insert_resume(parsed_data)
                
                if not candidate_id:
                    return jsonify({'error': 'Failed to store resume in database'}), 500
            
            # Index in Pinecone if enabled
            pinecone_indexed = False
            if ATSConfig.USE_PINECONE and ATSConfig.PINECONE_API_KEY:
                try:
                    from enhanced_pinecone_manager import EnhancedPineconeManager
                    pinecone_manager = EnhancedPineconeManager(
                        api_key=ATSConfig.PINECONE_API_KEY,
                        index_name=ATSConfig.PINECONE_INDEX_NAME,
                        dimension=ATSConfig.EMBEDDING_DIMENSION
                    )
                    pinecone_manager.get_or_create_index()
                    
                    # Prepare metadata for Pinecone with NULL value handling
                    pinecone_metadata = {
                        'candidate_id': candidate_id,
                        'name': parsed_data.get('name') or 'Unknown',
                        'email': parsed_data.get('email') or 'No email',
                        'domain': parsed_data.get('domain') or 'Unknown',
                        'primary_skills': parsed_data.get('primary_skills') or 'No skills',
                        'total_experience': parsed_data.get('total_experience', 0),
                        'education': parsed_data.get('education') or 'Unknown',
                        'file_type': file_type or 'Unknown',
                        'source': 'resume_upload',
                        'created_at': datetime.now().isoformat()
                    }
                    
                    # Create vector for Pinecone
                    vector_data = {
                        'id': f'resume_{candidate_id}',
                        'values': resume_embedding,
                        'metadata': pinecone_metadata
                    }
                    
                    # Upsert to Pinecone
                    pinecone_manager.upsert_vectors([vector_data])
                    pinecone_indexed = True
                    logger.info(f"Successfully indexed resume {candidate_id} in Pinecone")
                    
                except Exception as e:
                    logger.error(f"Failed to index resume {candidate_id} in Pinecone: {e}")
                    # Don't fail the entire request if Pinecone indexing fails
            
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Prepare response
            response_data = {
                'status': 'success',
                'message': 'Resume processed successfully',
                'candidate_id': candidate_id,
                'candidate_name': parsed_data.get('name'),
                'email': parsed_data.get('email'),
                'total_experience': parsed_data.get('total_experience'),
                'primary_skills': parsed_data.get('primary_skills'),
                'domain': parsed_data.get('domain'),
                'education': parsed_data.get('education'),
                'embedding_dimensions': 'stored_in_pinecone_only',
                'pinecone_indexed': pinecone_indexed,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Resume processed successfully: candidate_id={candidate_id}")
            return jsonify(response_data), 200
            
        except Exception as e:
            # Clean up file on error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise
            
    except Exception as e:
        logger.error(f"Error processing resume: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/indexExistingResumes', methods=['POST'])
def index_existing_resumes():
    """
    Index all existing resumes from database to Pinecone.
    
    Returns: JSON with indexing results
    """
    try:
        if not ATSConfig.USE_PINECONE or not ATSConfig.PINECONE_API_KEY:
            return jsonify({'error': 'Pinecone indexing is not enabled'}), 400
        
        # Get all resumes from database
        with create_ats_database() as db:
            resumes = db.get_all_resumes()
        
        if not resumes:
            return jsonify({'message': 'No resumes found in database'}), 200
        
        # Initialize Pinecone
        from enhanced_pinecone_manager import EnhancedPineconeManager
        pinecone_manager = EnhancedPineconeManager(
            api_key=ATSConfig.PINECONE_API_KEY,
            index_name=ATSConfig.PINECONE_INDEX_NAME,
            dimension=ATSConfig.EMBEDDING_DIMENSION
        )
        pinecone_manager.get_or_create_index()
        
        indexed_count = 0
        failed_count = 0
        vectors_to_upsert = []
        
        for resume in resumes:
            try:
                # Generate embedding for each resume since we don't store it in DB anymore
                # We need the resume text to generate embedding
                # Since resume_text is not stored in DB, we'll skip indexing
                logger.warning(f"Resume {resume['candidate_id']} cannot be indexed - no resume_text available in database")
                failed_count += 1
                continue
                
                # Prepare metadata for Pinecone with NULL value handling
                pinecone_metadata = {
                    'candidate_id': resume['candidate_id'],
                    'name': resume.get('name') or 'Unknown',
                    'email': resume.get('email') or 'No email',
                    'domain': resume.get('domain') or 'Unknown',
                    'primary_skills': resume.get('primary_skills') or 'No skills',
                    'total_experience': resume.get('total_experience', 0),
                    'education': resume.get('education') or 'Unknown',
                    'file_type': resume.get('file_type') or 'Unknown',
                    'source': 'batch_indexing',
                    'created_at': resume.get('created_at', datetime.now().isoformat())
                }
                
                # Create vector for Pinecone
                vector_data = {
                    'id': f'resume_{resume["candidate_id"]}',
                    'values': embedding,
                    'metadata': pinecone_metadata
                }
                
                vectors_to_upsert.append(vector_data)
                indexed_count += 1
                
                # Batch upsert every 100 vectors
                if len(vectors_to_upsert) >= 100:
                    pinecone_manager.upsert_vectors(vectors_to_upsert)
                    vectors_to_upsert = []
                    
            except Exception as e:
                logger.error(f"Failed to prepare resume {resume.get('candidate_id', 'unknown')} for indexing: {e}")
                failed_count += 1
                continue
        
        # Upsert remaining vectors
        if vectors_to_upsert:
            pinecone_manager.upsert_vectors(vectors_to_upsert)
        
        return jsonify({
            'status': 'success',
            'message': 'Batch indexing completed',
            'total_resumes': len(resumes),
            'indexed_count': indexed_count,
            'failed_count': failed_count,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error in batch indexing: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/searchResumes', methods=['POST'])
def search_resumes():
    """
    Search resumes using Pinecone vector similarity.
    
    Accepts: JSON with query text and optional filters
    Returns: JSON with matching resumes and scores
    
    Parameters:
        - query (required): Search query string
        - top_k (optional): Number of results to return (default: 10)
        - min_similarity_score (optional): Minimum similarity threshold (default: 0.3)
        - filters (optional): Metadata filters
    
    STATELESS GUARANTEE: Each search query is processed independently.
    A fresh embedding is generated for the search query, and results are
    retrieved from the vector database without any state from previous searches.
    
    FILTERING: Results below the similarity threshold are filtered out.
    For meaningless queries (like single characters), the API returns a proper
    "no results" message instead of random data.
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if 'query' not in data:
            return jsonify({'error': 'Missing query field'}), 400
        
        user_query = data['query'].strip()
        if not user_query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Optional parameters
        filters = data.get('filters', {})
        top_k = data.get('top_k', 10)
        min_similarity_score = data.get('min_similarity_score', 0.3)  # Default threshold
        
        # Validate query is meaningful
        # Reject queries that are 1 or 2 characters (too short to be meaningful)
        if len(user_query) <= 2:
            return jsonify({
                'message': 'Query is too short. Please provide a search query with at least 3 characters (e.g., "Python", "finance", "developer")',
                'query': user_query,
                'search_results': [],
                'total_matches': 0,
                'suggestion': 'Try searching for skills, job titles, or domains (e.g., "Python developer", "finance analyst", "data science")',
                'timestamp': datetime.now().isoformat()
            }), 200
        
        logger.info(f"Processing resume search query: {user_query}")
        logger.info(f"Applied filters: {filters}")
        logger.info(f"Minimum similarity score threshold: {min_similarity_score}")
        
        # Start timing for performance monitoring
        start_time = time.time()
        
        # Check if Pinecone is enabled
        if not ATSConfig.USE_PINECONE or not ATSConfig.PINECONE_API_KEY:
            return jsonify({'error': 'Pinecone indexing is not enabled'}), 400
        
        # Initialize Pinecone manager
        from enhanced_pinecone_manager import EnhancedPineconeManager
        pinecone_manager = EnhancedPineconeManager(
            api_key=ATSConfig.PINECONE_API_KEY,
            index_name=ATSConfig.PINECONE_INDEX_NAME,
            dimension=ATSConfig.EMBEDDING_DIMENSION
        )
        pinecone_manager.get_or_create_index()
        
        # Generate embedding for query
        query_embedding = embedding_service.generate_embedding(user_query)
        
        # Perform vector search in Pinecone
        search_results = pinecone_manager.query_vectors(
            query_vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filters if filters else None
        )
        
        if not search_results.matches:
            return jsonify({
                'message': 'No matching resumes found for your query',
                'query': user_query,
                'search_results': [],
                'total_matches': 0,
                'processing_time_ms': int((time.time() - start_time) * 1000),
                'timestamp': datetime.now().isoformat()
            }), 200
        
        # Process search results and filter by similarity score
        candidates = []
        for match in search_results.matches:
            # Filter out low-relevance results based on similarity score
            if match.score < min_similarity_score:
                logger.debug(f"Filtered out candidate {match.metadata.get('candidate_id')} with score {match.score:.3f} (threshold: {min_similarity_score})")
                continue
                
            candidate_info = {
                'candidate_id': match.metadata.get('candidate_id'),
                'name': match.metadata.get('name'),
                'email': match.metadata.get('email'),
                'match_score': match.score,
                'primary_skills': match.metadata.get('primary_skills'),
                'total_experience': match.metadata.get('total_experience'),
                'domain': match.metadata.get('domain'),
                'education': match.metadata.get('education'),
                'file_type': match.metadata.get('file_type'),
                'current_location': match.metadata.get('current_location'),
                'current_company': match.metadata.get('current_company'),
                'resume_summary': match.metadata.get('resume_summary'),
                'pinecone_score': match.score,
                'metadata': match.metadata
            }
            candidates.append(candidate_info)
        
        # If no candidates passed the similarity threshold
        if not candidates:
            return jsonify({
                'message': f'No resumes match your query with sufficient relevance (similarity threshold: {min_similarity_score}). Try refining your search terms.',
                'query': user_query,
                'search_results': [],
                'total_matches': 0,
                'total_before_filtering': len(search_results.matches),
                'similarity_threshold': min_similarity_score,
                'suggestion': 'Try searching for specific skills, job titles, or domains (e.g., "Python developer", "finance analyst", "data science")',
                'processing_time_ms': int((time.time() - start_time) * 1000),
                'timestamp': datetime.now().isoformat()
            }), 200
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        return jsonify({
            'message': 'Resume search completed',
            'query': user_query,
            'search_results': candidates,
            'total_matches': len(candidates),
            'processing_time_ms': processing_time,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error in resume search: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/profileRankingByJD', methods=['POST'])
def profile_ranking_by_jd():
    """
    Rank candidate profiles against a Job Description.
    
    Input: JSON with job_description OR job_id (at least one required)
    - If job_id provided: Uses metadata from MySQL database
    - If job_description provided: Extracts and uses metadata from text
    - If both provided: Uses MySQL metadata if available, otherwise extracts from text
    
    Optional inputs for overriding/extending:
    - job_title, required_skills, preferred_skills, min_experience, max_experience
    - domain, education_required, location, employment_type, salary_range
    
    Returns: Ranked list of candidates with scores
    
    STATELESS GUARANTEE: This endpoint creates a fresh ranking engine instance
    and database connection for each request. Rankings are calculated independently
    using only the current request's job description and candidate data from the database.
    No ranking data from one request influences another.
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Generate job_id if not provided
        job_id = data.get('job_id', f"JD_{int(time.time())}")
        
        logger.info(f"Processing ranking request for job_id: {job_id}")
        
        # Try to get job metadata from MySQL if job_id is provided
        job_metadata = None
        if job_id:
            logger.info(f"Attempting to fetch job metadata for job_id: {job_id}")
            with create_ats_database() as db:
                job_metadata = db.get_job_description(job_id)
                if job_metadata:
                    logger.info(f"Found existing job metadata for {job_id}")
                else:
                    logger.info(f"No existing metadata found for {job_id}, will extract from text")
        
        # Determine job_description - use provided text or from metadata
        job_description = data.get('job_description', '')
        if not job_description and job_metadata:
            job_description = job_metadata.get('job_description', '')
        
        if not job_description:
            return jsonify({'error': 'Either job_description or valid job_id is required'}), 400
        
        job_description = job_description.strip()
        
        logger.info(f"Job description length: {len(job_description)} characters")
        
        # Build job requirements using MySQL metadata or extraction
        # Priority: 1) Request parameters, 2) MySQL metadata, 3) Extraction from text
        
        # Use MySQL metadata as base if available
        required_skills = data.get('required_skills', '')
        preferred_skills = data.get('preferred_skills', '')
        min_experience = data.get('min_experience')
        max_experience = data.get('max_experience')
        domain = data.get('domain', '')
        education_required = data.get('education_required', '')
        
        if job_metadata:
            # Use metadata from MySQL, but allow overrides from request
            if not required_skills:
                required_skills = job_metadata.get('required_skills', '')
            if not preferred_skills:
                preferred_skills = job_metadata.get('preferred_skills', '')
            if min_experience is None:
                min_experience = job_metadata.get('min_experience')
            if max_experience is None:
                max_experience = job_metadata.get('max_experience')
            if not domain:
                domain = job_metadata.get('domain', '')
            if not education_required:
                education_required = job_metadata.get('education_required', '')
            
            logger.info("Using job metadata from MySQL database")
        else:
            # Extract from text if no MySQL metadata
            logger.info("Extracting requirements from job description text...")
            from resume_parser import extract_skills_from_text, extract_experience_from_text
            extracted_skills = extract_skills_from_text(job_description)
            extracted_experience = extract_experience_from_text(job_description)
            
            # Convert extracted skills to string if needed
            if not required_skills:
                required_skills = extracted_skills[:15] if isinstance(extracted_skills, list) else extracted_skills
                if isinstance(required_skills, list):
                    required_skills = ', '.join(required_skills[:15])
            
            if not preferred_skills and isinstance(extracted_skills, list):
                preferred_skills = ', '.join(extracted_skills[15:]) if len(extracted_skills) > 15 else ''
            
            if min_experience is None:
                min_experience = extracted_experience
            
        # Convert skills to lists if they're strings
        if isinstance(required_skills, str):
            required_skills_list = [s.strip() for s in required_skills.split(',') if s.strip()]
        elif isinstance(required_skills, list):
            required_skills_list = required_skills
        else:
            required_skills_list = []
        
        if isinstance(preferred_skills, str):
            preferred_skills_list = [s.strip() for s in preferred_skills.split(',') if s.strip()]
        elif isinstance(preferred_skills, list):
            preferred_skills_list = preferred_skills
        else:
            preferred_skills_list = []
        
        # Build job requirements
        job_requirements = {
            'job_id': job_id,
            'job_description': job_description,
            'required_skills': ', '.join(required_skills_list) if required_skills_list else '',
            'preferred_skills': ', '.join(preferred_skills_list) if preferred_skills_list else '',
            'min_experience': min_experience or 0,
            'max_experience': max_experience,
            'domain': domain,
            'education_required': education_required
        }
        
        # Log job requirements
        logger.info(f"Job requirements - required_skills: {len(required_skills_list)}, preferred_skills: {len(preferred_skills_list)}")
        logger.info(f"Job requirements - min_experience: {min_experience}, max_experience: {max_experience}")
        logger.info(f"Job requirements - domain: {domain}, education: {education_required}")
        
        # Generate JD embedding
        logger.info("Generating embedding for job description...")
        jd_embedding = embedding_service.generate_embedding(job_description)
        
        # Store job description in database (optional)
        with create_ats_database() as db:
            jd_data = {
                'job_id': job_id,
                'job_title': data.get('job_title', 'Not specified'),
                'job_description': job_description,
                'required_skills': job_requirements['required_skills'],
                'preferred_skills': job_requirements['preferred_skills'],
                'min_experience': job_requirements['min_experience'],
                'max_experience': job_requirements['max_experience'],
                'domain': job_requirements['domain'],
                'education_required': job_requirements['education_required']
            }
            
            # Try to insert (will fail if job_id already exists, which is ok)
            try:
                db.insert_job_description(jd_data, jd_embedding)
            except Exception:
                logger.info(f"Job description {job_id} may already exist in database")
        
        # Retrieve all active candidates from database
        logger.info("Retrieving candidate profiles from database...")
        with create_ats_database() as db:
            candidates = db.get_all_resumes(status='active')
        
        if not candidates:
            return jsonify({
                'status': 'success',
                'message': 'No active candidate profiles found in database',
                'job_id': job_id,
                'ranked_profiles': [],
                'total_candidates_evaluated': 0,
                'extracted_job_requirements': {
                    'required_skills': required_skills_list,
                    'preferred_skills': preferred_skills_list,
                    'min_experience': min_experience,
                    'max_experience': max_experience,
                    'domain': domain,
                    'education_required': education_required
                },
                'timestamp': datetime.now().isoformat()
            }), 200
        
        logger.info(f"Found {len(candidates)} candidate profiles")
        
        # Initialize ranking engine
        ranking_engine = create_ranking_engine()
        
        # Rank candidates
        logger.info("Ranking candidates against job requirements...")
        ranked_profiles = ranking_engine.rank_candidates(
            candidates,
            job_requirements,
            jd_embedding,
            top_k=data.get('top_k', 50)  # Return top 50 by default
        )
        
        # Store ranking results in database
        with create_ats_database() as db:
            for profile in ranked_profiles:
                ranking_data = {
                    'job_id': job_id,
                    'candidate_id': profile['candidate_id'],
                    'total_score': profile['total_score'],
                    'match_percent': profile['match_percent'],
                    'skills_score': profile['skills_score'],
                    'experience_score': profile['experience_score'],
                    'domain_score': profile['domain_score'],
                    'education_score': profile['education_score'],
                    'matched_skills': ', '.join(profile['matched_skills']),
                    'missing_skills': ', '.join(profile['missing_skills']),
                    'experience_match': profile['experience_match'],
                    'domain_match': profile['domain_match'],
                    'rank_position': profile['rank'],
                    'ranking_algorithm_version': 'v1.0'
                }
                db.insert_ranking_result(ranking_data)
        
        # Prepare response
        response_data = {
            'status': 'success',
            'message': 'Profile ranking completed successfully',
            'job_id': job_id,
            'ranked_profiles': ranked_profiles,
            'total_candidates_evaluated': len(candidates),
            'top_candidates_returned': len(ranked_profiles),
            'extracted_job_requirements': {
                'required_skills': required_skills_list,
                'preferred_skills': preferred_skills_list,
                'min_experience': min_experience,
                'max_experience': max_experience,
                'domain': domain,
                'education_required': education_required
            },
            'ranking_criteria': {
                'weights': ATSConfig.RANKING_WEIGHTS,
                'semantic_similarity': 'enabled'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        if ranked_profiles:
            logger.info(f"Ranking completed. Top candidate: {ranked_profiles[0]['name']} with score {ranked_profiles[0]['total_score']}")
        else:
            logger.info("Ranking completed. No candidates were ranked.")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error in profile ranking: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/comprehensive-profile-ranking', methods=['POST'])
def comprehensive_profile_ranking():
    """
    Comprehensive Profile Ranking Endpoint
    
    Input: JSON with job requirements and optional profiles directory
    Output: Ranked list of candidates with detailed analysis
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Extract job requirements
        job_requirements = data.get('job_requirements', {})
        profiles_dir = data.get('profiles_directory', os.path.join(os.getcwd(), 'profiles'))
        top_k = data.get('top_k', 10)
        
        # Validate job requirements
        if not job_requirements:
            return jsonify({'error': 'job_requirements is required'}), 400
        
        logger.info(f"Processing comprehensive ranking for directory: {profiles_dir}")
        
        # Read profiles from directory
        start_time = time.time()
        profiles = read_profiles_from_directory(profiles_dir)
        
        if not profiles:
            return jsonify({
                'status': 'success',
                'message': 'No profiles found in directory',
                'profiles_directory': profiles_dir,
                'ranked_profiles': [],
                'total_candidates_evaluated': 0,
                'job_requirements': job_requirements,
                'processing_time_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now().isoformat()
            }), 200
        
        logger.info(f"Found {len(profiles)} profiles")
        
        # Initialize ranking engine
        ranking_engine = create_ranking_engine()
        
        # Convert profiles to the format expected by ranking engine
        candidates = []
        for profile in profiles:
            try:
                # Extract skills and experience from content
                from resume_parser import extract_skills_from_text, extract_experience_from_text
                
                content = profile.get('content', '')
                logger.info(f"Processing profile {profile.get('candidate_id')} with content length: {len(content)}")
                
                extracted_skills = extract_skills_from_text(content)
                extracted_experience = extract_experience_from_text(content)
                
                logger.info(f"Extracted {len(extracted_skills)} skills and {extracted_experience} years experience")
                
                # Safely handle skills list slicing
                primary_skills = extracted_skills[:10] if len(extracted_skills) >= 10 else extracted_skills
                secondary_skills = extracted_skills[10:] if len(extracted_skills) > 10 else []
                
                candidate = {
                    'candidate_id': profile.get('candidate_id'),
                    'name': profile.get('name', 'Unknown'),
                    'email': profile.get('email', ''),
                    'phone': profile.get('phone', ''),
                    'primary_skills': ', '.join(primary_skills),
                    'secondary_skills': ', '.join(secondary_skills),
                    'total_experience': extracted_experience,
                    'domain': profile.get('domain', ''),
                    'education': profile.get('education', ''),
                    'resume_text': content,
                    'status': 'active'
                }
                candidates.append(candidate)
                
            except Exception as e:
                logger.error(f"Error processing profile {profile.get('candidate_id')}: {e}")
                # Create a minimal candidate entry to avoid breaking the ranking
                candidate = {
                    'candidate_id': profile.get('candidate_id'),
                    'name': profile.get('name', 'Unknown'),
                    'email': profile.get('email', ''),
                    'phone': profile.get('phone', ''),
                    'primary_skills': '',
                    'secondary_skills': '',
                    'total_experience': 0,
                    'domain': profile.get('domain', ''),
                    'education': profile.get('education', ''),
                    'resume_text': profile.get('content', ''),
                    'status': 'active'
                }
                candidates.append(candidate)
        
        # Rank candidates using existing ranking engine
        ranked_profiles = ranking_engine.rank_candidates(
            candidates=candidates,
            job_requirements=job_requirements,
            jd_embedding=None,  # Will be generated if needed
            top_k=top_k
        )
        
        # Limit to top_k
        ranked_profiles = ranked_profiles[:top_k]
        
        # Prepare response
        response_data = {
            'status': 'success',
            'message': 'Comprehensive profile ranking completed successfully',
            'profiles_directory': profiles_dir,
            'ranked_profiles': ranked_profiles,
            'total_candidates_evaluated': len(profiles),
            'top_candidates_returned': len(ranked_profiles),
            'job_requirements': job_requirements,
            'ranking_criteria': {
                'weights': ATSConfig.RANKING_WEIGHTS,
                'semantic_similarity': 'enabled',
                'analysis_depth': 'comprehensive'
            },
            'processing_time_ms': (time.time() - start_time) * 1000,
            'timestamp': datetime.now().isoformat()
        }
        
        if ranked_profiles:
            logger.info(f"Ranking completed. Top candidate: {ranked_profiles[0]['name']} with score {ranked_profiles[0]['total_score']}")
        else:
            logger.info("Ranking completed. No candidates were ranked.")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error in comprehensive profile ranking: {e}")
        return jsonify({'error': str(e)}), 500


def read_profiles_from_directory(profiles_dir: str) -> List[Dict]:
    """Read all profile files from directory"""
    profiles = []
    
    if not os.path.exists(profiles_dir):
        logger.warning(f"Profiles directory {profiles_dir} does not exist")
        return profiles
    
    try:
        logger.info(f"Scanning directory: {profiles_dir}")
        files = os.listdir(profiles_dir)
        logger.info(f"Found {len(files)} files in directory: {files}")
        
        for filename in files:
            if filename.lower().endswith(('.pdf', '.docx', '.doc', '.txt')):
                file_path = os.path.join(profiles_dir, filename)
                
                # Extract candidate ID from filename
                candidate_id = os.path.splitext(filename)[0]
                
                try:
                    # Read file content
                    if filename.lower().endswith('.txt'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    else:
                        # For PDF/DOCX files, try to extract text using available libraries
                        content = extract_text_from_file(file_path)
                        if not content:
                            content = f"Profile content for candidate {candidate_id} - PDF file detected but text extraction failed"
                    
                    profile = {
                        'candidate_id': candidate_id,
                        'filename': filename,
                        'content': content,
                        'name': f"Candidate {candidate_id}",
                        'email': f"candidate{candidate_id}@example.com",
                        'phone': '',
                        'domain': '',
                        'education': ''
                    }
                    
                    profiles.append(profile)
                    logger.info(f"Loaded profile: {filename} (content length: {len(content)} chars)")
                    
                except Exception as e:
                    logger.error(f"Error reading file {filename}: {e}")
                    # Still add the profile with placeholder content
                    profile = {
                        'candidate_id': candidate_id,
                        'filename': filename,
                        'content': f"Error reading file {filename}: {str(e)}",
                        'name': f"Candidate {candidate_id}",
                        'email': f"candidate{candidate_id}@example.com",
                        'phone': '',
                        'domain': '',
                        'education': ''
                    }
                    profiles.append(profile)
    
    except Exception as e:
        logger.error(f"Error reading profiles directory: {e}")
    
    logger.info(f"Successfully loaded {len(profiles)} profiles")
    return profiles


def extract_text_from_file(file_path: str) -> str:
    """Extract text from PDF/DOCX files"""
    try:
        if file_path.lower().endswith('.pdf'):
            # Try PyPDF2 first
            try:
                import PyPDF2
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text.strip()
            except ImportError:
                logger.warning("PyPDF2 not available, trying pdfplumber")
                try:
                    import pdfplumber
                    with pdfplumber.open(file_path) as pdf:
                        text = ""
                        for page in pdf.pages:
                            text += page.extract_text() or ""
                        return text.strip()
                except ImportError:
                    logger.warning("pdfplumber not available, trying pymupdf")
                    try:
                        import fitz  # PyMuPDF
                        doc = fitz.open(file_path)
                        text = ""
                        for page in doc:
                            text += page.get_text()
                        doc.close()
                        return text.strip()
                    except ImportError:
                        logger.warning("PyMuPDF not available, using placeholder text")
                        return f"PDF file content for {os.path.basename(file_path)} - text extraction libraries not available"
        
        elif file_path.lower().endswith(('.docx', '.doc')):
            try:
                from docx import Document
                doc = Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text.strip()
            except ImportError:
                logger.warning("python-docx not available")
                return f"DOCX file content for {os.path.basename(file_path)} - text extraction library not available"
        
        return ""
        
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return f"Error extracting text from {os.path.basename(file_path)}: {str(e)}"


@app.route('/api/debug-profiles-directory', methods=['POST'])
def debug_profiles_directory():
    """Debug endpoint to check directory access and file listing"""
    try:
        data = request.get_json() or {}
        profiles_dir = data.get('profiles_directory', 'D:\\profiles')
        
        debug_info = {
            'profiles_directory': profiles_dir,
            'directory_exists': os.path.exists(profiles_dir),
            'is_directory': os.path.isdir(profiles_dir) if os.path.exists(profiles_dir) else False,
            'current_working_directory': os.getcwd(),
            'files_in_directory': [],
            'error': None
        }
        
        if os.path.exists(profiles_dir):
            try:
                files = os.listdir(profiles_dir)
                debug_info['files_in_directory'] = files
                debug_info['total_files'] = len(files)
                debug_info['pdf_files'] = [f for f in files if f.lower().endswith('.pdf')]
                debug_info['txt_files'] = [f for f in files if f.lower().endswith('.txt')]
                debug_info['docx_files'] = [f for f in files if f.lower().endswith('.docx')]
                
                # Test reading one file
                if files:
                    test_file = files[0]
                    test_path = os.path.join(profiles_dir, test_file)
                    debug_info['test_file'] = test_file
                    debug_info['test_file_path'] = test_path
                    debug_info['test_file_exists'] = os.path.exists(test_path)
                    debug_info['test_file_size'] = os.path.getsize(test_path) if os.path.exists(test_path) else 0
                    
            except Exception as e:
                debug_info['error'] = f"Error listing directory: {str(e)}"
        else:
            debug_info['error'] = f"Directory {profiles_dir} does not exist"
        
        return jsonify(debug_info), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload-profile', methods=['POST'])
def upload_profile():
    """Upload a profile file to the server"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Create profiles directory if it doesn't exist
        profiles_dir = os.path.join(os.getcwd(), 'profiles')
        os.makedirs(profiles_dir, exist_ok=True)
        
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(profiles_dir, filename)
        file.save(file_path)
        
        logger.info(f"Profile uploaded: {filename} to {file_path}")
        
        return jsonify({
            'status': 'success',
            'message': 'Profile uploaded successfully',
            'filename': filename,
            'file_path': file_path,
            'profiles_directory': profiles_dir
        }), 200
        
    except Exception as e:
        logger.error(f"Error uploading profile: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/list-uploaded-profiles', methods=['GET'])
def list_uploaded_profiles():
    """List all uploaded profiles"""
    try:
        profiles_dir = os.path.join(os.getcwd(), 'profiles')
        
        if not os.path.exists(profiles_dir):
            return jsonify({
                'status': 'success',
                'profiles_directory': profiles_dir,
                'profiles': [],
                'total_profiles': 0,
                'message': 'Profiles directory does not exist'
            }), 200
        
        files = os.listdir(profiles_dir)
        profile_files = [f for f in files if f.lower().endswith(('.pdf', '.docx', '.doc', '.txt'))]
        
        profiles = []
        for filename in profile_files:
            file_path = os.path.join(profiles_dir, filename)
            file_size = os.path.getsize(file_path)
            profiles.append({
                'filename': filename,
                'candidate_id': os.path.splitext(filename)[0],
                'file_size': file_size,
                'file_path': file_path
            })
        
        return jsonify({
            'status': 'success',
            'profiles_directory': profiles_dir,
            'profiles': profiles,
            'total_profiles': len(profiles)
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing profiles: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/candidate/<int:candidate_id>', methods=['GET'])
def get_candidate(candidate_id):
    # Get candidate details by ID
    try:
        with create_ats_database() as db:
            candidate = db.get_resume_by_id(candidate_id)
            
            if not candidate:
                return jsonify({'error': 'Candidate not found'}), 404
            
            # Note: embedding and resume_text are no longer stored in database
            # They are handled in Pinecone only
            
            return jsonify(candidate), 200
            
    except Exception as e:
        logger.error(f"Error fetching candidate: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/job/<job_id>/rankings', methods=['GET'])
def get_job_rankings(job_id):
    # Get ranking history for a specific job
    try:
        limit = request.args.get('limit', 50, type=int)
        
        with create_ats_database() as db:
            rankings = db.get_rankings_for_job(job_id, limit)
            
            return jsonify({
                'job_id': job_id,
                'total_rankings': len(rankings),
                'rankings': rankings,
                'timestamp': datetime.now().isoformat()
            }), 200
            
    except Exception as e:
        logger.error(f"Error fetching rankings: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    # Get ATS system statistics
    try:
        with create_ats_database() as db:
            stats = db.get_statistics()
        
        return jsonify({
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching statistics: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/database/status', methods=['GET'])
def database_status():
    # Detailed database connection status endpoint
    try:
        db_status = {
            'connected': False,
            'error': None,
            'database_info': {},
            'environment_variables': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Check environment variables
        env_vars = {
            'MYSQLHOST': os.getenv('MYSQLHOST'),
            'MYSQLUSER': os.getenv('MYSQLUSER'),
            'MYSQLPASSWORD': '***' if os.getenv('MYSQLPASSWORD') else None,
            'MYSQLDATABASE': os.getenv('MYSQLDATABASE'),
            'MYSQLPORT': os.getenv('MYSQLPORT')
        }
        db_status['environment_variables'] = env_vars
        
        # Test database connection
        try:
            with create_ats_database() as db:
                # Get database statistics
                stats = db.get_statistics()
                db_status['connected'] = True
                db_status['database_info'] = {
                    'database_name': ATSConfig.MYSQL_DATABASE,
                    'host': ATSConfig.MYSQL_HOST,
                    'port': ATSConfig.MYSQL_PORT,
                    'user': ATSConfig.MYSQL_USER,
                    'statistics': stats
                }
                
        except Exception as db_error:
            db_status['error'] = str(db_error)
            db_status['connected'] = False
        
        return jsonify(db_status), 200
        
    except Exception as e:
        logger.error(f"Database status check failed: {e}")
        return jsonify({
            'connected': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/database/test', methods=['POST'])
def test_database_connection():
    # Test database connection with detailed output
    try:
        test_results = {
            'environment_check': {},
            'connection_test': {},
            'database_operations': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Test 1: Environment Variables
        required_vars = ['MYSQLHOST', 'MYSQLUSER', 'MYSQLPASSWORD', 'MYSQLDATABASE', 'MYSQLPORT']
        env_status = {}
        
        for var in required_vars:
            value = os.getenv(var)
            env_status[var] = {
                'present': bool(value),
                'value': '***' if 'PASSWORD' in var and value else value
            }
        
        test_results['environment_check'] = env_status
        
        # Test 2: Database Connection
        try:
            with create_ats_database() as db:
                connection_info = {
                    'status': 'success',
                    'host': ATSConfig.MYSQL_HOST,
                    'database': ATSConfig.MYSQL_DATABASE,
                    'port': ATSConfig.MYSQL_PORT,
                    'user': ATSConfig.MYSQL_USER
                }
                
                # Test 3: Database Operations
                try:
                    stats = db.get_statistics()
                    operations_info = {
                        'status': 'success',
                        'statistics': stats,
                        'operations_tested': ['connection', 'query_execution', 'data_retrieval']
                    }
                except Exception as op_error:
                    operations_info = {
                        'status': 'partial_success',
                        'error': str(op_error),
                        'note': 'Connection successful but some operations failed'
                    }
                
                test_results['connection_test'] = connection_info
                test_results['database_operations'] = operations_info
                
        except Exception as conn_error:
            test_results['connection_test'] = {
                'status': 'failed',
                'error': str(conn_error)
            }
            test_results['database_operations'] = {
                'status': 'not_tested',
                'reason': 'Connection failed'
            }
        
        return jsonify(test_results), 200
        
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.errorhandler(413)
def file_too_large(e):
    # Handle file too large error
    return jsonify({
        'error': 'File too large. Maximum size is ' + str(ATSConfig.MAX_FILE_SIZE_MB) + 'MB'
    }), 413


@app.errorhandler(500)
def internal_error(e):
    # Handle internal server errors
    logger.error("Internal server error: " + str(e))
    return jsonify({
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500


if __name__ == '__main__':
    logger.info("Starting ATS API on port " + str(ATSConfig.ATS_API_PORT))
    app.run(
        host='0.0.0.0',
        port=ATSConfig.ATS_API_PORT,
        debug=ATSConfig.FLASK_DEBUG
    )

