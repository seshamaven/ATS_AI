"""
Flask API for ATS (Application Tracking System).
Provides endpoints for resume processing and candidate ranking.
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
    # Service to generate embeddings using Azure OpenAI or OpenAI
    
    def __init__(self):
        # Initialize embedding service based on configuration
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


# Initialize services
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
            
            # Store in database
            with create_ats_database() as db:
                candidate_id = db.insert_resume(parsed_data, resume_embedding)
                
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
                'embedding_dimensions': len(resume_embedding),
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
                # Skip if no embedding exists
                if not resume.get('embedding'):
                    logger.warning(f"Resume {resume['candidate_id']} has no embedding, skipping")
                    failed_count += 1
                    continue
                
                # Parse embedding from JSON
                import json
                embedding = json.loads(resume['embedding']) if isinstance(resume['embedding'], str) else resume['embedding']
                
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
        
        logger.info(f"Processing resume search query: {user_query}")
        logger.info(f"Applied filters: {filters}")
        
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
                'message': 'No matching resumes found',
                'query': user_query,
                'search_results': [],
                'total_matches': 0,
                'processing_time_ms': int((time.time() - start_time) * 1000),
                'timestamp': datetime.now().isoformat()
            }), 200
        
        # Process search results
        candidates = []
        for match in search_results.matches:
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
    
    Input: JSON with job_description (required) and optional job_id
    Returns: Ranked list of candidates with scores
    
    Supports both structured input and general job descriptions.
    Automatically extracts skills, experience, and requirements from JD text.
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Validate required fields - job_description is required, job_id is optional
        if 'job_description' not in data:
            return jsonify({'error': 'Missing required field: job_description'}), 400
        
        job_description = data['job_description'].strip()
        if not job_description:
            return jsonify({'error': 'job_description cannot be empty'}), 400
        
        # Generate job_id if not provided
        job_id = data.get('job_id', f"JD_{int(time.time())}")
        
        logger.info(f"Processing ranking request for job_id: {job_id}")
        logger.info(f"Job description length: {len(job_description)} characters")
        
        # Import extraction functions
        from resume_parser import extract_skills_from_text, extract_experience_from_text
        
        # Extract all requirements from job description text
        logger.info("Extracting requirements from job description...")
        extracted_skills = extract_skills_from_text(job_description)
        extracted_experience = extract_experience_from_text(job_description)
        
        # Extract additional information using regex patterns
        import re
        
        # Extract education requirements
        education_patterns = [
            r'(?:Bachelor|Master|PhD|B\.?S\.?|M\.?S\.?|Ph\.?D\.?)\s*(?:in|of)?\s*([A-Za-z\s]+)',
            r'(?:degree|diploma)\s*(?:in|of)?\s*([A-Za-z\s]+)',
            r'(?:Graduate|Postgraduate)\s*(?:in|of)?\s*([A-Za-z\s]+)'
        ]
        
        education_required = ''
        for pattern in education_patterns:
            match = re.search(pattern, job_description, re.IGNORECASE)
            if match:
                education_required = match.group(1).strip()
                break
        
        # Extract domain/industry
        domain_patterns = [
            r'(?:in|for)\s+([A-Za-z\s]+(?:engineering|technology|finance|healthcare|education|marketing|sales))',
            r'(?:industry|sector|field)\s*:?\s*([A-Za-z\s]+)',
            r'(?:experience\s+in|background\s+in)\s+([A-Za-z\s]+)'
        ]
        
        domain = ''
        for pattern in domain_patterns:
            match = re.search(pattern, job_description, re.IGNORECASE)
            if match:
                domain = match.group(1).strip()
                break
        
        # Extract years of experience range
        experience_range_patterns = [
            r'(\d+)\s*[-–]\s*(\d+)\s*years?',
            r'(\d+)\s*to\s*(\d+)\s*years?',
            r'(\d+)\s*-\s*(\d+)\s*years?'
        ]
        
        min_experience = extracted_experience
        max_experience = None
        
        for pattern in experience_range_patterns:
            match = re.search(pattern, job_description, re.IGNORECASE)
            if match:
                min_experience = int(match.group(1))
                max_experience = int(match.group(2))
                break
        
        # Separate required vs preferred skills based on context
        required_skills = []
        preferred_skills = []
        
        # Look for "Must Have", "Required", "Essential" sections
        must_have_section = re.search(r'(?:must\s+have|required|essential|mandatory)[\s\S]*?(?=\n\n|\n[A-Z]|$)', job_description, re.IGNORECASE)
        if must_have_section:
            must_have_text = must_have_section.group(0)
            required_skills = extract_skills_from_text(must_have_text)
        
        # Look for "Preferred", "Nice to Have", "Bonus" sections
        preferred_section = re.search(r'(?:preferred|nice\s+to\s+have|bonus|advantage)[\s\S]*?(?=\n\n|\n[A-Z]|$)', job_description, re.IGNORECASE)
        if preferred_section:
            preferred_text = preferred_section.group(0)
            preferred_skills = extract_skills_from_text(preferred_text)
        
        # If no specific sections found, use all extracted skills as required
        if not required_skills and not preferred_skills:
            required_skills = extracted_skills[:15]  # Top 15 skills as required
            preferred_skills = extracted_skills[15:]  # Rest as preferred
        
        # Build job requirements
        job_requirements = {
            'job_id': job_id,
            'job_description': job_description,
            'required_skills': ', '.join(required_skills) if required_skills else '',
            'preferred_skills': ', '.join(preferred_skills) if preferred_skills else '',
            'min_experience': min_experience,
            'max_experience': max_experience,
            'domain': domain,
            'education_required': education_required
        }
        
        # Log extracted information
        logger.info(f"Extracted skills: {len(required_skills)} required, {len(preferred_skills)} preferred")
        logger.info(f"Extracted experience: {min_experience}-{max_experience} years")
        logger.info(f"Extracted domain: {domain}")
        logger.info(f"Extracted education: {education_required}")
        
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
                    'required_skills': required_skills,
                    'preferred_skills': preferred_skills,
                    'min_experience': min_experience,
                    'max_experience': max_experience,
                    'domain': domain,
                    'education_required': education_required,
                    'all_extracted_skills': extracted_skills
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
                'required_skills': required_skills,
                'preferred_skills': preferred_skills,
                'min_experience': min_experience,
                'max_experience': max_experience,
                'domain': domain,
                'education_required': education_required,
                'all_extracted_skills': extracted_skills
            },
            'ranking_criteria': {
                'weights': ATSConfig.RANKING_WEIGHTS,
                'semantic_similarity': 'enabled'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Ranking completed. Top candidate: {ranked_profiles[0]['name']} with score {ranked_profiles[0]['total_score']}")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error in profile ranking: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/candidate/<int:candidate_id>', methods=['GET'])
def get_candidate(candidate_id):
    # Get candidate details by ID
    try:
        with create_ats_database() as db:
            candidate = db.get_resume_by_id(candidate_id)
            
            if not candidate:
                return jsonify({'error': 'Candidate not found'}), 404
            
            # Remove large fields for response
            if 'embedding' in candidate:
                candidate['embedding_dimensions'] = len(candidate['embedding'])
                del candidate['embedding']
            
            if 'resume_text' in candidate and len(candidate['resume_text']) > 500:
                candidate['resume_text_preview'] = candidate['resume_text'][:500] + '...'
                del candidate['resume_text']
            
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

