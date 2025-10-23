"""
Flask API for ATS (Application Tracking System).
Provides endpoints for resume processing and candidate ranking.
"""

import os
import logging
import json
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
    """Service to generate embeddings using Azure OpenAI or OpenAI."""
    
    def __init__(self):
        """Initialize embedding service based on configuration."""
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
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ATSConfig.ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        with create_ats_database() as db:
            stats = db.get_statistics()
        
        return jsonify({
            'status': 'healthy',
            'service': 'ATS API',
            'timestamp': datetime.now().isoformat(),
            'database': 'connected',
            'statistics': stats
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
        
        file = request.files['file']
        
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


@app.route('/api/profileRankingByJD', methods=['POST'])
def profile_ranking_by_jd():
    """
    Rank candidate profiles against a Job Description.
    
    Input: JSON with job_id and job_description
    Returns: Ranked list of candidates with scores
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Validate required fields
        if 'job_id' not in data:
            return jsonify({'error': 'Missing required field: job_id'}), 400
        if 'job_description' not in data:
            return jsonify({'error': 'Missing required field: job_description'}), 400
        
        job_id = data['job_id']
        job_description = data['job_description'].strip()
        
        if not job_description:
            return jsonify({'error': 'job_description cannot be empty'}), 400
        
        logger.info(f"Processing ranking request for job_id: {job_id}")
        
        # Extract job requirements from data or JD text
        job_requirements = {
            'job_id': job_id,
            'job_description': job_description,
            'required_skills': data.get('required_skills', ''),
            'preferred_skills': data.get('preferred_skills', ''),
            'min_experience': data.get('min_experience', 0),
            'max_experience': data.get('max_experience'),
            'domain': data.get('domain', ''),
            'education_required': data.get('education_required', '')
        }
        
        # If skills not explicitly provided, extract from JD
        if not job_requirements['required_skills']:
            from resume_parser import extract_skills_from_text, extract_experience_from_text
            
            extracted_skills = extract_skills_from_text(job_description)
            job_requirements['required_skills'] = ', '.join(extracted_skills[:15])  # Top 15 skills
            
            if not job_requirements['min_experience']:
                job_requirements['min_experience'] = extract_experience_from_text(job_description)
        
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
    """Get candidate details by ID."""
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
    """Get ranking history for a specific job."""
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
    """Get ATS system statistics."""
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


@app.errorhandler(413)
def file_too_large(e):
    """Handle file too large error."""
    return jsonify({
        'error': f'File too large. Maximum size is {ATSConfig.MAX_FILE_SIZE_MB}MB'
    }), 413


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {e}")
    return jsonify({
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500


if __name__ == '__main__':
    logger.info(f"Starting ATS API on port {ATSConfig.ATS_API_PORT}")
    app.run(
        host='0.0.0.0',
        port=ATSConfig.ATS_API_PORT,
        debug=ATSConfig.FLASK_DEBUG
    )

