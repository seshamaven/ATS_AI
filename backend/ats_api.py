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
import re
from typing import Dict, List, Any, Optional
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
from profile_type_utils import (
    detect_profile_types_from_text,
    infer_profile_type_from_requirements,
    canonicalize_profile_type_list,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = ATSConfig.MAX_FILE_SIZE_MB * 1024 * 1024

# In-memory cache for latest profile ranking result (Option 1)
# Stores the most recent output from /api/profileRankingByJD
# Automatically overwritten on each new call
_latest_ranking_cache = {
    'ranked_profiles': [],
    'job_requirements': {},
    'timestamp': None,
    'job_id': None
}

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


# Constants controlling layered search
SQL_LAYER_LIMIT = int(os.getenv('ATS_SQL_LAYER_LIMIT', '10000'))
VECTOR_CHUNK_SIZE = int(os.getenv('ATS_VECTOR_CHUNK_SIZE', '200'))
LLM_REFINEMENT_WINDOW = int(os.getenv('ATS_LLM_REFINEMENT_WINDOW', '60'))


class LLMRefinementService:
    """Optional third-layer refinement using Azure/OpenAI chat completions."""
    
    def __init__(self):
        self.client = None
        self.model = None
        try:
            if ATSConfig.AZURE_OPENAI_ENDPOINT:
                self.client = AzureOpenAI(
                    api_key=ATSConfig.AZURE_OPENAI_API_KEY,
                    api_version=ATSConfig.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=ATSConfig.AZURE_OPENAI_ENDPOINT
                )
                self.model = ATSConfig.AZURE_OPENAI_MODEL
            elif ATSConfig.OPENAI_API_KEY:
                self.client = OpenAI(api_key=ATSConfig.OPENAI_API_KEY)
                self.model = ATSConfig.OPENAI_MODEL
        except Exception as exc:
            logger.warning(f"LLM refinement disabled: {exc}")
            self.client = None
            self.model = None
    
    @property
    def available(self) -> bool:
        return self.client is not None and self.model is not None
    
    def rerank_candidates(self, job_context: str, candidates: List[Dict[str, Any]], top_n: int = 50) -> List[Dict[str, Any]]:
        if not self.available or not candidates:
            return candidates
        
        subset = candidates[:top_n]
        try:
            prompt_candidates = []
            for idx, candidate in enumerate(subset, start=1):
                prompt_candidates.append({
                    'rank': idx,
                    'candidate_id': candidate.get('candidate_id'),
                    'name': candidate.get('name'),
                    'profile_type': candidate.get('profile_type'),
                    'current_designation': candidate.get('current_designation'),
                    'primary_skills': candidate.get('primary_skills'),
                    'total_experience': candidate.get('total_experience'),
                    'domain': candidate.get('domain'),
                    'match_score': candidate.get('match_score')
                })
            
            prompt = (
                "You are an expert technical recruiter. "
                "Given the job requirements and candidate summaries below, "
                "return a JSON array named ranking where each entry contains "
                "candidate_id, rank (1 is best) and a short reason. "
                "Only rerank the provided candidates; do not invent new ids.\n\n"
                f"Job context:\n{job_context[:2000]}\n\n"
                f"Candidates:\n{json.dumps(prompt_candidates, ensure_ascii=False)}"
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You rank technical candidates objectively."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            parsed = json.loads(content)
            ranking = parsed.get('ranking', [])
            rank_order = {entry['candidate_id']: entry for entry in ranking if 'candidate_id' in entry}
            
            if not rank_order:
                return candidates
            
            def sort_key(candidate):
                data = rank_order.get(candidate.get('candidate_id'))
                if data:
                    return data.get('rank', 1000)
                return 1000 + candidates.index(candidate)
            
            ordered = sorted(candidates, key=sort_key)
            
            # Attach LLM reasoning if available
            for candidate in ordered:
                llm_meta = rank_order.get(candidate.get('candidate_id'))
                if llm_meta:
                    candidate.setdefault('layer_scores', {})['llm_rank'] = llm_meta.get('rank')
                    candidate['llm_reason'] = llm_meta.get('reason')
            
            return ordered
        except Exception as exc:
            logger.warning(f"LLM refinement failed, falling back to vector order: {exc}")
            return candidates


# Initialize services as global singletons
# SAFETY: These are stateless - they only hold configuration and thread-safe clients.
# No request data is stored in these instances, ensuring complete request isolation.
embedding_service = EmbeddingService()
resume_parser = ResumeParser()
llm_refinement_service = LLMRefinementService()


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Yield evenly sized chunks from a list."""
    return [items[i:i + chunk_size] for i in range(0, len(items), max(chunk_size, 1))]


def normalize_filter_value(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if v not in (None, '', [])]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return [value]


def build_structured_filters(raw_filters: Dict[str, Any], query_text: str) -> Dict[str, Any]:
    """Merge user-provided filters with inferred profile type from the query or skill filters."""
    structured: Dict[str, Any] = {}
    raw_filters = raw_filters or {}
    
    passthrough_keys = [
        'education', 'domain', 'domains', 'current_location', 'locations',
        'current_designation', 'job_title', 'primary_skills', 'skills',
        'min_experience', 'max_experience'
    ]
    for key in passthrough_keys:
        if key in raw_filters and raw_filters[key]:
            structured[key] = raw_filters[key]
    
    # Honor explicitly provided profile type filters and normalize them
    provided_profile_filters: List[str] = []
    for key in ('profile_type', 'profile_types'):
        if key in raw_filters and raw_filters[key]:
            provided_profile_filters.extend(normalize_filter_value(raw_filters[key]))
    canonical_profiles = canonicalize_profile_type_list(provided_profile_filters)
    if canonical_profiles:
        structured['profile_type'] = canonical_profiles
    
    # Auto-infer profile type if not provided
    if 'profile_type' not in structured:
        hint_sources: List[str] = []
        if query_text:
            hint_sources.append(query_text)
        
        hint_keys = [
            'primary_skills', 'skills', 'job_title', 'current_designation',
            'domain', 'domains'
        ]
        for key in hint_keys:
            if not raw_filters.get(key):
                continue
            values = normalize_filter_value(raw_filters[key])
            hint_sources.extend(values)
        
        inferred = detect_profile_types_from_text(*hint_sources)
        if inferred:
            structured['profile_type'] = canonicalize_profile_type_list(inferred[:1]) or inferred[:1]
    
    return structured


def build_pinecone_metadata_filter(structured_filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Build a Pinecone metadata filter using profile_type and designation hints so that
    vector searches respect the same structured constraints as the SQL layer.
    """
    if not structured_filters:
        return None
    
    clauses: List[Dict[str, Any]] = []
    
    profile_types = structured_filters.get('profile_type')
    if profile_types:
        clauses.append({'profile_type': {'$in': profile_types}})
    
    designation_terms: List[str] = []
    for key in ('current_designation', 'job_title'):
        if structured_filters.get(key):
            designation_terms.extend(normalize_filter_value(structured_filters[key]))
    designation_terms = [term for term in designation_terms if term]
    if designation_terms:
        clauses.append({'current_designation': {'$in': designation_terms}})
    
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {'$and': clauses}


def merge_pinecone_filters(*filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Combine multiple Pinecone filter fragments with logical AND."""
    clauses = [f for f in filters if f]
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {'$and': clauses}


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
    
    Accepts: PDF, DOCX, or DOC file
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
            
            # Convert file to base64 for sample storage
            import base64
            with open(file_path, 'rb') as f:
                file_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            # Parse resume
            parsed_data = resume_parser.parse_resume(file_path, file_type)
            
            # Add base64 to parsed data for storage
            parsed_data['file_base64'] = file_base64
            
            # Generate embedding from resume text
            logger.info("Generating embedding for resume...")
            resume_embedding = embedding_service.generate_embedding(parsed_data['resume_text'])
            
            # Store in database (without embedding - stored in Pinecone only)
            with create_ats_database() as db:
                candidate_id = db.insert_resume(parsed_data)
                
                if not candidate_id:
                    return jsonify({'error': 'Failed to store resume in database'}), 500
                
                # Calculate and store profile scores
                try:
                    from profile_type_utils import get_all_profile_type_scores
                    profile_scores = get_all_profile_type_scores(
                        primary_skills=parsed_data.get('primary_skills', ''),
                        secondary_skills=parsed_data.get('secondary_skills', ''),
                        resume_text=parsed_data.get('resume_text', '')
                    )
                    db.insert_or_update_profile_scores(candidate_id, profile_scores)
                    logger.info(f"Stored profile scores for candidate_id={candidate_id}")
                except Exception as e:
                    logger.error(f"Failed to store profile scores for candidate_id={candidate_id}: {e}")
            
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
                        'profile_type': parsed_data.get('profile_type') or 'Generalist',
                        'current_location': parsed_data.get('current_location') or 'Unknown',
                        'current_designation': parsed_data.get('current_designation') or 'Unknown',
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
                'phone': parsed_data.get('phone'),
                'total_experience': parsed_data.get('total_experience'),
                'primary_skills': parsed_data.get('primary_skills'),
                'domain': parsed_data.get('domain'),
                'education': parsed_data.get('education'),
                'profile_type': parsed_data.get('profile_type'),
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


@app.route('/api/processResumeBase64', methods=['POST'])
def process_resume_base64():
    """
    Process and store resume when provided as JSON with base64 content.
    
    Accepts: application/json body with fields:
      - filename: original filename with extension (e.g., resume.pdf)
      - fileBase64: base64-encoded file content
      - text: optional additional context (ignored by parser)
    Returns: JSON with candidate_id and status
    """
    try:
        # Validate JSON request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        filename = (data.get('filename') or '').strip()
        file_b64 = (data.get('fileBase64') or '').strip()

        if not filename or not file_b64:
            return jsonify({'error': 'filename and fileBase64 are required'}), 400

        if not allowed_file(filename):
            return jsonify({
                'error': f'Invalid file type. Allowed types: {", ".join(ATSConfig.ALLOWED_EXTENSIONS)}'
            }), 400

        # Decode base64 and write to temporary file
        from werkzeug.utils import secure_filename
        import base64, binascii

        safe_name = secure_filename(filename)
        file_path = os.path.join(ATSConfig.UPLOAD_FOLDER, safe_name)

        try:
            file_bytes = base64.b64decode(file_b64, validate=True)
        except (binascii.Error, ValueError):
            return jsonify({'error': 'Invalid base64 data'}), 400

        with open(file_path, 'wb') as f:
            f.write(file_bytes)

        logger.info(f"Processing base64 resume: {safe_name}")

        try:
            # Extract file type
            file_type = safe_name.rsplit('.', 1)[1].lower()

            # Parse resume
            parsed_data = resume_parser.parse_resume(file_path, file_type)
            
            # Add base64 to parsed data for storage (use original from request)
            parsed_data['file_base64'] = file_b64

            # Generate embedding from resume text
            logger.info("Generating embedding for resume...")
            resume_embedding = embedding_service.generate_embedding(parsed_data['resume_text'])

            # Store in database (with base64 - stored in database)
            with create_ats_database() as db:
                candidate_id = db.insert_resume(parsed_data)

                if not candidate_id:
                    return jsonify({'error': 'Failed to store resume in database'}), 500

                # Calculate and store profile scores
                try:
                    from profile_type_utils import get_all_profile_type_scores
                    profile_scores = get_all_profile_type_scores(
                        primary_skills=parsed_data.get('primary_skills', ''),
                        secondary_skills=parsed_data.get('secondary_skills', ''),
                        resume_text=parsed_data.get('resume_text', '')
                    )
                    db.insert_or_update_profile_scores(candidate_id, profile_scores)
                    logger.info(f"Stored profile scores for candidate_id={candidate_id}")
                except Exception as e:
                    logger.error(f"Failed to store profile scores for candidate_id={candidate_id}: {e}")

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
                        'profile_type': parsed_data.get('profile_type') or 'Generalist',
                        'current_location': parsed_data.get('current_location') or 'Unknown',
                        'current_designation': parsed_data.get('current_designation') or 'Unknown',
                        'file_type': file_type or 'Unknown',
                        'source': 'resume_upload',
                        'created_at': datetime.now().isoformat()
                    }

                    vector_data = {
                        'id': f'resume_{candidate_id}',
                        'values': resume_embedding,
                        'metadata': pinecone_metadata
                    }

                    pinecone_manager.upsert_vectors([vector_data])
                    pinecone_indexed = True
                    logger.info(f"Successfully indexed resume {candidate_id} in Pinecone")
                except Exception as e:
                    logger.error(f"Failed to index resume {candidate_id} in Pinecone: {e}")
                    # Do not fail request for Pinecone errors

            # Clean up temporary file
            if os.path.exists(file_path):
                os.remove(file_path)

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
                'profile_type': parsed_data.get('profile_type'),
                'embedding_dimensions': 'stored_in_pinecone_only',
                'pinecone_indexed': pinecone_indexed,
                'timestamp': datetime.now().isoformat()
            }

            return jsonify(response_data), 200

        except Exception:
            # Ensure temp file cleanup on error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise

    except Exception as e:
        logger.error(f"Error processing base64 resume: {e}")
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
            temp_file_path = None
            try:
                # Check if file_base64 is available
                file_base64 = resume.get('file_base64')
                if not file_base64:
                    logger.warning(f"Resume {resume['candidate_id']} cannot be indexed - no file_base64 available in database")
                    failed_count += 1
                    continue
                
                # Decode base64 file and create temporary file
                import base64
                import binascii
                import tempfile
                
                try:
                    file_bytes = base64.b64decode(file_base64, validate=True)
                except (binascii.Error, ValueError) as e:
                    logger.error(f"Invalid base64 data for resume {resume['candidate_id']}: {e}")
                    failed_count += 1
                    continue
                
                # Create temporary file from decoded bytes
                file_name = resume.get('file_name', 'resume.pdf')
                file_type = resume.get('file_type', 'pdf').lower()
                
                # Create temp file with proper extension
                suffix = f'.{file_type}' if file_type else '.pdf'
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=ATSConfig.UPLOAD_FOLDER) as tmp_file:
                    tmp_file.write(file_bytes)
                    temp_file_path = tmp_file.name
                
                # Parse resume to extract text
                resume_text = resume_parser.extract_text_from_file(temp_file_path, file_type)
                
                if not resume_text or len(resume_text) < 100:
                    logger.warning(f"Resume {resume['candidate_id']} text too short or empty after parsing")
                    failed_count += 1
                    continue
                
                # Generate embedding for the resume text
                logger.info(f"Generating embedding for resume {resume['candidate_id']}...")
                embedding = embedding_service.generate_embedding(resume_text)
                
                # Prepare metadata for Pinecone with NULL value handling
                pinecone_metadata = {
                    'candidate_id': resume['candidate_id'],
                    'name': resume.get('name') or 'Unknown',
                    'email': resume.get('email') or 'No email',
                    'domain': resume.get('domain') or 'Unknown',
                    'primary_skills': resume.get('primary_skills') or 'No skills',
                    'total_experience': resume.get('total_experience', 0),
                    'education': resume.get('education') or 'Unknown',
                    'profile_type': resume.get('profile_type') or 'Generalist',
                    'current_location': resume.get('current_location') or 'Unknown',
                    'current_designation': resume.get('current_designation') or 'Unknown',
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
                logger.error(traceback.format_exc())
                failed_count += 1
            finally:
                # Clean up temporary file
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")
        
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

def parse_comma_separated_query(query: str) -> Dict:
    """
    Parse comma-separated query into AND conditions.
    
    Handles queries like: "bhargavi, .net, mvc" or "python, java, django"
    Treats comma-separated values as AND conditions.
    Preserves quoted phrases that may contain commas.
    
    Args:
        query: Query string with comma-separated values
        
    Returns:
        Dict with 'and_terms' structure where each term is a separate AND condition
    """
    query = query.strip()
    if not query:
        return {'and_terms': []}
    
    # Extract quoted phrases first (preserve them, including commas inside quotes)
    quoted_phrases = {}
    quote_counter = 0
    
    def replace_quote(match):
        nonlocal quote_counter
        quote_counter += 1
        key = f"__QUOTE_{quote_counter}__"
        quoted_phrases[key] = match.group(1)
        return key
    
    # Replace quoted phrases with placeholders
    query_normalized = re.sub(r'"([^"]+)"', replace_quote, query)
    
    # Split by comma (but quoted phrases are already replaced, so commas inside quotes won't split)
    parts = [part.strip() for part in query_normalized.split(',') if part.strip()]
    
    # Process each part and restore quoted phrases
    and_terms = []
    for part in parts:
        # Restore quoted phrases if any placeholder is found
        term = part
        for placeholder, original in quoted_phrases.items():
            if placeholder in term:
                term = term.replace(placeholder, original)
        
        # Each comma-separated term becomes a separate AND condition
        if term.strip():
            and_terms.append([term.strip()])
    
    return {'and_terms': and_terms}


def parse_boolean_query(query: str) -> Dict:
    """
    Parse Boolean query into structured format.
    
    Handles:
    - Simple queries: "python"
    - AND queries: "python AND java"
    - OR queries: "python OR java"
    - Complex queries: ("Product Owner" OR "Product Manager") AND "Business" AND "Analyst"
    
    Example: ("Product Owner" OR "Product Manager") AND "Business" AND "Analyst"
    Returns: {
        'and_terms': [['Product Owner', 'Product Manager'], ['Business'], ['Analyst']]
    }
    """
    # Normalize query
    query = query.strip()
    if not query:
        return {'and_terms': []}
    
    # Extract quoted phrases first (preserve them)
    quoted_phrases = {}
    quote_counter = 0
    
    def replace_quote(match):
        nonlocal quote_counter
        quote_counter += 1
        key = f"__QUOTE_{quote_counter}__"
        quoted_phrases[key] = match.group(1)
        return key
    
    # Replace quoted phrases with placeholders
    query_normalized = re.sub(r'"([^"]+)"', replace_quote, query)
    
    # Split by AND (case-insensitive)
    and_parts = re.split(r'\s+AND\s+', query_normalized, flags=re.IGNORECASE)
    
    # Process each AND part
    and_terms = []
    for part in and_parts:
        part = part.strip()
        if not part:
            continue
        
        # Remove outer parentheses if present
        part = part.strip('()')
        
        # Check for OR groups (inside parentheses or just OR separated)
        if ' OR ' in part.upper() or '|' in part:
            # Split by OR
            or_terms = re.split(r'\s+OR\s+', part, flags=re.IGNORECASE)
            or_list = [t.strip().strip('()') for t in or_terms if t.strip()]
        else:
            or_list = [part]
        
        # Replace placeholders back with quoted phrases
        restored_terms = []
        for term in or_list:
            if term in quoted_phrases:
                restored_terms.append(quoted_phrases[term])
            else:
                restored_terms.append(term)
        
        and_terms.append(restored_terms)
    
    return {'and_terms': and_terms}


def build_searchable_text(metadata: Dict) -> str:
    """
    Build searchable text from candidate metadata.
    
    Includes all searchable fields: skills, location, name, company, domain, education, summary.
    This ensures Boolean queries can match candidates by any text field.
    """
    searchable_fields = [
        # Skills
        'primary_skills', 'secondary_skills', 'all_skills',
        # Personal info
        'name', 'email',
        # Professional info
        'current_company', 'current_designation', 'current_location',
        'resume_summary',
        # Background
        'domain', 'education',
        # Additional fields that might be in metadata
        'certifications', 'preferred_locations'
    ]
    
    text_parts = []
    for field in searchable_fields:
        value = metadata.get(field, '')
        if value and value != 'Unknown' and value != 'No skills' and value != 'No email':
            # Convert to string and lowercase for case-insensitive matching
            text_parts.append(str(value).lower())
    
    return ' '.join(text_parts)


def matches_boolean_query(candidate_text: str, parsed_query: Dict) -> bool:
    """
    Check if candidate text matches Boolean query.
    
    Uses word boundary matching to ensure tech skills are matched as standalone words
    or comma-separated terms, not as parts of other words (e.g., "net" in "outlook.com").
    
    Args:
        candidate_text: Lowercased searchable text from candidate
        parsed_query: Parsed Boolean query structure
        
    Returns:
        True if candidate matches all AND conditions
    """
    and_terms = parsed_query.get('and_terms', [])
    
    if not and_terms:
        return True  # No query terms means match all
    
    def _is_valid_match(term: str, text: str) -> bool:
        """
        Check if term matches in text using word boundaries, excluding email/URL contexts.
        
        Args:
            term: Search term (e.g., "net", ".net", "c#", "c++")
            text: Text to search in
            
        Returns:
            True if term is found as a valid tech skill (not in email/URL)
        """
        if not term:
            return False
        
        term_lower = term.lower().strip().strip('"')
        if not term_lower:
            return False
        
        escaped = re.escape(term_lower)
        
        # Handle special cases with non-word characters (., #, +, etc.)
        # Word boundaries \b don't work properly with special characters
        # So we use context-based matching (start, comma, semicolon, space)
        if term_lower.startswith('.') or '#' in term_lower or '+' in term_lower:
            # For ".net", "c#", "c++", "f#", etc.
            # Match at start, after comma, semicolon, or space
            # Pattern: (start|comma|semicolon|space) + term + (comma|semicolon|space|end)
            pattern = r'(?:^|[,;\s]+)' + escaped + r'(?:[,;\s]+|$)'
        else:
            # For regular terms, use word boundaries
            pattern = r'\b' + escaped + r'\b'
        
        # Find all matches
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if not matches:
            return False
        
        # Check each match to exclude email/URL contexts
        for match in matches:
            start, end = match.span()
            
            # Extract context around the match (100 chars before and after for better detection)
            context_start = max(0, start - 100)
            context_end = min(len(text), end + 100)
            context = text[context_start:context_end]
            match_pos_in_context = start - context_start
            
            # Exclude if match is part of an email address
            # Email pattern: [text]@[domain]
            # Check if @ appears before our match and domain extension appears after
            at_pos = context.find('@')
            if at_pos != -1 and at_pos < match_pos_in_context:
                # @ is before our match - check if domain extension is after
                domain_pattern = r'\.(com|net|org|edu|gov|io|co|in|uk|au|ca|us|de|fr|jp)(?:\b|/|$)'
                domain_match = re.search(domain_pattern, context[match_pos_in_context:], re.IGNORECASE)
                if domain_match:
                    # Match is between @ and domain extension - it's in an email
                    continue
            
            # Exclude if match is part of an email address or domain name
            # Get context around the match (50 chars before and after)
            context_before = text[max(0, start - 50):start]
            context_after = text[end:min(len(text), end + 50)]
            
            # Check 1: If @ appears before our match, it might be in an email
            # Look for email pattern: [text]@[domain].[extension]
            if '@' in context_before:
                # Check if there's a domain extension after our match
                domain_extensions = r'\.(com|net|org|edu|gov|io|co|in|uk|au|ca|us|de|fr|jp)(?:\b|/|$)'
                if re.search(domain_extensions, context_after, re.IGNORECASE):
                    # Pattern: @[something]...[our_match]...[extension] = email address
                    continue  # Skip, it's in an email
            
            # Check 2: If match is within a domain pattern [word].[extension]
            # Look for domain pattern that contains our match
            # Pattern: [alphanumeric-dots].[extension]
            domain_full_pattern = r'[a-z0-9.-]+\.(com|net|org|edu|gov|io|co|in|uk|au|ca|us|de|fr|jp)(?:\b|/|$)'
            # Search in wider context (100 chars total)
            wider_context = text[max(0, start - 50):min(len(text), end + 50)]
            domain_matches = list(re.finditer(domain_full_pattern, wider_context, re.IGNORECASE))
            for dm in domain_matches:
                dm_start = max(0, start - 50) + dm.start()
                dm_end = max(0, start - 50) + dm.end()
                # If our match position is within the domain match, exclude it
                if start >= dm_start and end <= dm_end:
                    continue  # Skip, it's part of a domain name
            
            # Valid match found (not in email/URL context)
            return True
        
        # All matches were in email/URL context
        return False
    
    # All AND groups must match
    for or_group in and_terms:
        # At least one term in OR group must match
        group_matched = False
        for term in or_group:
            if _is_valid_match(term, candidate_text):
                group_matched = True
                break
        
        # If any AND group doesn't match, candidate fails
        if not group_matched:
            return False
    
    return True


@app.route('/api/searchResumes', methods=['POST'])
def search_resumes():
    """
    AI Search with Role + Skill + Profile Type logic.
    
    Requirements:
    1. Role + skill query ("java developer"):
       - Extract profile_type and sub_role_type from query
       - Derive role_type from sub_role_type
       - Filter by role_type AND profile_type
    
    2. Skill-only query ("ASP.NET"):
       - Extract profile_type from skill (using profile_type_utils)
       - Infer sub_role_type from profile_type
       - Derive role_type from sub_role_type
       - Filter by role_type AND profile_type (or profile_type only as fallback)
    
    3. Name search ("sesha"):
       - Fuzzy match on LOWER(name) LIKE '%search_term%'
    
    4. Multi-skill AND logic:
       - All detected skill scores > 0 (AND condition)
    
    5. Input validation:
       - Require at least one role OR one skill (not both)
    
    Input: {"query": "<search_text>", "limit": 50}
    Output: JSON with matching candidates
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json() or {}
        query = (data.get('query') or '').strip()
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Get limit - if not provided, return all results (None = no limit)
        limit_param = data.get('limit')
        limit = int(limit_param) if limit_param is not None else None
        use_semantic = data.get('use_semantic', True)  # Enable semantic search by default
        use_boolean = data.get('use_boolean', True)  # Enable Boolean filter by default
        logger.info(f"Processing /api/searchResumes query: {query}, use_semantic: {use_semantic}, use_boolean: {use_boolean}")
        start_time = time.time()
        
        # === STEP 1: Name Detection ===
        if looks_like_name(query):
            logger.info(f"Detected name search for: {query}")
            with create_ats_database() as db:
                sql = """
                    SELECT 
                        rm.candidate_id,
                        rm.name,
                        rm.email,
                        rm.phone,
                        rm.total_experience,
                        rm.primary_skills,
                        rm.secondary_skills,
                        rm.all_skills,
                        rm.profile_type,
                        rm.role_type,
                        rm.subrole_type,
                        rm.sub_profile_type,
                        rm.current_designation,
                        rm.current_location,
                        rm.current_company,
                        rm.domain,
                        rm.education,
                        rm.resume_summary,
                        rm.created_at
                    FROM resume_metadata rm
                    WHERE rm.status = 'active'
                      AND LOWER(rm.name) LIKE %s
                    ORDER BY rm.total_experience DESC
                    LIMIT %s
                """
                db.cursor.execute(sql, (f"%{query.lower()}%", limit))
                results = db.cursor.fetchall()
                
                candidates = []
                for r in results:
                    candidates.append({
                        'candidate_id': r['candidate_id'],
                        'name': r['name'],
                        'email': r.get('email', ''),
                        'phone': r.get('phone', ''),
                        'total_experience': r.get('total_experience', 0),
                        'primary_skills': r.get('primary_skills', ''),
                        'secondary_skills': r.get('secondary_skills', ''),
                        'all_skills': r.get('all_skills', ''),
                        'profile_type': r.get('profile_type', ''),
                        'role_type': r.get('role_type', ''),
                        'subrole_type': r.get('subrole_type', ''),
                        'sub_profile_type': r.get('sub_profile_type', ''),
                        'current_designation': r.get('current_designation', ''),
                        'current_location': r.get('current_location', ''),
                        'current_company': r.get('current_company', ''),
                        'domain': r.get('domain', ''),
                        'education': r.get('education', ''),
                        'resume_summary': r.get('resume_summary', ''),
                    })
                
                return jsonify({
                    'query': query,
                    'search_type': 'name_search',
                    'analysis': {'detected_as': 'name'},
                    'count': len(candidates),
                    'results': candidates,
                    'processing_time_ms': int((time.time() - start_time) * 1000),
                    'timestamp': datetime.now().isoformat()
                }), 200
        
        # === STEP 2: Role Detection ===
        role_info = detect_subrole_from_query(query)
        detected_subrole = role_info['sub_role'] if role_info else None
        detected_main_role = role_info['main_role'] if role_info else None
        detected_profile_type_from_role = role_info['profile_type'] if role_info else None
        
        # === STEP 3: Skill Extraction ===
        detected_skills = extract_master_skills(query)
        skill_score_columns = [s['score_column'] for s in detected_skills]
        skill_profile_types = [s['profile_type'] for s in detected_skills]
        first_skill_profile_type = skill_profile_types[0] if skill_profile_types else None
        
        # === STEP 4: Profile Type Detection (for skill-only queries) ===
        if not detected_profile_type_from_role and first_skill_profile_type:
            # Use profile_type from first skill
            detected_profile_type_from_role = first_skill_profile_type
            # Infer sub-role from profile_type
            if not detected_subrole:
                inferred_role_info = infer_subrole_from_profile_type(first_skill_profile_type)
                if inferred_role_info:
                    detected_subrole = inferred_role_info['sub_role']
                    detected_main_role = inferred_role_info['main_role']
        
        # === STEP 5: Input Validation ===
        if not detected_subrole and not detected_skills:
            return jsonify({
                'error': 'Please refine the search and include at least one role or one skill.'
            }), 400
        
        # === STEP 6: Build SQL Query ===
        where_clauses = ["rm.status = 'active'"]
        params = []
        
        # Role filter (only role_type)
        if detected_main_role:
            where_clauses.append("rm.role_type = %s")
            params.append(detected_main_role)
        
        # Profile Type filter
        profile_type_filters = []
        if detected_profile_type_from_role:
            profile_type_filters.append("rm.profile_type LIKE %s")
            params.append(f"%{detected_profile_type_from_role}%")
        # Also add profile types from skills
        for pt in skill_profile_types:
            if pt not in [detected_profile_type_from_role] if detected_profile_type_from_role else []:
                profile_type_filters.append("rm.profile_type LIKE %s")
                params.append(f"%{pt}%")
        
        if profile_type_filters:
            where_clauses.append("(" + " OR ".join(profile_type_filters) + ")")
        
        # Multi-skill AND filter (all scores > 0) - Only apply when multiple skills detected
        if skill_score_columns and len(detected_skills) > 1:
            # Need to join with candidate_profile_scores
            for col in skill_score_columns:
                where_clauses.append(f"cps.{col} > 0")
        
        # Build final SQL
        join_clause = ""
        if skill_score_columns and len(detected_skills) > 1:
            join_clause = "INNER JOIN candidate_profile_scores cps ON rm.candidate_id = cps.candidate_id"
        
        # Conditionally add LIMIT clause
        limit_clause = "LIMIT %s" if limit is not None else ""
        
        sql = f"""
            SELECT 
                rm.candidate_id,
                rm.name,
                rm.email,
                rm.phone,
                rm.total_experience,
                rm.primary_skills,
                rm.secondary_skills,
                rm.all_skills,
                rm.profile_type,
                rm.role_type,
                rm.subrole_type,
                rm.sub_profile_type,
                rm.current_designation,
                rm.current_location,
                rm.current_company,
                rm.domain,
                rm.education,
                rm.resume_summary,
                rm.created_at
            FROM resume_metadata rm
            {join_clause}
            WHERE {' AND '.join(where_clauses)}
            ORDER BY rm.total_experience DESC
            {limit_clause}
        """
        if limit is not None:
            params.append(limit)
        
        # === STEP 7: Execute SQL Query ===
        with create_ats_database() as db:
            db.cursor.execute(sql, params)
            results = db.cursor.fetchall()
            
            sql_candidates = []
            for r in results:
                sql_candidates.append({
                    'candidate_id': r['candidate_id'],
                    'name': r['name'],
                    'email': r.get('email', ''),
                    'phone': r.get('phone', ''),
                    'total_experience': r.get('total_experience', 0),
                    'primary_skills': r.get('primary_skills', ''),
                    'secondary_skills': r.get('secondary_skills', ''),
                    'all_skills': r.get('all_skills', ''),
                    'profile_type': r.get('profile_type', ''),
                    'role_type': r.get('role_type', ''),
                    'subrole_type': r.get('subrole_type', ''),
                    'sub_profile_type': r.get('sub_profile_type', ''),
                    'current_designation': r.get('current_designation', ''),
                    'current_location': r.get('current_location', ''),
                    'current_company': r.get('current_company', ''),
                    'domain': r.get('domain', ''),
                    'education': r.get('education', ''),
                    'resume_summary': r.get('resume_summary', ''),
                })
        
        # === STEP 8: Semantic Search (Hybrid Approach) ===
        semantic_applied = False
        final_candidates = sql_candidates
        semantic_scores = {}
        
        # Initialize match_score for SQL-only candidates (will be updated if semantic search succeeds)
        for candidate in final_candidates:
            candidate['match_score'] = 1.0  # Perfect SQL match (default)
        
        if use_semantic and sql_candidates and ATSConfig.USE_PINECONE and ATSConfig.PINECONE_API_KEY:
            try:
                logger.info(f"Applying semantic search to {len(sql_candidates)} SQL-filtered candidates")
                
                # Generate query embedding
                query_embedding = embedding_service.generate_embedding(query)
                logger.info(f"Generated query embedding with {len(query_embedding)} dimensions")
                
                # Initialize Pinecone manager
                from enhanced_pinecone_manager import EnhancedPineconeManager
                pinecone_manager = EnhancedPineconeManager(
                    api_key=ATSConfig.PINECONE_API_KEY,
                    index_name=ATSConfig.PINECONE_INDEX_NAME,
                    dimension=ATSConfig.EMBEDDING_DIMENSION
                )
                pinecone_manager.get_or_create_index()
                
                # Get candidate IDs from SQL results (limit to 1000 for performance)
                candidate_ids = [c['candidate_id'] for c in sql_candidates[:1000]]
                
                if candidate_ids:
                    # Perform semantic search on SQL-filtered candidates
                    # Use larger top_k to get better ranking, then limit later
                    # Handle None limit - use all candidates (up to 200) when limit is not provided
                    if limit is None:
                        vector_top_k = min(len(candidate_ids), 200)
                    else:
                        vector_top_k = min(limit * 3, len(candidate_ids), 200)
                    
                    # Build Pinecone filter for candidate IDs
                    # Pinecone filter format: {'candidate_id': {'$in': [id1, id2, ...]}}
                    pinecone_filter = {'candidate_id': {'$in': candidate_ids}}
                    
                    logger.info(f"Querying Pinecone with {len(candidate_ids)} candidate IDs, top_k={vector_top_k}")
                    
                    # Query Pinecone index
                    vector_results = pinecone_manager.index.query(
                        vector=query_embedding,
                        top_k=vector_top_k,
                        include_metadata=True,
                        filter=pinecone_filter
                    )
                    
                    # Extract semantic scores
                    for match in vector_results.matches:
                        candidate_id = match.metadata.get('candidate_id')
                        if candidate_id:
                            semantic_scores[candidate_id] = match.score
                    
                    logger.info(f"Semantic search found {len(semantic_scores)} candidates with scores")
                    
                    # Merge SQL candidates with semantic scores
                    # Create a map for fast lookup
                    candidate_map = {c['candidate_id']: c for c in sql_candidates}
                    
                    # Add semantic scores and calculate combined scores
                    for candidate_id, semantic_score in semantic_scores.items():
                        if candidate_id in candidate_map:
                            candidate_map[candidate_id]['semantic_score'] = round(semantic_score, 4)
                            # Combined score: 30% SQL match (implicit 1.0) + 70% semantic similarity
                            candidate_map[candidate_id]['combined_score'] = round(0.3 + (semantic_score * 0.7), 4)
                            # Set match_score to combined_score
                            candidate_map[candidate_id]['match_score'] = candidate_map[candidate_id]['combined_score']
                    
                    # For candidates without semantic scores, set match_score to 1.0 (perfect SQL match)
                    for candidate_id, candidate in candidate_map.items():
                        if 'match_score' not in candidate:
                            candidate['match_score'] = 1.0  # Perfect SQL match when no semantic score
                    
                    # Sort by combined_score (or semantic_score if available), then by experience
                    final_candidates = list(candidate_map.values())
                    final_candidates.sort(
                        key=lambda x: (
                            x.get('combined_score', x.get('semantic_score', 0)),
                            x.get('total_experience', 0)
                        ),
                        reverse=True
                    )
                    
                    semantic_applied = True
                    logger.info(f"Hybrid search completed: {len(final_candidates)} candidates ranked")
                    
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}, falling back to SQL-only results")
                logger.error(traceback.format_exc())
                # Fallback to SQL-only results (already in final_candidates)
                semantic_applied = False
                # Set match_score to 1.0 for SQL-only candidates (perfect SQL match)
                for candidate in final_candidates:
                    candidate['match_score'] = 1.0
        
        # === STEP 9: Boolean Filter ===
        boolean_applied = False
        vector_candidates_count = len(final_candidates)  # Count before Boolean filter
        
        if use_boolean and final_candidates:
            # Check if query contains Boolean operators
            has_boolean_operators = (
                ' AND ' in query.upper() or 
                ' OR ' in query.upper() or 
                ' NOT ' in query.upper() or
                ' in ' in query.lower()  # Location queries like "engineer in portland"
            )
            
            if has_boolean_operators:
                try:
                    logger.info(f"Applying Boolean filter to {len(final_candidates)} candidates")
                    
                    # Parse Boolean query
                    parsed_boolean = parse_boolean_query(query)
                    
                    if parsed_boolean.get('and_terms'):
                        # Filter candidates by Boolean logic
                        boolean_filtered = []
                        for candidate in final_candidates:
                            # Build searchable text from candidate metadata
                            candidate_text = build_searchable_text(candidate)
                            
                            # Check if candidate matches Boolean query
                            if matches_boolean_query(candidate_text, parsed_boolean):
                                boolean_filtered.append(candidate)
                        
                        boolean_applied = True
                        final_candidates = boolean_filtered
                        boolean_filtered_count = len(final_candidates)
                        logger.info(f"Boolean filter applied: {boolean_filtered_count}/{vector_candidates_count} candidates match Boolean logic")
                    else:
                        logger.info("Boolean query parsed but no terms found, skipping Boolean filter")
                        boolean_filtered_count = vector_candidates_count
                        
                except Exception as e:
                    logger.warning(f"Boolean filter failed: {e}, using unfiltered results")
                    logger.error(traceback.format_exc())
                    boolean_applied = False
                    boolean_filtered_count = vector_candidates_count
            else:
                logger.info("No Boolean operators detected, skipping Boolean filter")
                boolean_filtered_count = vector_candidates_count
        else:
            boolean_filtered_count = vector_candidates_count
        
        # Limit results (only if limit is provided)
        if limit is not None:
            final_candidates = final_candidates[:limit]
        
        # Build analysis response
        analysis = {
            'detected_subrole': detected_subrole,
            'detected_main_role': detected_main_role,
            'detected_profile_type': detected_profile_type_from_role,
            'detected_skills': [s['skill'] for s in detected_skills],
            'skill_profile_types': skill_profile_types,
            'skill_score_columns': skill_score_columns,
            'semantic_search_applied': semantic_applied,
            'boolean_filter_applied': boolean_applied,
            'sql_candidates_count': len(sql_candidates),
            'vector_candidates_count': vector_candidates_count if semantic_applied else len(sql_candidates),
            'boolean_filtered_count': boolean_filtered_count if boolean_applied else None,
            'final_candidates_count': len(final_candidates),
        }
        
        # Determine search type
        if boolean_applied and semantic_applied:
            search_type = 'sql_vector_boolean'
        elif semantic_applied:
            search_type = 'sql_vector'
        else:
            search_type = 'sql_only'
        
        return jsonify({
            'query': query,
            'search_type': search_type,
            'analysis': analysis,
            'count': len(final_candidates),
            'results': final_candidates,
            'processing_time_ms': int((time.time() - start_time) * 1000),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error in /api/searchResumes: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500



# ============================================================================
# Role Mapping Structures for /api/searchResume
# ============================================================================

# Role hierarchy: (main_role, {set of sub_roles})
ROLE_MAPPING = [
    ("Software Engineer", {
        ".NET Developer", "BI Analyst", "BI Developer", "C# Full Stack Developer", "Cloud Engineer",
        "Cold Fusion Developer", "Data Engineer", "Data Engineer & Contract Management", "Analyst",
        "Python Developer", "Data Engineering", "Data Scientist", "Data Scientist Intern", "ML Engineer",
        "Data & Visualization Engineer", "Dev Ops & Cloud Engineer", "Dev Ops Engineer",
        "Site Reliability Engineer", "System Administrator", "Development Operations Engineer",
        "Developer", "Firmware Engineer", "ETL Developer", "ETL Informatica Developer",
        "Frontend Developer", "Full Stack .NET Developer", "Full Stack Developer", "Full Stack Engineer",
        "Full-Stack .Net Developer", "Full-Stack Consultant", "Game Developer", "Senior Software Engineer",
        "Java Developer", "Java Full-Stack Developer", ".Net Full Stack Developer", "Microsoft .Net Developer",
        "Mobile Developer", "Power Apps Developer", "Power Platform Developer", "Power BI Developer",
        "Professional .Net Developer", "Programmer", "React .Net Full Stack Developer", "SQL Developer",
        "Salesforce", "Salesforce Developer", "Salesforce Administrator", ".Net Web Developer",
        "Front-End Web UI Developer", "Big Data Engineer", "Staff Machine Learning Engineer",
        "Service Now Developer", "Software Developer", "Software Developer Intern", "Software Engineer",
        "Software Engineer & Instructor", "Software Engineering", "Analyst Ii Software Engineering",
        "Business Intelligence Developer", "Cold Fusion Application Developer", "Platform Architect",
        "Dot Net Developer", "Full Stack Java Developer", "Java Full Stack Developer",
        "Net Full Stack Engineer", "Web Designer", "Web Developer", "Website Specialist",
        ".NET Full Stack Engineer", ".NET Lead", ".NET Software Engineer", "Azure Developer",
        "ALML Engineer", "Application Developer"
    }),
    ("Administrative", {
        "Executive Assistant", "Admin Assistant", "Administrative Assistant",
        "Administrative Coordinator", "Administrator", "Branch Admin", "Commercial Assistant",
        "Community Administrator", "Contract Administrator", "Mail Administrator",
        "Project Administrator", "Office Assistant", "Program & Operations Coordinator",
        "Scheduling Coordinator", "Supervision. Responsible Administrator",
        "Warranty Administrator"
    }),
    ("Analyst", {
        "Analyst", "Analyst II", "Data Analyst", "Data Analyst Intern", "Power BI Developer",
        "Security Specialist", "Dataanalyst2", "Detail-Focused Data Analyst", "Financial Analyst",
        "Fraud Analyst", "R Risk Analyst", "Network & Communications Analyst", "Pricing Analyst",
        "Procurement Analyst", "Professional Analyst", "Resource Analyst", "Pricing Analyst Iii",
        "Sales Operations Analyst", "Business Development Software Analyst", "Risk Analyst",
        "Sr. Data Analyst", "HR Transformation Analyst", "Business Intelligence Analyst",
        "Settlement Analyst", "T-SQL Programmer Analyst", "Third-Party Risk Analyst"
    }),
    ("Architect", {
        "Architecture", "Architecture Collaborators", "Architecture Experience", "Architect",
        "Business Architect", "Cloud Architect", "Senior Data Architect",
        "Freelance User Experience And User Interface Architect", "Results-Driven Solution Architect",
        "Seasoned Architect", "Solution Architect", "Technical Architect"
    }),
    ("Associate", {
        "Associate", "Associate Reliability", "Walmart Associate"
    }),
    ("Business Analyst", {
        "Business Analyst", "Business Analyst-Data Analytics", "IT Analyst", "Project Manager",
        "Technical Writer", "Functional Analyst", "SAS Certified Statistical Business Analyst",
        "Agile Business Analyst", "Functional Consultant", "Business System Analyst",
        "Statistical Business Analyst"
    }),
    ("Consultant", {
        "Consultant", "Consultant And Jira Project Admin", "Creative & Marketing Consultant",
        "Energy Consultant", "External Consultant", "Ocm Consultant",
        "Organizational Effectiveness Consultant", "Project Consultant", "Pseudo Consultant",
        "Consultant & Peoplesoft Business Lead", "Learning Consultant", "Management Consultant",
        "People Soft Functional Consultant", "Quantitative Risk Management Consultant",
        "Sr.Workday Consultant", "Technical Consultant"
    }),
    ("Database Administrator", {
        "DB2 Database Administrator", "Database Administrator", "Engineer", "Database Engineer",
        "Developer Consultant", "Project Manager- DBA Team"
    }),
    ("Designer", {
        "Designer", "Flow Designer", "Graphic Designer",
        "Graphic Designer & Interactive Designer", "Retail Consultant",
        "Marketing Graphic Designer", "UX Designer"
    }),
    ("Director", {
        "Assistant Director", "Delivery Director", "Director Of Marketing",
        "Marketing Director", "Program Director"
    }),
    ("Education", {
        "Teaching Assistant"
    }),
    ("Engineer", {
        "Chemical Engineer", "Engineer", "Engineering", "Office 365 Engineer",
        "Reporting Engineer", "Standards Engineer"
    }),
    ("Engineering Manager", {
        "Engineering Management", "Engineering Manager"
    }),
    ("Finance", {
        "Accounting Assistant", "Accounting Specialist", "Billing Specialist", "Escrow Officer"
    }),
    ("HR", {
        "HR Manager", "HR Consultant", "HR Coordinator", "Payroll & Benefit Administrator",
        "Human Resources Administrative Specialist", "Human Resources Manager",
        "Payroll Analyst", "Payroll Manager", "Payroll Specialist", "Recruiting Leader"
    }),
    ("Healthcare", {
        "CNA - Certified Nursing Assistant", "Personal Assistant"
    }),
    ("IT Manager", {
        "Credit Director- Credit & Collections", "Manager", "IT Manager", "Director"
    }),
    ("IT Support", {
        "Credit Analyst", "First-Line Support", "Helpdesk Technician", "IT Analyst", "IT Specialist",
        "IT Support", "IT Technician", "Medication Technician", "PC Repair Technician",
        "Pc Technician", "Desktop Support Technician", "Sterile Technician",
        "Technical Support Engineer", "Information Tech", "Technicians",
        "Tier 2 IT Specialist", "Tier 2 Technical Support Engineer"
    }),
    ("Intern", {
        "Intern", "Internship"
    }),
    ("Lead", {
        "Enterprise Transformation Leader", "Leader", "Leadership", "Leadhostess",
        "Ecommerce Lead", "Provide Lead", "Shift Leader"
    }),
    ("Manager", {
        "Account Manager", "Assistant Store Manager", "Case Manager", "Contract Manager",
        "District Manager", "Editorial Manager", "Enterprise Manager", "Field Manager",
        "Finance Manager", "Floor Manager", "Freelance Senior Digital Asset Manager",
        "General Manager", "Manager", "Manager II", "Marketing Manager", "Operations Manager",
        "PMO - Tranformation Manager", "Platform Manager", "SLA Manager",
        "Manager- Change Management", "Social Media Manager And Content Specialist",
        "Solution-Driven Customer Success Manager", "Support Manager", "Team Manager"
    }),
    ("Network Engineer", {
        "Cisco Certified Network Professional", "Network Administration", "Network Administrator",
        "Network Engineer", "Network Support Engineer"
    }),
    ("Product Manager", {
        "Product Manager", "Product Owner"
    }),
    ("Program Manager", {
        "Engineering Program Manager", "Program Analyst", "Program Coordinator", "Program Manager",
        "Digital Transformation Program Manager", "Technical Program Manager"
    }),
    ("Project Manager", {
        "Automation & Robotics Project Manager", "Proxy Lead Project Manager",
        "E-Learning IT Project Manager II", "Marketing & Creative Project Manager",
        "Marketing Project Manager", "Project Manager", "Project Manager Consultant",
        "Project Coordinator", "Manager", "Project Engineer", "Project Expeditor",
        "IT Project Manager", "Project Manager Independent Contractor",
        "Technical Project Manager"
    }),
    ("QA Engineer", {
        "508 Tester", "Automation Engineer", "QA Analyst", "QA Engineer", "Analyst", "Test Engineer"
    }),
    ("Sales", {
        "Account Executive", "Sales Associate", "Sales Technology"
    }),
    ("Security Engineer", {
        "Cybersecurity Engineer", "Grc Analyst", "IAM Consultant", "IAM Engineer",
        "Security Analyst", "Identity & Access Management Engineer",
        "Information Assurance Officer", "Information Security Analyst",
        "Network Security", "Office 365 Identity Management Engineer",
        "Penetration Tester", "Security Management", "Soc Analyst", "Cyber Security Analyst",
        "Network & Security Engineer", "Network Security Engineer",
        "Sci Cyber Security Team Lead"
    }),
    ("Specialist", {
        "Analytics Specialist", "Ap Specialist", "Contract Specialist", "Funding Specialist",
        "Learning And Development Specialist", "Legal Specialist",
        "Music Licensing Contract Specialist", "Program Support Specialist",
        "Project Specialist", "Specialist", "Tier-2 Benefits Account Specialist",
        "Training Specialist"
    }),
    ("Supervisor", {
        "Con Supervisor", "Service Supervisor", "Supervisor", "Supervisory"
    }),
    ("System Administrator", {
        "Unix System Administrator On Rhel", "System Administrator",
        "System Administrator Technician", "Business Analyst", "System Engineer"
    }),
    ("Technical Lead", {
        "Supervise Team Lead", "Team Lead", "Technical Lead"
    }),
    ("Technical Writer", {
        "Documentation Specialist", "Technical Writer"
    }),
    ("Writer", {
        "Co-Writer"
    })
]

# Build reverse mappings for fast lookup
SUBROLE_TO_MAIN_ROLE = {}
SUBROLE_TO_PROFILE_TYPE = {}

# Profile type inference from sub-role (when skill-only query)
PROFILE_TYPE_TO_DEFAULT_SUBROLE = {
    "Java": "Java Developer",
    ".Net": ".Net Developer",
    "Python": "Python Developer",
    "JavaScript": "React Developer",
    "Full Stack": "Full Stack Developer",
    "DevOps": "DevOps Engineer",
    "Data Engineering": "Data Engineer",
    "Data Science": "Data Scientist",
    "SAP": "SAP Consultant",
    "ERP": "ERP Analyst",
    "Database": "Database Administrator",
    "Business Intelligence (BI)": "BI Developer",
    "Salesforce": "Salesforce Developer",
    "Testing / QA": "QA Engineer",
    "Cloud / Infra": "Cloud Engineer",
    "Security Engineer": "Security Engineer",
    "Network Engineer": "Network Engineer",
}

# Master skill to profile type and score column mapping
MASTER_SKILL_TO_PROFILE_AND_SCORE = {
    "java": {"profile_type": "Java", "score_column": "java_score"},
    ".net": {"profile_type": ".Net", "score_column": "dotnet_score"},
    "dotnet": {"profile_type": ".Net", "score_column": "dotnet_score"},
    "asp.net": {"profile_type": ".Net", "score_column": "dotnet_score"},
    "aspnet": {"profile_type": ".Net", "score_column": "dotnet_score"},
    "c#": {"profile_type": ".Net", "score_column": "dotnet_score"},
    "csharp": {"profile_type": ".Net", "score_column": "dotnet_score"},
    "react": {"profile_type": "JavaScript", "score_column": "javascript_score"},
    "react js": {"profile_type": "JavaScript", "score_column": "javascript_score"},
    "reactjs": {"profile_type": "JavaScript", "score_column": "javascript_score"},
    "javascript": {"profile_type": "JavaScript", "score_column": "javascript_score"},
    "js": {"profile_type": "JavaScript", "score_column": "javascript_score"},
    "angular": {"profile_type": "JavaScript", "score_column": "javascript_score"},
    "vue": {"profile_type": "JavaScript", "score_column": "javascript_score"},
    "node": {"profile_type": "JavaScript", "score_column": "javascript_score"},
    "node.js": {"profile_type": "JavaScript", "score_column": "javascript_score"},
    "python": {"profile_type": "Python", "score_column": "python_score"},
    "sql": {"profile_type": "Database", "score_column": "database_score"},
    "database": {"profile_type": "Database", "score_column": "database_score"},
    "erp": {"profile_type": "ERP", "score_column": "erp_score"},
    "sap": {"profile_type": "SAP", "score_column": "sap_score"},
    "devops": {"profile_type": "DevOps", "score_column": "devops_score"},
    "full stack": {"profile_type": "Full Stack", "score_column": "fullstack_score"},
    "fullstack": {"profile_type": "Full Stack", "score_column": "fullstack_score"},
}

# Initialize reverse mappings
for main_role, sub_roles in ROLE_MAPPING:
    for sub_role in sub_roles:
        SUBROLE_TO_MAIN_ROLE[sub_role.lower()] = main_role
        # Try to infer profile_type from sub_role name
        sub_lower = sub_role.lower()
        if "java" in sub_lower and "developer" in sub_lower:
            SUBROLE_TO_PROFILE_TYPE[sub_role.lower()] = "Java"
        elif ".net" in sub_lower or "dotnet" in sub_lower or "c#" in sub_lower:
            SUBROLE_TO_PROFILE_TYPE[sub_role.lower()] = ".Net"
        elif "python" in sub_lower:
            SUBROLE_TO_PROFILE_TYPE[sub_role.lower()] = "Python"
        elif "react" in sub_lower or "javascript" in sub_lower or "js" in sub_lower:
            SUBROLE_TO_PROFILE_TYPE[sub_role.lower()] = "JavaScript"
        elif "full stack" in sub_lower or "fullstack" in sub_lower:
            SUBROLE_TO_PROFILE_TYPE[sub_role.lower()] = "Full Stack"
        elif "devops" in sub_lower or "dev ops" in sub_lower:
            SUBROLE_TO_PROFILE_TYPE[sub_role.lower()] = "DevOps"
        elif "data engineer" in sub_lower:
            SUBROLE_TO_PROFILE_TYPE[sub_role.lower()] = "Data Engineering"
        elif "data scientist" in sub_lower or "ml engineer" in sub_lower:
            SUBROLE_TO_PROFILE_TYPE[sub_role.lower()] = "Data Science"
        elif "sap" in sub_lower:
            SUBROLE_TO_PROFILE_TYPE[sub_role.lower()] = "SAP"
        elif "erp" in sub_lower:
            SUBROLE_TO_PROFILE_TYPE[sub_role.lower()] = "ERP"
        elif "database" in sub_lower or "dba" in sub_lower or "sql" in sub_lower:
            SUBROLE_TO_PROFILE_TYPE[sub_role.lower()] = "Database"
        elif "salesforce" in sub_lower:
            SUBROLE_TO_PROFILE_TYPE[sub_role.lower()] = "Salesforce"
        elif "qa" in sub_lower or "test" in sub_lower:
            SUBROLE_TO_PROFILE_TYPE[sub_role.lower()] = "Testing / QA"
        elif "cloud" in sub_lower:
            SUBROLE_TO_PROFILE_TYPE[sub_role.lower()] = "Cloud / Infra"
        elif "security" in sub_lower or "cyber" in sub_lower:
            SUBROLE_TO_PROFILE_TYPE[sub_role.lower()] = "Cyber Security"
        elif "bi" in sub_lower or "business intelligence" in sub_lower:
            SUBROLE_TO_PROFILE_TYPE[sub_role.lower()] = "Business Intelligence (BI)"


def normalize_text(text: str) -> str:
    """Normalize text: lowercase, remove extra spaces."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def looks_like_name(text: str) -> bool:
    """Detect if input looks like a candidate name."""
    words = text.strip().split()
    if 2 <= len(words) <= 4:
        if all(word.isalpha() for word in words):
            tech_keywords = [
                "java", "python", "react", "sql", "developer", "engineer", 
                "manager", "analyst", ".net", "asp", "c#", "javascript"
            ]
            text_lower = text.lower()
            if not any(kw in text_lower for kw in tech_keywords):
                return True
    return False


def detect_subrole_from_query(query: str) -> Optional[Dict[str, str]]:
    """
    Detect sub-role from query text.
    Returns: {'sub_role': '...', 'main_role': '...', 'profile_type': '...'} or None
    """
    query_normalized = normalize_text(query)
    
    # Try exact match first (case-insensitive)
    for main_role, sub_roles in ROLE_MAPPING:
        for sub_role in sub_roles:
            sub_normalized = normalize_text(sub_role)
            # Check if sub_role appears in query
            if sub_normalized in query_normalized or query_normalized in sub_normalized:
                profile_type = SUBROLE_TO_PROFILE_TYPE.get(sub_normalized)
                return {
                    'sub_role': sub_role,
                    'main_role': main_role,
                    'profile_type': profile_type
                }
    
    # Try partial match (sub_role contains query or vice versa)
    for main_role, sub_roles in ROLE_MAPPING:
        for sub_role in sub_roles:
            sub_normalized = normalize_text(sub_role)
            # Check if key words match
            query_words = set(query_normalized.split())
            sub_words = set(sub_normalized.split())
            if len(query_words & sub_words) >= 2:  # At least 2 words match
                profile_type = SUBROLE_TO_PROFILE_TYPE.get(sub_normalized)
                return {
                    'sub_role': sub_role,
                    'main_role': main_role,
                    'profile_type': profile_type
                }
    
    return None


def extract_master_skills(query: str) -> List[Dict[str, str]]:
    """
    Extract master skills from query.
    Returns: [{'skill': '...', 'profile_type': '...', 'score_column': '...'}, ...]
    """
    query_lower = normalize_text(query)
    detected_skills = []
    
    # Check each master skill
    for skill_key, mapping in MASTER_SKILL_TO_PROFILE_AND_SCORE.items():
        if skill_key in query_lower:
            detected_skills.append({
                'skill': skill_key,
                'profile_type': mapping['profile_type'],
                'score_column': mapping['score_column']
            })
    
    return detected_skills


def infer_subrole_from_profile_type(profile_type: str) -> Optional[Dict[str, str]]:
    """
    Infer sub-role from profile_type (for skill-only queries).
    Returns: {'sub_role': '...', 'main_role': '...', 'profile_type': '...'} or None
    """
    default_subrole = PROFILE_TYPE_TO_DEFAULT_SUBROLE.get(profile_type)
    if not default_subrole:
        return None
    
    main_role = SUBROLE_TO_MAIN_ROLE.get(default_subrole.lower())
    if not main_role:
        return None
    
    return {
        'sub_role': default_subrole,
        'main_role': main_role,
        'profile_type': profile_type
    }


@app.route('/api/searchResume', methods=['POST'])
def search_resume():
    """
    AI Search with Role + Skill + Profile Type logic.
    
    Requirements:
    1. Role + skill query ("java developer"):
       - Extract profile_type and sub_role_type from query
       - Derive role_type from sub_role_type
       - Filter by role_type AND profile_type
    
    2. Skill-only query ("ASP.NET"):
       - Extract profile_type from skill (using profile_type_utils)
       - Infer sub_role_type from profile_type
       - Derive role_type from sub_role_type
       - Filter by role_type AND profile_type (or profile_type only as fallback)
    
    3. Name search ("sesha"):
       - Fuzzy match on LOWER(name) LIKE '%search_term%'
    
    4. Multi-skill AND logic:
       - All detected skill scores > 0 (AND condition)
    
    5. Input validation:
       - Require at least one role OR one skill (not both)
    
    Input: {"query": "<search_text>", "limit": 50}
    Output: JSON with matching candidates
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json() or {}
        query = (data.get('query') or '').strip()
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        limit = int(data.get('limit', 50))
        logger.info(f"Processing /api/searchResume query: {query}")
        start_time = time.time()
        
        # === STEP 1: Name Detection ===
        if looks_like_name(query):
            logger.info(f"Detected name search for: {query}")
            with create_ats_database() as db:
                sql = """
                    SELECT 
                        rm.candidate_id,
                        rm.name,
                        rm.email,
                        rm.phone,
                        rm.total_experience,
                        rm.primary_skills,
                        rm.secondary_skills,
                        rm.all_skills,
                        rm.profile_type,
                        rm.role_type,
                        rm.subrole_type,
                        rm.sub_profile_type,
                        rm.current_designation,
                        rm.current_location,
                        rm.current_company,
                        rm.domain,
                        rm.education,
                        rm.resume_summary,
                        rm.created_at
                    FROM resume_metadata rm
                    WHERE rm.status = 'active'
                      AND LOWER(rm.name) LIKE %s
                    ORDER BY rm.total_experience DESC
                    LIMIT %s
                """
                db.cursor.execute(sql, (f"%{query.lower()}%", limit))
                results = db.cursor.fetchall()
                
                candidates = []
                for r in results:
                    candidates.append({
                        'candidate_id': r['candidate_id'],
                        'name': r['name'],
                        'email': r.get('email', ''),
                        'phone': r.get('phone', ''),
                        'total_experience': r.get('total_experience', 0),
                        'primary_skills': r.get('primary_skills', ''),
                        'secondary_skills': r.get('secondary_skills', ''),
                        'all_skills': r.get('all_skills', ''),
                        'profile_type': r.get('profile_type', ''),
                        'role_type': r.get('role_type', ''),
                        'subrole_type': r.get('subrole_type', ''),
                        'sub_profile_type': r.get('sub_profile_type', ''),
                        'current_designation': r.get('current_designation', ''),
                        'current_location': r.get('current_location', ''),
                        'current_company': r.get('current_company', ''),
                        'domain': r.get('domain', ''),
                        'education': r.get('education', ''),
                        'resume_summary': r.get('resume_summary', ''),
                    })
                
                return jsonify({
                    'query': query,
                    'search_type': 'name_search',
                    'analysis': {'detected_as': 'name'},
                    'count': len(candidates),
                    'results': candidates,
                    'processing_time_ms': int((time.time() - start_time) * 1000),
                    'timestamp': datetime.now().isoformat()
                }), 200
        
        # === STEP 2: Role Detection ===
        role_info = detect_subrole_from_query(query)
        detected_subrole = role_info['sub_role'] if role_info else None
        detected_main_role = role_info['main_role'] if role_info else None
        detected_profile_type_from_role = role_info['profile_type'] if role_info else None
        
        # === STEP 3: Skill Extraction ===
        detected_skills = extract_master_skills(query)
        skill_score_columns = [s['score_column'] for s in detected_skills]
        skill_profile_types = [s['profile_type'] for s in detected_skills]
        first_skill_profile_type = skill_profile_types[0] if skill_profile_types else None
        
        # === STEP 4: Profile Type Detection (for skill-only queries) ===
        if not detected_profile_type_from_role and first_skill_profile_type:
            # Use profile_type from first skill
            detected_profile_type_from_role = first_skill_profile_type
            # Infer sub-role from profile_type
            if not detected_subrole:
                inferred_role_info = infer_subrole_from_profile_type(first_skill_profile_type)
                if inferred_role_info:
                    detected_subrole = inferred_role_info['sub_role']
                    detected_main_role = inferred_role_info['main_role']
        
        # === STEP 5: Input Validation ===
        if not detected_subrole and not detected_skills:
            return jsonify({
                'error': 'Please refine the search and include at least one role or one skill.'
            }), 400
        
        # === STEP 6: Build SQL Query ===
        where_clauses = ["rm.status = 'active'"]
        params = []
        
        # Role filter (only role_type)
        if detected_main_role:
            where_clauses.append("rm.role_type = %s")
            params.append(detected_main_role)
        
        # Profile Type filter
        profile_type_filters = []
        if detected_profile_type_from_role:
            profile_type_filters.append("rm.profile_type LIKE %s")
            params.append(f"%{detected_profile_type_from_role}%")
        # Also add profile types from skills
        for pt in skill_profile_types:
            if pt not in [detected_profile_type_from_role] if detected_profile_type_from_role else []:
                profile_type_filters.append("rm.profile_type LIKE %s")
                params.append(f"%{pt}%")
        
        if profile_type_filters:
            where_clauses.append("(" + " OR ".join(profile_type_filters) + ")")
        
        # Multi-skill AND filter (all scores > 0) - Only apply when multiple skills detected
        if skill_score_columns and len(detected_skills) > 1:
            # Need to join with candidate_profile_scores
            for col in skill_score_columns:
                where_clauses.append(f"cps.{col} > 0")
        
        # Build final SQL
        join_clause = ""
        if skill_score_columns and len(detected_skills) > 1:
            join_clause = "INNER JOIN candidate_profile_scores cps ON rm.candidate_id = cps.candidate_id"
        
        sql = f"""
            SELECT 
                rm.candidate_id,
                rm.name,
                rm.email,
                rm.phone,
                rm.total_experience,
                rm.primary_skills,
                rm.secondary_skills,
                rm.all_skills,
                rm.profile_type,
                rm.role_type,
                rm.subrole_type,
                rm.sub_profile_type,
                rm.current_designation,
                rm.current_location,
                rm.current_company,
                rm.domain,
                rm.education,
                rm.resume_summary,
                rm.created_at
            FROM resume_metadata rm
            {join_clause}
            WHERE {' AND '.join(where_clauses)}
            ORDER BY rm.total_experience DESC
            LIMIT %s
        """
        params.append(limit)
        
        # === STEP 7: Execute Query ===
        with create_ats_database() as db:
            db.cursor.execute(sql, params)
            results = db.cursor.fetchall()
            
            candidates = []
            for r in results:
                candidates.append({
                    'candidate_id': r['candidate_id'],
                    'name': r['name'],
                    'email': r.get('email', ''),
                    'phone': r.get('phone', ''),
                    'total_experience': r.get('total_experience', 0),
                    'primary_skills': r.get('primary_skills', ''),
                    'secondary_skills': r.get('secondary_skills', ''),
                    'all_skills': r.get('all_skills', ''),
                    'profile_type': r.get('profile_type', ''),
                    'role_type': r.get('role_type', ''),
                    'subrole_type': r.get('subrole_type', ''),
                    'sub_profile_type': r.get('sub_profile_type', ''),
                    'current_designation': r.get('current_designation', ''),
                    'current_location': r.get('current_location', ''),
                    'current_company': r.get('current_company', ''),
                    'domain': r.get('domain', ''),
                    'education': r.get('education', ''),
                    'resume_summary': r.get('resume_summary', ''),
                })
        
        # Build analysis response
        analysis = {
            'detected_subrole': detected_subrole,
            'detected_main_role': detected_main_role,
            'detected_profile_type': detected_profile_type_from_role,
            'detected_skills': [s['skill'] for s in detected_skills],
            'skill_profile_types': skill_profile_types,
            'skill_score_columns': skill_score_columns,
        }
        
        return jsonify({
            'query': query,
            'search_type': 'role_skill_search',
            'analysis': analysis,
            'count': len(candidates),
            'results': candidates,
            'processing_time_ms': int((time.time() - start_time) * 1000),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error in /api/searchResume: {e}")
        logger.error(traceback.format_exc())
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
    - min_match_percent: Minimum match percentage threshold (default: 50)
      Only candidates meeting this threshold OR having at least 1 matching skill are returned
    
    Returns: Ranked list of ELIGIBLE candidates only (not all candidates)
    
    ELIGIBILITY FILTERING:
    - Candidates are filtered to include only those who meet the eligibility criteria
    - Eligibility criteria: match_percent >= min_match_percent OR having at least 1 matching skill
    - By default, min_match_percent = 50 (candidates with <50% match are excluded)
    - Only eligible candidates are stored in database and returned in response
    
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
        
        # Determine profile types relevant to this JD
        inferred_profile_types = infer_profile_type_from_requirements(required_skills_list, job_description)
        
        candidate_filters = {
            'domain': domain,
            'education': education_required,
            'min_experience': min_experience,
            'max_experience': max_experience,
            'primary_skills': required_skills_list[:5] if required_skills_list else None,
        }
        if inferred_profile_types:
            candidate_filters['profile_type'] = inferred_profile_types
        
        logger.info(f"Retrieving candidate profiles with structured filters: {candidate_filters}")
        with create_ats_database() as db:
            candidates = db.filter_candidates(candidate_filters, limit=data.get('sql_limit', SQL_LAYER_LIMIT))
        
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
        
        # Get minimum match percent threshold from request (default: 50)
        min_match_percent = data.get('min_match_percent', 50)
        
        logger.info(f"Filtering candidates with match_percent >= {min_match_percent}")
        
        # Filter to eligible candidates only
        eligible_profiles = []
        for profile in ranked_profiles:
            # Eligible if: match_percent >= threshold OR has at least 1 matching skill
            has_matching_skill = len(profile.get('matched_skills', [])) > 0
            meets_match_threshold = profile.get('match_percent', 0) >= min_match_percent
            
            if meets_match_threshold or has_matching_skill:
                eligible_profiles.append(profile)
        
        logger.info(f"Found {len(eligible_profiles)} eligible candidates out of {len(ranked_profiles)} total ranked")
        
        # Store ranking results in database (only for eligible candidates)
        with create_ats_database() as db:
            for profile in eligible_profiles:
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
        eligible_count = len(eligible_profiles)
        total_count = len(ranked_profiles)
        
        response_data = {
            'status': 'success',
            'message': f'Profile ranking completed. Found {eligible_count} eligible candidates out of {total_count} total ranked candidates.',
            'job_id': job_id,
            'ranked_profiles': eligible_profiles,
            'total_candidates_evaluated': len(candidates),
            'total_candidates_ranked': len(ranked_profiles),
            'eligible_candidates_returned': len(eligible_profiles),
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
                'semantic_similarity': 'enabled',
                'min_match_percent': min_match_percent
            },
            'timestamp': datetime.now().isoformat()
        }
        
        if eligible_profiles:
            logger.info(f"Ranking completed. Top candidate: {eligible_profiles[0]['name']} with score {eligible_profiles[0]['total_score']}")
        else:
            logger.info("Ranking completed. No eligible candidates found (all candidates filtered out).")
        
        # Store latest result in memory cache (overwrites previous)
        global _latest_ranking_cache
        _latest_ranking_cache = {
            'ranked_profiles': eligible_profiles,
            'job_requirements': {
                'required_skills': required_skills_list,
                'preferred_skills': preferred_skills_list,
                'min_experience': min_experience,
                'max_experience': max_experience,
                'domain': domain,
                'education_required': education_required
            },
            'timestamp': datetime.now().isoformat(),
            'job_id': job_id
        }
        logger.info(f"Stored {len(eligible_profiles)} candidates in memory cache for comprehensive ranking")
        
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
    
    Input: JSON with job requirements and either:
    - Option 1: candidates array (from /api/profileRankingByJD output) - PRIORITY
    - Option 2: profiles_directory (read from file system) - FALLBACK
    
    Output: Ranked list of candidates with detailed analysis
    
    Workflow:
    1. If 'candidates' provided: Use those directly (from profileRankingByJD)
    2. If 'candidates' NOT provided: Read from profiles_directory (current behavior)
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Get cached data from latest profileRankingByJD result
        global _latest_ranking_cache
        cached_job_requirements = _latest_ranking_cache.get('job_requirements', {})
        cached_candidates = _latest_ranking_cache.get('ranked_profiles', [])
        
        # Extract job requirements with priority: request > cache
        request_job_requirements = data.get('job_requirements', {})
        
        # Merge: Request overrides cache, cache fills missing fields
        if cached_job_requirements:
            # Start with cached job_requirements, then override with request values
            job_requirements = {**cached_job_requirements, **request_job_requirements}
            if request_job_requirements:
                logger.info("Using job_requirements: cached values merged with request overrides")
            else:
                logger.info("Using job_requirements from latest profileRankingByJD result (cached)")
        else:
            # No cache, use request only
            job_requirements = request_job_requirements
            if request_job_requirements:
                logger.info("Using job_requirements from request (no cache available)")
        
        # Validate job requirements (must have at least one source)
        if not job_requirements:
            return jsonify({
                'error': 'job_requirements is required. Either provide in request or call /api/profileRankingByJD first to populate cache'
            }), 400
        
        # Normalize job_requirements: Convert skills from lists to comma-separated strings
        # The ranking engine expects required_skills and preferred_skills as strings, not lists
        required_skills = job_requirements.get('required_skills', '')
        if isinstance(required_skills, list):
            required_skills = ', '.join(str(s).strip() for s in required_skills if s)
        elif not isinstance(required_skills, str):
            required_skills = ''
        
        preferred_skills = job_requirements.get('preferred_skills', '')
        if isinstance(preferred_skills, list):
            preferred_skills = ', '.join(str(s).strip() for s in preferred_skills if s)
        elif not isinstance(preferred_skills, str):
            preferred_skills = ''
        
        # Update job_requirements with normalized skills
        job_requirements['required_skills'] = required_skills
        job_requirements['preferred_skills'] = preferred_skills
        
        profiles_dir = data.get('profiles_directory', os.path.join(os.getcwd(), 'profiles'))
        top_k = data.get('top_k', 10)
        input_candidates = data.get('candidates', [])  # Candidates from profileRankingByJD (manual override)
        
        start_time = time.time()
        
        if cached_candidates and len(cached_candidates) > 0:
            logger.info(f"Using {len(cached_candidates)} candidates from latest profileRankingByJD result (stored at {_latest_ranking_cache.get('timestamp', 'unknown')})")
            
            # Normalize candidates from profileRankingByJD format to ranking engine format
            candidates = []
            for candidate_data in cached_candidates:
                try:
                    # Handle both string and list formats for skills
                    primary_skills = candidate_data.get('primary_skills', '')
                    if isinstance(primary_skills, list):
                        primary_skills = ', '.join(primary_skills)
                    
                    secondary_skills = candidate_data.get('secondary_skills', '')
                    if isinstance(secondary_skills, list):
                        secondary_skills = ', '.join(secondary_skills)
                    
                    # Build candidate object in format expected by ranking engine
                    candidate = {
                        'candidate_id': candidate_data.get('candidate_id'),
                        'name': candidate_data.get('name', 'Unknown'),
                        'email': candidate_data.get('email', ''),
                        'phone': candidate_data.get('phone', ''),
                        'primary_skills': primary_skills,
                        'secondary_skills': secondary_skills,
                        'total_experience': candidate_data.get('total_experience', 0),
                        'domain': candidate_data.get('domain', ''),
                        'education': candidate_data.get('education', ''),
                        'resume_text': candidate_data.get('resume_text', ''),
                        'status': candidate_data.get('status', 'active')
                    }
                    candidates.append(candidate)
                    
                except Exception as e:
                    logger.error(f"Error processing candidate {candidate_data.get('candidate_id')}: {e}")
                    # Skip invalid candidates
                    continue
            
            if candidates:
                logger.info(f"Successfully normalized {len(candidates)} candidates from cache")
                source_info = f"latest profileRankingByJD result (cached, {len(candidates)} candidates)"
            else:
                logger.warning("Cached candidates found but none were valid, falling back to other sources")
        
        # Priority 2: Use candidates from request (if provided manually and cache not used)
        if ('candidates' not in locals() or not candidates) and input_candidates and isinstance(input_candidates, list) and len(input_candidates) > 0:
            logger.info(f"Using {len(input_candidates)} candidates from request (from profileRankingByJD)")
            
            # Normalize candidates from profileRankingByJD format to ranking engine format
            candidates = []
            for candidate_data in input_candidates:
                try:
                    # Handle both string and list formats for skills
                    primary_skills = candidate_data.get('primary_skills', '')
                    if isinstance(primary_skills, list):
                        primary_skills = ', '.join(primary_skills)
                    
                    secondary_skills = candidate_data.get('secondary_skills', '')
                    if isinstance(secondary_skills, list):
                        secondary_skills = ', '.join(secondary_skills)
                    
                    # Build candidate object in format expected by ranking engine
                    candidate = {
                        'candidate_id': candidate_data.get('candidate_id'),
                        'name': candidate_data.get('name', 'Unknown'),
                        'email': candidate_data.get('email', ''),
                        'phone': candidate_data.get('phone', ''),
                        'primary_skills': primary_skills,
                        'secondary_skills': secondary_skills,
                        'total_experience': candidate_data.get('total_experience', 0),
                        'domain': candidate_data.get('domain', ''),
                        'education': candidate_data.get('education', ''),
                        'resume_text': candidate_data.get('resume_text', ''),
                        'status': candidate_data.get('status', 'active')
                    }
                    candidates.append(candidate)
                    
                except Exception as e:
                    logger.error(f"Error processing candidate {candidate_data.get('candidate_id')}: {e}")
                    # Skip invalid candidates
                    continue
            
            if candidates:
                logger.info(f"Successfully normalized {len(candidates)} candidates from manual request")
                source_info = f"candidates from request ({len(candidates)} candidates)"
            else:
                logger.warning("Manual candidates provided but none were valid, falling back to directory")
        
        # Priority 3: Read from directory (fallback - current behavior)
        if 'candidates' not in locals() or not candidates:
            logger.info(f"Reading profiles from directory: {profiles_dir}")
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
            
            logger.info(f"Found {len(profiles)} profiles from directory")
            
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
            
            source_info = f"profiles_directory ({profiles_dir}, {len(profiles)} files)"
        
        # Initialize ranking engine (if not already initialized)
        if 'ranking_engine' not in locals():
            ranking_engine = create_ranking_engine()
        
        # Rank candidates using existing ranking engine
        logger.info(f"Ranking {len(candidates)} candidates against job requirements...")
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
            'source': source_info,
            'ranked_profiles': ranked_profiles,
            'total_candidates_evaluated': len(candidates),
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
        
        # Include profiles_directory in response only if it was used
        if 'profiles_dir' in locals() and not (input_candidates and len(input_candidates) > 0):
            response_data['profiles_directory'] = profiles_dir
        
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
    """Extract text from PDF, DOCX, or DOC files"""
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
        
        elif file_path.lower().endswith('.docx'):
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
        
        elif file_path.lower().endswith('.doc'):
            # DOC files (older binary format) require different libraries
            try:
                # Try NT-TextFileLoader first
                from nt_textfileloader import TextFileLoader
                loader = TextFileLoader()
                text = loader.load(file_path)
                if text and isinstance(text, str) and text.strip():
                    return text.strip()
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"NT-TextFileLoader failed: {e}, trying textract")
            
            try:
                import textract
                text = textract.process(file_path).decode('utf-8')
                return text.strip()
            except ImportError:
                try:
                    import pypandoc
                    text = pypandoc.convert_file(file_path, 'plain')
                    return text.strip()
                except ImportError:
                    logger.warning("No DOC parsing library available")
                    return f"DOC file content for {os.path.basename(file_path)} - DOC parsing requires 'NT-TextFileLoader', 'textract', or 'pypandoc' library"
            except Exception as e:
                logger.error(f"Error parsing DOC file: {e}")
                return f"Error extracting text from DOC file {os.path.basename(file_path)}: {str(e)}"
        
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

