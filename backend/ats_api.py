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
    Hybrid Boolean + Semantic Resume Search using Pinecone.
    
    The system allows searching by any text (e.g., location, candidate name, skills, 
    or keywords found in resumes) and retrieves related candidates using Pinecone's 
    vector similarity search.
    
    Search Coverage:
    - Semantic Search: Searches full resume text content via embeddings (understands meaning)
    - Boolean Filtering: Searches metadata fields (skills, location, name, company, domain, education, summary)
    
    Query Types Supported:
    - Simple queries: "python", "Bangalore", "John Doe"
    - Boolean queries: "python AND java", "Bangalore AND AWS"
    - Complex queries: ("Product Owner" OR "Product Manager") AND "Business" AND "Analyst"
    - Location queries: "Portland" OR "Oregon" (searches current_location field)
    - Name queries: "John Smith" (searches name field)
    - Skill queries: "Python" AND "Django" (searches primary_skills, secondary_skills)
    - Company queries: "Microsoft" OR "Google" (searches current_company field)
    
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
        use_boolean_search = data.get('use_boolean_search', True)  # Enable Boolean by default
        use_llm_refinement = data.get('use_llm_refinement', False)
        llm_window = data.get('llm_window_size', LLM_REFINEMENT_WINDOW)
        
        logger.info(f"Processing hybrid search query: {user_query}")
        logger.info(f"Applied filters: {filters}")
        logger.info(f"Boolean search enabled: {use_boolean_search}")
        
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
        
        # === Step 1: Parse Boolean Query (if enabled) ===
        parsed_query = None
        if use_boolean_search:
            # Check for explicit boolean operators (AND, OR, parentheses)
            if ' AND ' in user_query.upper() or ' OR ' in user_query.upper() or '(' in user_query:
                parsed_query = parse_boolean_query(user_query)
                logger.info(f"Parsed Boolean query: {parsed_query}")
            # Check for comma-separated values (treat as AND conditions)
            elif ',' in user_query:
                parsed_query = parse_comma_separated_query(user_query)
                logger.info(f"Parsed comma-separated query: {parsed_query}")
            else:
                # Simple query - treat as single term
                parsed_query = {'and_terms': [[user_query]]}
                logger.info(f"Simple query detected, treating as single term")
        else:
            # Boolean search disabled - treat as single term
            parsed_query = {'and_terms': [[user_query]]}
            logger.info(f"Boolean search disabled, treating as single term")
        
        # === Step 2: Generate Embedding Once ===
        query_embedding = embedding_service.generate_embedding(user_query)
        
        # === Layer 1: SQL metadata filtering ===
        structured_filters = build_structured_filters(filters, user_query)
        sql_limit = data.get('sql_limit', SQL_LAYER_LIMIT)
        with create_ats_database() as db:
            sql_candidates = db.filter_candidates(structured_filters, limit=sql_limit)
        
        if not sql_candidates:
            # SQL returned 0 results - run two-step search directly (skip Pinecone)
            logger.info(f"SQL filter returned 0 results, running two-step search directly...")
            
            # Mapping from query to score column
            query_to_score_column = {
                "java": "java_score",
                ".net": "dotnet_score",
                "dotnet": "dotnet_score",
                "python": "python_score",
                "javascript": "javascript_score",
                "js": "javascript_score",
                "full stack": "fullstack_score",
                "fullstack": "fullstack_score",
                "devops": "devops_score",
                "data engineering": "data_engineering_score",
                "data science": "data_science_score",
                "testing": "testing_qa_score",
                "qa": "testing_qa_score",
                "sap": "sap_score",
                "erp": "erp_score",
                "cloud": "cloud_infra_score",
                "infra": "cloud_infra_score",
                "bi": "business_intelligence_score",
                "business intelligence": "business_intelligence_score",
                "power platform": "microsoft_power_platform_score",
                "rpa": "rpa_score",
                "cyber security": "cyber_security_score",
                "security": "cyber_security_score",
                "mobile": "mobile_development_score",
                "salesforce": "salesforce_score",
                "low code": "low_code_no_code_score",
                "no code": "low_code_no_code_score",
                "database": "database_score",
                "sql": "database_score",
                "integration": "integration_apis_score",
                "api": "integration_apis_score",
                "ui/ux": "ui_ux_score",
                "ux": "ui_ux_score",
                "support": "support_score",
                "business development": "business_development_score",
            }
            
            # Determine score column based on query
            query_lower = user_query.lower()
            score_column = "python_score"  # Default
            for key, col in query_to_score_column.items():
                if key in query_lower or query_lower in key:
                    score_column = col
                    break
            
            logger.info(f"Two-step search using score column: {score_column}")
            
            two_step_results = []
            two_step_stats = {'step1_count': 0, 'step2_count': 0}
            
            with create_ats_database() as db:
                # Step 1: Search by skill only
                step1_results = db.search_by_skill(user_query, limit=100)
                two_step_stats['step1_count'] = len(step1_results)
                logger.info(f"Two-step Step 1: Found {len(step1_results)} candidates")
                
                # Step 2: Search by score
                step2_results = db.search_by_skill_with_score(user_query, score_column=score_column, limit=100)
                two_step_stats['step2_count'] = len(step2_results)
                logger.info(f"Two-step Step 2: Found {len(step2_results)} candidates")
                
                # Helper function to build candidate object
                def build_candidate_obj(r, match_score, search_source):
                    cid = r['candidate_id']
                    metadata = {
                        'candidate_id': cid,
                        'name': r.get('name', ''),
                        'email': r.get('email', ''),
                        'primary_skills': r.get('primary_skills', ''),
                        'secondary_skills': r.get('secondary_skills', ''),
                        'all_skills': r.get('all_skills', '') or r.get('primary_skills', ''),
                        'profile_type': r.get('profile_type', ''),
                        'total_experience': r.get('total_experience', 0),
                        'current_designation': r.get('current_designation', '') or 'Unknown',
                        'current_location': r.get('current_location', '') or 'Unknown',
                        'current_company': r.get('current_company', ''),
                        'domain': r.get('domain', ''),
                        'education': r.get('education', ''),
                        'resume_summary': r.get('resume_summary', ''),
                        'created_at': str(r.get('created_at', '')),
                        'source': search_source
                    }
                    return {
                        'candidate_id': cid,
                        'name': r.get('name', ''),
                        'email': r.get('email', ''),
                        'primary_skills': r.get('primary_skills', ''),
                        'profile_type': r.get('profile_type', ''),
                        'total_experience': r.get('total_experience', 0),
                        'current_designation': r.get('current_designation', ''),
                        'current_location': r.get('current_location', ''),
                        'current_company': r.get('current_company', ''),
                        'domain': r.get('domain', ''),
                        'education': r.get('education', ''),
                        'resume_summary': r.get('resume_summary', ''),
                        'match_score': round(match_score, 3),
                        'layer_scores': {'vector': None},
                        'metadata': metadata
                    }
                
                # Merge results (deduplicate)
                seen_ids = set()
                
                # Add Step 1 results
                for r in step1_results:
                    cid = r['candidate_id']
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        two_step_results.append(build_candidate_obj(r, 0.5, 'two_step_primary'))
                
                # Add Step 2 results
                for r in step2_results:
                    cid = r['candidate_id']
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        skill_score = r.get('skill_match_score', 0)
                        match_score = min(0.8, (skill_score / 100)) if skill_score else 0.4
                        two_step_results.append(build_candidate_obj(r, match_score, 'two_step_fallback'))
            
            # Sort by match_score DESC
            two_step_results.sort(key=lambda x: (x.get('match_score', 0), x.get('total_experience', 0)), reverse=True)
            
            return jsonify({
                'message': f'Found {len(two_step_results)} candidates via two-step search (SQL filter returned 0)',
                'query': user_query,
                'search_results': two_step_results,
                'total_matches': len(two_step_results),
                'layer_counts': {
                    'sql_filtered': 0,
                    'vector_considered': 0,
                    'boolean_passed': 0,
                    'llm_evaluated': 0,
                    'two_step_step1': two_step_stats['step1_count'],
                    'two_step_step2': two_step_stats['step2_count'],
                    'two_step_new': len(two_step_results)
                },
                'profile_type_filter': structured_filters.get('profile_type'),
                'two_step_search_applied': True,
                'processing_time_ms': int((time.time() - start_time) * 1000),
                'timestamp': datetime.now().isoformat()
            }), 200
        
        candidate_map = {
            record['candidate_id']: record
            for record in sql_candidates
            if record.get('candidate_id') is not None
        }
        candidate_ids = list(candidate_map.keys())
        if not candidate_ids:
            return jsonify({
                'message': 'No matching resumes found after SQL filtering',
                'query': user_query,
                'search_results': [],
                'total_matches': 0,
                'layer_counts': {
                    'sql_filtered': 0,
                    'vector_considered': 0,
                    'boolean_passed': 0,
                    'llm_evaluated': 0
                },
                'profile_type_filter': structured_filters.get('profile_type'),
                'processing_time_ms': int((time.time() - start_time) * 1000),
                'timestamp': datetime.now().isoformat()
            }), 200
        
        # === Layer 2: Vector search constrained to SQL subset ===
        layer2_matches = {}
        chunk_size = int(data.get('vector_chunk_size', VECTOR_CHUNK_SIZE))
        chunk_size = max(chunk_size, 50)
        vector_top_k = max(top_k * 3, 60)
        # Don't use profile_type filter in Pinecone since we already filtered in SQL
        # Pinecone stores profile_type as comma-separated string which doesn't work well with $in
        # SQL filtering is more reliable for profile_type matching
        pinecone_filters_for_vector = structured_filters.copy()
        if 'profile_type' in pinecone_filters_for_vector:
            del pinecone_filters_for_vector['profile_type']
        pinecone_metadata_filter = build_pinecone_metadata_filter(pinecone_filters_for_vector)
        
        for chunk in chunk_list(candidate_ids, chunk_size):
            if not chunk:
                continue
            base_filter = {'candidate_id': {'$in': chunk}}
            chunk_filter = merge_pinecone_filters(base_filter, pinecone_metadata_filter)
            chunk_top_k = min(len(chunk), vector_top_k)
            results = pinecone_manager.query_vectors(
                query_vector=query_embedding,
                top_k=chunk_top_k,
                include_metadata=True,
                filter=chunk_filter
            )
            for match in results.matches:
                cid = match.metadata.get('candidate_id')
                if cid not in candidate_map:
                    continue
                existing = layer2_matches.get(cid)
                if not existing or match.score > existing['score']:
                    layer2_matches[cid] = {
                        'score': match.score,
                        'metadata': match.metadata
                    }
        
        if not layer2_matches:
            # Fallback: Return SQL-filtered candidates when vector search finds no matches
            # This handles cases where candidates exist in SQL but:
            # 1. Don't have embeddings in Pinecone
            # 2. Pinecone metadata filter is too restrictive (e.g., comma-separated profile_type)
            # 3. Vector similarity is below threshold
            logger.warning(f"Vector search found 0 matches for {len(candidate_ids)} SQL-filtered candidates. Returning SQL results as fallback.")
            fallback_results = []
            
            # Calculate match scores based on profile_type and skills overlap
            search_profile_types = structured_filters.get('profile_type', [])
            search_query_lower = user_query.lower() if user_query else ""
            
            for record in sql_candidates[:top_k]:
                candidate_profile_types = record.get('profile_type', '')
                candidate_skills = (record.get('primary_skills', '') or '').lower()
                
                # Calculate match score based on profile_type overlap
                base_score = 0.5
                profile_type_bonus = 0.0
                profile_type_penalty = 0.0
                skills_bonus = 0.0
                
                # Profile type matching bonus/penalty
                if candidate_profile_types and search_profile_types:
                    # Split candidate's profile types (handle both comma and comma+space formats)
                    candidate_pts = [pt.strip() for pt in candidate_profile_types.replace(', ', ',').split(',')]
                    has_match = False
                    for search_pt in search_profile_types:
                        if search_pt in candidate_pts:
                            # Exact profile type match
                            profile_type_bonus = 0.3
                            has_match = True
                            break
                        elif any(search_pt in pt or pt in search_pt for pt in candidate_pts):
                            # Partial match (e.g., "Microsoft Power Platform" in "Microsoft Power Platform,Integration / APIs")
                            profile_type_bonus = 0.2
                            has_match = True
                            break
                    
                    # Penalty if profile_type filter exists but candidate doesn't match
                    if not has_match and search_profile_types:
                        profile_type_penalty = -0.2  # Reduce score for non-matching profile types
                
                # Skills matching bonus (check if query terms appear in skills)
                if search_query_lower and candidate_skills:
                    query_terms = search_query_lower.split()
                    matched_terms = sum(1 for term in query_terms if term in candidate_skills and len(term) > 2)
                    if matched_terms > 0:
                        skills_bonus = min(0.2, matched_terms * 0.05)
                
                match_score = max(0.0, min(1.0, base_score + profile_type_bonus + profile_type_penalty + skills_bonus))
                
                # Build metadata object (same structure as Pinecone results)
                cid = record.get('candidate_id')
                metadata = {
                    'candidate_id': cid,
                    'name': record.get('name', ''),
                    'email': record.get('email', ''),
                    'primary_skills': record.get('primary_skills', ''),
                    'secondary_skills': record.get('secondary_skills', ''),
                    'all_skills': record.get('all_skills', '') or record.get('primary_skills', ''),
                    'profile_type': record.get('profile_type', ''),
                    'total_experience': record.get('total_experience', 0),
                    'current_designation': record.get('current_designation', '') or 'Unknown',
                    'current_location': record.get('current_location', '') or 'Unknown',
                    'current_company': record.get('current_company', ''),
                    'domain': record.get('domain', ''),
                    'education': record.get('education', ''),
                    'resume_summary': record.get('resume_summary', ''),
                    'created_at': str(record.get('created_at', '')),
                    'source': 'sql_fallback'
                }
                
                fallback_results.append({
                    'candidate_id': cid,
                    'name': record.get('name', ''),
                    'email': record.get('email', ''),
                    'primary_skills': record.get('primary_skills', ''),
                    'profile_type': record.get('profile_type', ''),
                    'total_experience': record.get('total_experience', 0.0),
                    'current_designation': record.get('current_designation', ''),
                    'current_location': record.get('current_location', ''),
                    'current_company': record.get('current_company', ''),
                    'domain': record.get('domain', ''),
                    'education': record.get('education', ''),
                    'resume_summary': record.get('resume_summary', ''),
                    'match_score': round(match_score, 3),
                    'layer_scores': {'vector': None},
                    'metadata': metadata
                })
            
            # Sort by match_score descending
            fallback_results.sort(key=lambda x: x['match_score'], reverse=True)
            
            return jsonify({
                'message': f'Found {len(fallback_results)} candidates via SQL filtering (vector search found no matches)',
                'query': user_query,
                'search_results': fallback_results,
                'total_matches': len(fallback_results),
                'layer_counts': {
                    'sql_filtered': len(candidate_ids),
                    'vector_considered': 0,
                    'boolean_passed': 0,
                    'llm_evaluated': 0
                },
                'profile_type_filter': structured_filters.get('profile_type'),
                'processing_time_ms': int((time.time() - start_time) * 1000),
                'timestamp': datetime.now().isoformat()
            }), 200
        
        vector_ranked = []
        search_profile_types = structured_filters.get('profile_type', [])
        
        for cid, match_data in layer2_matches.items():
            db_record = candidate_map[cid]
            pinecone_meta = match_data.get('metadata') or {}
            combined_metadata = dict(pinecone_meta)
            for field in [
                'primary_skills', 'secondary_skills', 'all_skills',
                'current_location', 'current_company', 'current_designation',
                'domain', 'education', 'resume_summary', 'profile_type'
            ]:
                if db_record.get(field):
                    combined_metadata[field] = db_record[field]
            
            # Calculate base match score from vector similarity
            base_score = match_data['score']
            
            # Add profile_type matching bonus
            profile_type_bonus = 0.0
            candidate_profile_type = db_record.get('profile_type') or pinecone_meta.get('profile_type', '')
            if candidate_profile_type and search_profile_types:
                # Split candidate's profile types (handle both comma and comma+space formats)
                candidate_pts = [pt.strip() for pt in candidate_profile_type.replace(', ', ',').split(',')]
                for search_pt in search_profile_types:
                    if search_pt in candidate_pts:
                        # Exact profile type match - boost score
                        profile_type_bonus = 0.1
                        break
                    elif any(search_pt in pt or pt in search_pt for pt in candidate_pts):
                        # Partial match bonus
                        profile_type_bonus = 0.05
                        break
            
            # Final score with profile_type bonus (capped at 1.0)
            final_score = min(1.0, base_score + profile_type_bonus)
            
            candidate_info = {
                'candidate_id': cid,
                'name': db_record.get('name') or pinecone_meta.get('name'),
                'email': db_record.get('email') or pinecone_meta.get('email'),
                'match_score': round(final_score, 3),
                'profile_type': candidate_profile_type,
                'primary_skills': db_record.get('primary_skills'),
                'total_experience': db_record.get('total_experience'),
                'domain': db_record.get('domain'),
                'education': db_record.get('education'),
                'current_location': db_record.get('current_location'),
                'current_company': db_record.get('current_company'),
                'current_designation': db_record.get('current_designation'),
                'resume_summary': db_record.get('resume_summary'),
                'layer_scores': {'vector': match_data['score']},
                'metadata': combined_metadata
            }
            vector_ranked.append(candidate_info)
        
        vector_ranked.sort(key=lambda c: c['match_score'], reverse=True)
        
        # === Layer 2b: Optional Boolean filtering ===
        if use_boolean_search and parsed_query:
            boolean_filtered = []
            for candidate in vector_ranked:
                candidate_text = build_searchable_text(candidate['metadata'])
                if matches_boolean_query(candidate_text, parsed_query):
                    boolean_filtered.append(candidate)
        else:
            boolean_filtered = vector_ranked
        
        if not boolean_filtered:
            return jsonify({
                'message': 'Boolean filter removed all semantic matches',
                'query': user_query,
                'search_results': [],
                'total_matches': 0,
                'layer_counts': {
                    'sql_filtered': len(candidate_ids),
                    'vector_considered': len(vector_ranked),
                    'boolean_passed': 0,
                    'llm_evaluated': 0
                },
                'profile_type_filter': structured_filters.get('profile_type'),
                'boolean_filter_applied': True,
                'processing_time_ms': int((time.time() - start_time) * 1000),
                'timestamp': datetime.now().isoformat()
            }), 200
        
        # === Layer 3: Optional LLM refinement ===
        llm_applied = False
        final_candidates = boolean_filtered
        if use_llm_refinement and llm_refinement_service.available:
            job_context = data.get('job_description') or user_query
            final_candidates = llm_refinement_service.rerank_candidates(
                job_context,
                boolean_filtered,
                top_n=min(llm_window, len(boolean_filtered))
            )
            llm_applied = True
        elif use_llm_refinement and not llm_refinement_service.available:
            logger.warning("LLM refinement requested but service is not available.")
        
        response_candidates = final_candidates[:top_k]
        
        # === Layer 4: Two-Step Skill Search (Auto-fallback if results < 20) ===
        two_step_applied = False
        two_step_stats = {'step1_count': 0, 'step2_count': 0, 'new_from_two_step': 0}
        
        if len(response_candidates) < 20:
            logger.info(f"Results ({len(response_candidates)}) < 20, running two-step skill search fallback...")
            
            # Mapping from query to score column
            query_to_score_column = {
                "java": "java_score",
                ".net": "dotnet_score",
                "dotnet": "dotnet_score",
                "python": "python_score",
                "javascript": "javascript_score",
                "js": "javascript_score",
                "full stack": "fullstack_score",
                "fullstack": "fullstack_score",
                "devops": "devops_score",
                "data engineering": "data_engineering_score",
                "data science": "data_science_score",
                "testing": "testing_qa_score",
                "qa": "testing_qa_score",
                "sap": "sap_score",
                "erp": "erp_score",
                "cloud": "cloud_infra_score",
                "infra": "cloud_infra_score",
                "bi": "business_intelligence_score",
                "business intelligence": "business_intelligence_score",
                "power platform": "microsoft_power_platform_score",
                "rpa": "rpa_score",
                "cyber security": "cyber_security_score",
                "security": "cyber_security_score",
                "mobile": "mobile_development_score",
                "salesforce": "salesforce_score",
                "low code": "low_code_no_code_score",
                "no code": "low_code_no_code_score",
                "database": "database_score",
                "sql": "database_score",
                "integration": "integration_apis_score",
                "api": "integration_apis_score",
                "ui/ux": "ui_ux_score",
                "ux": "ui_ux_score",
                "support": "support_score",
                "business development": "business_development_score",
            }
            
            # Determine score column based on query
            query_lower = user_query.lower()
            score_column = "python_score"  # Default
            
            # Find best matching score column
            for key, col in query_to_score_column.items():
                if key in query_lower or query_lower in key:
                    score_column = col
                    break
            
            logger.info(f"Two-step search using score column: {score_column}")
            
            # Track existing candidate IDs
            existing_ids = {c.get('candidate_id') for c in response_candidates if c.get('candidate_id')}
            
            with create_ats_database() as db:
                # Step 1: Primary Search (skillset only)
                step1_results = db.search_by_skill(user_query, limit=100)
                two_step_stats['step1_count'] = len(step1_results)
                logger.info(f"Two-step Step 1: Found {len(step1_results)} candidates")
                
                # Step 2: Fallback Search (score > 0 AND skillset LIKE query)
                step2_results = db.search_by_skill_with_score(user_query, score_column=score_column, limit=100)
                two_step_stats['step2_count'] = len(step2_results)
                logger.info(f"Two-step Step 2: Found {len(step2_results)} candidates")
                
                # Merge new candidates from two-step search
                new_candidates = []
                
                # Helper function to build Pinecone-style candidate object
                def build_candidate_object(r, match_score, search_source):
                    """Build candidate object with same structure as Pinecone results."""
                    cid = r['candidate_id']
                    
                    # Build metadata object (same as Pinecone)
                    metadata = {
                        'candidate_id': cid,
                        'name': r.get('name', ''),
                        'email': r.get('email', ''),
                        'primary_skills': r.get('primary_skills', ''),
                        'secondary_skills': r.get('secondary_skills', ''),
                        'all_skills': r.get('all_skills', '') or r.get('primary_skills', ''),
                        'profile_type': r.get('profile_type', ''),
                        'total_experience': r.get('total_experience', 0),
                        'current_designation': r.get('current_designation', '') or 'Unknown',
                        'current_location': r.get('current_location', '') or 'Unknown',
                        'current_company': r.get('current_company', ''),
                        'domain': r.get('domain', ''),
                        'education': r.get('education', ''),
                        'resume_summary': r.get('resume_summary', ''),
                        'created_at': str(r.get('created_at', '')),
                        'source': search_source
                    }
                    
                    return {
                        'candidate_id': cid,
                        'name': r.get('name', ''),
                        'email': r.get('email', ''),
                        'primary_skills': r.get('primary_skills', ''),
                        'profile_type': r.get('profile_type', ''),
                        'total_experience': r.get('total_experience', 0),
                        'current_designation': r.get('current_designation', ''),
                        'current_location': r.get('current_location', ''),
                        'current_company': r.get('current_company', ''),
                        'domain': r.get('domain', ''),
                        'education': r.get('education', ''),
                        'resume_summary': r.get('resume_summary', ''),
                        'match_score': round(match_score, 3),
                        'layer_scores': {
                            'vector': None
                        },
                        'metadata': metadata
                    }
                
                # Add from Step 1 (not already in results)
                for r in step1_results:
                    cid = r['candidate_id']
                    if cid not in existing_ids:
                        existing_ids.add(cid)
                        new_candidates.append(build_candidate_object(r, 0.5, 'two_step_primary'))
                
                # Add from Step 2 (not already in results)
                for r in step2_results:
                    cid = r['candidate_id']
                    if cid not in existing_ids:
                        existing_ids.add(cid)
                        skill_score = r.get('skill_match_score', 0)
                        match_score = min(0.8, (skill_score / 100)) if skill_score else 0.4
                        new_candidates.append(build_candidate_object(r, match_score, 'two_step_fallback'))
                
                two_step_stats['new_from_two_step'] = len(new_candidates)
                
                if new_candidates:
                    two_step_applied = True
                    # Sort new candidates by skill_match_score DESC
                    new_candidates.sort(key=lambda x: (x.get('skill_match_score') or 0, x.get('total_experience') or 0), reverse=True)
                    # Append to response_candidates
                    response_candidates.extend(new_candidates)
                    logger.info(f"Added {len(new_candidates)} new candidates from two-step search")
        
        processing_time = int((time.time() - start_time) * 1000)
        layer_counts = {
            'sql_filtered': len(candidate_ids),
            'vector_considered': len(vector_ranked),
            'boolean_passed': len(boolean_filtered),
            'llm_evaluated': len(final_candidates) if llm_applied else 0,
            'two_step_step1': two_step_stats['step1_count'],
            'two_step_step2': two_step_stats['step2_count'],
            'two_step_new': two_step_stats['new_from_two_step']
        }
        
        message = 'Hybrid SQL + vector search completed'
        if llm_applied:
            message += ' with LLM refinement'
        elif use_llm_refinement:
            message += ' (LLM refinement unavailable)'
        if two_step_applied:
            message += f' + two-step fallback (+{two_step_stats["new_from_two_step"]} candidates)'
        
        return jsonify({
            'message': message,
            'query': user_query,
            'search_results': response_candidates,
            'total_matches': len(response_candidates),
            'layer_counts': layer_counts,
            'total_before_boolean_filter': len(vector_ranked) if use_boolean_search else None,
            'boolean_filter_applied': use_boolean_search and parsed_query is not None,
            'llm_refinement_applied': llm_applied,
            'two_step_search_applied': two_step_applied,
            'profile_type_filter': structured_filters.get('profile_type'),
            'processing_time_ms': processing_time,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/searchResumesBySkill', methods=['POST'])
def search_resumes_by_skill():
    """
    Two-step skill-based resume search.
    
    Step 1 (Primary Search):
    - Search candidates where profile_type matches/relates to query AND skillset contains query
    - If result count >= 20, return immediately
    
    Step 2 (Fallback Search):
    - If Step 1 < 20 results, search candidate_profile_scores WHERE score > 0 AND skillset LIKE query
    - Combine results (deduplicated), sorted by skill_match_score DESC
    
    Input: {"query": "<skill_name>"}
    Output: Combined candidates from both steps (deduplicated)
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if 'query' not in data:
            return jsonify({'error': 'Missing query field'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        logger.info(f"Skill-based search for: {query}")
        start_time = time.time()
        
        # Mapping from query to score column
        query_to_score_column = {
            "java": "java_score",
            ".net": "dotnet_score",
            "dotnet": "dotnet_score",
            "python": "python_score",
            "javascript": "javascript_score",
            "js": "javascript_score",
            "full stack": "fullstack_score",
            "fullstack": "fullstack_score",
            "devops": "devops_score",
            "data engineering": "data_engineering_score",
            "data science": "data_science_score",
            "testing": "testing_qa_score",
            "qa": "testing_qa_score",
            "sap": "sap_score",
            "erp": "erp_score",
            "cloud": "cloud_infra_score",
            "infra": "cloud_infra_score",
            "bi": "business_intelligence_score",
            "business intelligence": "business_intelligence_score",
            "power platform": "microsoft_power_platform_score",
            "rpa": "rpa_score",
            "cyber security": "cyber_security_score",
            "security": "cyber_security_score",
            "mobile": "mobile_development_score",
            "salesforce": "salesforce_score",
            "low code": "low_code_no_code_score",
            "no code": "low_code_no_code_score",
            "database": "database_score",
            "sql": "database_score",
            "integration": "integration_apis_score",
            "api": "integration_apis_score",
            "ui/ux": "ui_ux_score",
            "ux": "ui_ux_score",
            "support": "support_score",
            "business development": "business_development_score",
        }
        
        # Determine score column based on query
        query_lower = query.lower()
        score_column = query_to_score_column.get(query_lower, "python_score")  # Default to python_score
        
        # Find best matching score column if exact match not found
        if query_lower not in query_to_score_column:
            for key, col in query_to_score_column.items():
                if key in query_lower or query_lower in key:
                    score_column = col
                    break
        
        logger.info(f"Using score column: {score_column}")
        
        with create_ats_database() as db:
            # Step 1: Primary Search (profile_type + skillset)
            step1_results = db.search_by_skill(query, limit=100)
            step1_count = len(step1_results)
            logger.info(f"Step 1 (Primary Search): Found {step1_count} candidates")
            
            # Track candidate IDs from Step 1
            step1_ids = {r['candidate_id'] for r in step1_results}
            
            # If Step 1 has >= 20 results, return immediately
            if step1_count >= 20:
                # Format results
                results = []
                for r in step1_results:
                    results.append({
                        'candidate_id': r['candidate_id'],
                        'name': r['name'],
                        'email': r['email'],
                        'phone': r.get('phone'),
                        'total_experience': r.get('total_experience', 0),
                        'primary_skills': r.get('primary_skills', ''),
                        'profile_type': r.get('profile_type', ''),
                        'domain': r.get('domain', ''),
                        'education': r.get('education', ''),
                        'current_location': r.get('current_location', ''),
                        'current_company': r.get('current_company', ''),
                        'current_designation': r.get('current_designation', ''),
                        'skill_match_score': None,  # Not from score table
                        'search_step': 'primary'
                    })
                
                return jsonify({
                    'message': f'Found {len(results)} candidates via primary search',
                    'query': query,
                    'search_results': results,
                    'total_matches': len(results),
                    'step1_count': step1_count,
                    'step2_count': 0,
                    'search_type': 'primary_only',
                    'processing_time_ms': int((time.time() - start_time) * 1000),
                    'timestamp': datetime.now().isoformat()
                }), 200
            
            # Step 2: Fallback Search (score > 0 AND skillset LIKE query)
            step2_results = db.search_by_skill_with_score(query, score_column=score_column, limit=100)
            step2_count = len(step2_results)
            logger.info(f"Step 2 (Fallback Search): Found {step2_count} candidates")
            
            # Combine results (deduplicate by candidate_id)
            combined_results = []
            seen_ids = set()
            
            # Add Step 1 results first (they matched profile_type)
            for r in step1_results:
                cid = r['candidate_id']
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    combined_results.append({
                        'candidate_id': cid,
                        'name': r['name'],
                        'email': r['email'],
                        'phone': r.get('phone'),
                        'total_experience': r.get('total_experience', 0),
                        'primary_skills': r.get('primary_skills', ''),
                        'profile_type': r.get('profile_type', ''),
                        'domain': r.get('domain', ''),
                        'education': r.get('education', ''),
                        'current_location': r.get('current_location', ''),
                        'current_company': r.get('current_company', ''),
                        'current_designation': r.get('current_designation', ''),
                        'skill_match_score': None,
                        'search_step': 'primary'
                    })
            
            # Add Step 2 results (deduplicated)
            for r in step2_results:
                cid = r['candidate_id']
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    combined_results.append({
                        'candidate_id': cid,
                        'name': r['name'],
                        'email': r['email'],
                        'phone': r.get('phone'),
                        'total_experience': r.get('total_experience', 0),
                        'primary_skills': r.get('primary_skills', ''),
                        'profile_type': r.get('profile_type', ''),
                        'domain': r.get('domain', ''),
                        'education': r.get('education', ''),
                        'current_location': r.get('current_location', ''),
                        'current_company': r.get('current_company', ''),
                        'current_designation': r.get('current_designation', ''),
                        'skill_match_score': r.get('skill_match_score', 0),
                        'search_step': 'fallback'
                    })
            
            # Sort by skill_match_score DESC (None values go last), then by experience
            combined_results.sort(
                key=lambda x: (x['skill_match_score'] or 0, x['total_experience'] or 0),
                reverse=True
            )
            
            return jsonify({
                'message': f'Found {len(combined_results)} candidates (Step1: {step1_count}, Step2: {step2_count - len(step1_ids & {r["candidate_id"] for r in step2_results})} new)',
                'query': query,
                'search_results': combined_results,
                'total_matches': len(combined_results),
                'step1_count': step1_count,
                'step2_count': step2_count,
                'search_type': 'combined',
                'score_column_used': score_column,
                'processing_time_ms': int((time.time() - start_time) * 1000),
                'timestamp': datetime.now().isoformat()
            }), 200
            
    except Exception as e:
        logger.error(f"Error in skill-based search: {e}")
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

