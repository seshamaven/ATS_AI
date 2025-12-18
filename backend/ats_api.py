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
from collections import Counter
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
    canonicalize_profile_type,
)
from skill_extractor import TECH_SKILLS
from role_extract import detect_role_subrole, detect_role_only, detect_subrole_only

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
    Service to generate embeddings using Ollama, Azure OpenAI, or OpenAI.
    
    STATELESS DESIGN: This class is stateless and thread-safe.
    It holds only immutable configuration (API keys, model names).
    Each generate_embedding() call is independent and doesn't cache results.
    Safe to use as a global singleton across concurrent requests.
    """
    
    def __init__(self):
        # Initialize embedding service based on configuration
        # NOTE: Only immutable configuration stored here (safe for global instance)
        if ATSConfig.USE_OLLAMA:
            from ollama_client import get_ollama_client
            logger.info("=" * 60)
            logger.info("EMBEDDINGS SERVICE: Using OLLAMA")
            logger.info(f"  Model: {ATSConfig.OLLAMA_MODEL}")
            logger.info(f"  Base URL: {ATSConfig.OLLAMA_BASE_URL}")
            logger.info("=" * 60)
            self.ollama_client = get_ollama_client(
                base_url=ATSConfig.OLLAMA_BASE_URL,
                model=ATSConfig.OLLAMA_MODEL
            )
            self.client = None
            self.model = ATSConfig.OLLAMA_MODEL
            self.use_ollama = True
        elif ATSConfig.AZURE_OPENAI_ENDPOINT:
            logger.info("=" * 60)
            logger.info("EMBEDDINGS SERVICE: Using AZURE OPENAI")
            logger.info(f"  Model: {ATSConfig.AZURE_OPENAI_EMBEDDING_DEPLOYMENT}")
            logger.info(f"  Endpoint: {ATSConfig.AZURE_OPENAI_ENDPOINT}")
            logger.info("=" * 60)
            self.client = AzureOpenAI(
                api_key=ATSConfig.AZURE_OPENAI_API_KEY,
                api_version=ATSConfig.AZURE_OPENAI_API_VERSION,
                azure_endpoint=ATSConfig.AZURE_OPENAI_ENDPOINT
            )
            self.model = ATSConfig.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
            self.use_ollama = False
        else:
            logger.info("=" * 60)
            logger.info("EMBEDDINGS SERVICE: Using OPENAI")
            logger.info(f"  Model: {ATSConfig.OPENAI_EMBEDDING_MODEL}")
            logger.info("=" * 60)
            self.client = OpenAI(api_key=ATSConfig.OPENAI_API_KEY)
            self.model = ATSConfig.OPENAI_EMBEDDING_MODEL
            self.use_ollama = False
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for given text.
        
        Args:
            text: Input text
        
        Returns:
            1536-dimension embedding vector
        """
        try:
            # Truncate text if too long
            if len(text) > 30000:
                text = text[:30000]
            
            if self.use_ollama:
                logger.debug(f"Generating embedding using OLLAMA (model: {self.model})")
                embedding = self.ollama_client.embeddings(text)
                # Ensure embedding is 1536 dimensions (pad or truncate if needed)
                if len(embedding) < 1536:
                    embedding.extend([0.0] * (1536 - len(embedding)))
                elif len(embedding) > 1536:
                    embedding = embedding[:1536]
                logger.info(f"Generated embedding using OLLAMA - {len(embedding)} dimensions")
                return embedding
            else:
                service_name = "AZURE OPENAI" if ATSConfig.AZURE_OPENAI_ENDPOINT else "OPENAI"
                logger.debug(f"Generating embedding using {service_name} (model: {self.model})")
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
                embedding = response.data[0].embedding
                logger.info(f"Generated embedding using {service_name} - {len(embedding)} dimensions")
                return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise


# Constants controlling layered search
SQL_LAYER_LIMIT = int(os.getenv('ATS_SQL_LAYER_LIMIT', '10000'))
VECTOR_CHUNK_SIZE = int(os.getenv('ATS_VECTOR_CHUNK_SIZE', '200'))
LLM_REFINEMENT_WINDOW = int(os.getenv('ATS_LLM_REFINEMENT_WINDOW', '60'))


class LLMRefinementService:
    """Optional third-layer refinement using Ollama, Azure OpenAI, or OpenAI chat completions."""
    
    def __init__(self):
        self.client = None
        self.ollama_client = None
        self.model = None
        self.use_ollama = False
        try:
            if ATSConfig.USE_OLLAMA:
                from ollama_client import get_ollama_openai_client
                logger.info("=" * 60)
                logger.info("LLM REFINEMENT SERVICE: Using OLLAMA")
                logger.info(f"  Model: {ATSConfig.OLLAMA_MODEL}")
                logger.info(f"  Base URL: {ATSConfig.OLLAMA_BASE_URL}")
                logger.info("=" * 60)
                self.client = get_ollama_openai_client(
                    base_url=ATSConfig.OLLAMA_BASE_URL,
                    model=ATSConfig.OLLAMA_MODEL
                )
                self.model = ATSConfig.OLLAMA_MODEL
                self.use_ollama = True
            elif ATSConfig.AZURE_OPENAI_ENDPOINT:
                logger.info("=" * 60)
                logger.info("LLM REFINEMENT SERVICE: Using AZURE OPENAI")
                logger.info(f"  Model: {ATSConfig.AZURE_OPENAI_MODEL}")
                logger.info(f"  Endpoint: {ATSConfig.AZURE_OPENAI_ENDPOINT}")
                logger.info("=" * 60)
                self.client = AzureOpenAI(
                    api_key=ATSConfig.AZURE_OPENAI_API_KEY,
                    api_version=ATSConfig.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=ATSConfig.AZURE_OPENAI_ENDPOINT
                )
                self.model = ATSConfig.AZURE_OPENAI_MODEL
            elif ATSConfig.OPENAI_API_KEY:
                logger.info("=" * 60)
                logger.info("LLM REFINEMENT SERVICE: Using OPENAI")
                logger.info(f"  Model: {ATSConfig.OPENAI_MODEL}")
                logger.info("=" * 60)
                self.client = OpenAI(api_key=ATSConfig.OPENAI_API_KEY)
                self.model = ATSConfig.OPENAI_MODEL
        except Exception as exc:
            logger.warning(f"LLM refinement disabled: {exc}")
            self.client = None
            self.ollama_client = None
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
            
            service_type = "OLLAMA" if self.use_ollama else ("AZURE OPENAI" if ATSConfig.AZURE_OPENAI_ENDPOINT else "OPENAI")
            logger.info(f"Reranking candidates using {service_type} (model: {self.model})")
            
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
resume_parser = ResumeParser(use_ai_extraction=False)  # Disabled: No OpenAI/Ollama for AI extraction
llm_refinement_service = LLMRefinementService()


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Yield evenly sized chunks from a list."""
    return [items[i:i + chunk_size] for i in range(0, len(items), max(chunk_size, 1))]


def get_index_name_from_profile_type(profile_type: str) -> str:
    """
    Map profile_type to Pinecone index name.
    
    Args:
        profile_type: Profile type string (e.g., 'Java', 'Python', 'Business Analyst')
    
    Returns:
        Pinecone index name (e.g., 'java', 'python', 'dotnet', 'business-analyst', 'others')
    """
    if not profile_type:
        return 'others'
    
    # Normalize profile type
    profile_lower = profile_type.lower().strip()
    
    # Handle multi-profile types (comma-separated)
    if ',' in profile_lower:
        profile_lower = profile_lower.split(',')[0].strip()
    
    # Map to index names (matching the user's existing indexes: java, dotnet, python, business analyst, others)
    # Note: Pinecone index names are case-insensitive and may have spaces or hyphens
    index_mapping = {
        'java': 'java',
        'python': 'python',
        '.net': 'dotnet',
        'dotnet': 'dotnet',
        'net': 'dotnet',
        'c#': 'dotnet',
        'csharp': 'dotnet',
        'business analyst': 'business analyst',  # Try exact match first
        'business-analyst': 'business analyst',
        'ba': 'business analyst',
        'others': 'others',
        'other': 'others',
        'generalist': 'others',
        'general': 'others'
    }
    
    # Direct match
    if profile_lower in index_mapping:
        return index_mapping[profile_lower]
    
    # Partial match
    for key, index_name in index_mapping.items():
        if key in profile_lower or profile_lower in key:
            return index_name
    
    # Default to others
    return 'others'


def get_namespace_from_profile_type(profile_type: str) -> str:
    """
    Map profile_type to Pinecone namespace.
    
    Uses namespaces within a single index instead of separate indexes.
    This allows unlimited profile types without hitting the 5-index limit.
    
    Args:
        profile_type: Profile type string (e.g., 'Java', 'Python', 'Business Analyst')
    
    Returns:
        Normalized namespace name (e.g., 'java', 'python', 'business-analyst')
    """
    if not profile_type:
        return 'others'
    
    # Normalize profile type
    profile_lower = profile_type.lower().strip()
    
    # Handle multi-profile types (comma-separated)
    if ',' in profile_lower:
        # Use the first profile type
        profile_lower = profile_lower.split(',')[0].strip()
    
    # Map to namespace names
    namespace_mapping = {
        'java': 'java',
        'python': 'python',
        '.net': 'dotnet',
        'dotnet': 'dotnet',
        'net': 'dotnet',
        'c#': 'dotnet',
        'csharp': 'dotnet',
        'business analyst': 'business-analyst',
        'business-analyst': 'business-analyst',
        'ba': 'business-analyst',
        'project manager': 'project-manager',
        'project-manager': 'project-manager',
        'pm': 'project-manager',
        'sql': 'sql',
        'database': 'sql',
        'dba': 'sql',
        'others': 'others',
        'other': 'others',
        'generalist': 'others',
        'general': 'others'
    }
    
    # Direct match
    if profile_lower in namespace_mapping:
        return namespace_mapping[profile_lower]
    
    # Partial match
    for key, namespace in namespace_mapping.items():
        if key in profile_lower or profile_lower in key:
            return namespace
    
    # Normalize and return (replace spaces with hyphens, lowercase)
    normalized = profile_lower.replace(' ', '-').replace('.', '').replace('/', '-')
    normalized = ''.join(c if c.isalnum() or c == '-' else '' for c in normalized)
    while '--' in normalized:
        normalized = normalized.replace('--', '-')
    normalized = normalized.strip('-')
    
    return normalized if normalized else 'others'


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
            
            # Skip embedding generation - will be done later via /api/indexExistingResumes
            logger.info("Skipping embedding generation - will be indexed later via /api/indexExistingResumes")
            # Store in database (without embedding - stored in Pinecone only)
            try:
                with create_ats_database() as db:
                    # Ensure pinecone_indexed is set to False
                    parsed_data['pinecone_indexed'] = False
                    candidate_id = db.insert_resume(parsed_data)
                    
                    if not candidate_id:
                        error_details = db.last_error or "Unknown database error"
                        error_code = db.last_error_code
                        logger.error(f"insert_resume returned None. Error: {error_details} (Code: {error_code})")
                        
                        # Provide helpful error message based on error code
                        if error_code == 1054:
                            error_message = "Database schema is missing required columns (role_type, subrole_type). Please run the migration."
                        elif "Data too long" in error_details:
                            error_message = f"Data too long for a column: {error_details}"
                        else:
                            error_message = error_details
                        
                        return jsonify({
                            'error': 'Failed to store resume in database',
                            'details': error_message,
                            'error_code': error_code
                        }), 500
                    
                    # Calculate and store profile scores
                    try:
                        from profile_type_utils import (
                            get_all_profile_type_scores,
                            get_second_highest_profile_type,
                        )
                        profile_scores = get_all_profile_type_scores(
                            primary_skills=parsed_data.get('primary_skills', ''),
                            secondary_skills=parsed_data.get('secondary_skills', ''),
                            resume_text=parsed_data.get('resume_text', '')
                        )
                        
                        # Log profile scores before storing
                        sorted_scores = sorted(profile_scores.items(), key=lambda x: x[1], reverse=True)
                        non_zero_scores = [(pt, score) for pt, score in sorted_scores if score > 0]
                        logger.info(f"Calculated profile scores for candidate_id={candidate_id}: {non_zero_scores[:5]}")  # Top 5
                        
                        # Store profile scores and check return value
                        success = db.insert_or_update_profile_scores(candidate_id, profile_scores)
                        if success:
                            logger.info(f"Successfully stored profile scores for candidate_id={candidate_id}")
                        else:
                            logger.error(f"FAILED to store profile scores for candidate_id={candidate_id}. Check database logs for details.")
                            # Log the scores that should have been stored for debugging
                            logger.error(f"Profile scores that failed to store: {non_zero_scores}")
                        
                        # Set sub_profile_type to the second highest scoring profile type
                        second_highest_profile = get_second_highest_profile_type(profile_scores)
                        if second_highest_profile:
                            db.update_resume(candidate_id, {'sub_profile_type': second_highest_profile})
                            # Update parsed_data so the API response reflects the correct value
                            parsed_data['sub_profile_type'] = second_highest_profile
                            logger.info(f"Set sub_profile_type={second_highest_profile} for candidate_id={candidate_id}")
                        else:
                            logger.warning(f"No second highest profile type found for candidate_id={candidate_id}. Profile scores: {non_zero_scores}")
                            # Update to None explicitly to ensure database consistency
                            db.update_resume(candidate_id, {'sub_profile_type': None})
                            parsed_data['sub_profile_type'] = None
                    except Exception as e:
                        logger.error(f"Failed to store profile scores for candidate_id={candidate_id}: {e}", exc_info=True)
            except ConnectionError as e:
                logger.error(f"Database connection error: {e}")
                return jsonify({
                    'error': f'Database connection failed: {str(e)}',
                    'details': 'Please ensure MySQL is running and the database exists. Check server logs for more details.'
                }), 500
            except Exception as e:
                logger.error(f"Unexpected error storing resume: {e}", exc_info=True)
                return jsonify({
                    'error': 'Failed to store resume in database',
                    'details': str(e)
                }), 500
            
            # Skip Pinecone indexing - will be done later via /api/indexExistingResumes
            # This allows batch processing and avoids embedding generation during resume upload
            pinecone_indexed = False
            pinecone_error = "Indexing deferred to /api/indexExistingResumes endpoint"
            logger.info(f"Pinecone indexing skipped for resume {candidate_id} - will be indexed later via /api/indexExistingResumes")
            
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Fetch latest data from database to ensure we have the updated sub_profile_type
            # (it might have been updated after initial insert)
            latest_resume_data = None
            try:
                with create_ats_database() as db:
                    latest_resume_data = db.get_resume_by_id(candidate_id)
            except Exception as e:
                logger.warning(f"Failed to fetch latest resume data for response: {e}")
                # Fall back to parsed_data if database fetch fails
            
            # Use database values if available, otherwise fall back to parsed_data
            if latest_resume_data:
                # Database has the most up-to-date values (especially sub_profile_type)
                response_data = {
                    'status': 'success',
                    'message': 'Resume processed successfully',
                    'candidate_id': candidate_id,
                    'candidate_name': latest_resume_data.get('name') or parsed_data.get('name'),
                    'email': latest_resume_data.get('email') or parsed_data.get('email'),
                    'phone': latest_resume_data.get('phone') or parsed_data.get('phone'),
                    'total_experience': latest_resume_data.get('total_experience') or parsed_data.get('total_experience'),
                    'primary_skills': latest_resume_data.get('primary_skills') or parsed_data.get('primary_skills'),
                    'domain': latest_resume_data.get('domain') or parsed_data.get('domain'),
                    'education': latest_resume_data.get('education') or parsed_data.get('education'),
                    'profile_type': latest_resume_data.get('profile_type') or parsed_data.get('profile_type'),
                    'role_type': latest_resume_data.get('role_type') or parsed_data.get('role_type'),
                    'subrole_type': latest_resume_data.get('subrole_type') or parsed_data.get('subrole_type'),
                    'sub_profile_type': latest_resume_data.get('sub_profile_type') or parsed_data.get('sub_profile_type'),
                    'current_designation': latest_resume_data.get('current_designation') or parsed_data.get('current_designation'),
                    'embedding_dimensions': 'stored_in_pinecone_only',
                    'pinecone_indexed': pinecone_indexed,
                    'pinecone_error': pinecone_error if not pinecone_indexed else None,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Fallback to parsed_data if database fetch failed
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
                    'role_type': parsed_data.get('role_type'),
                    'subrole_type': parsed_data.get('subrole_type'),
                    'sub_profile_type': parsed_data.get('sub_profile_type'),
                    'embedding_dimensions': 'stored_in_pinecone_only',
                    'pinecone_indexed': pinecone_indexed,
                    'pinecone_error': pinecone_error if not pinecone_indexed else None,
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
                    from profile_type_utils import (
                        get_all_profile_type_scores,
                        get_second_highest_profile_type,
                    )
                    profile_scores = get_all_profile_type_scores(
                        primary_skills=parsed_data.get('primary_skills', ''),
                        secondary_skills=parsed_data.get('secondary_skills', ''),
                        resume_text=parsed_data.get('resume_text', '')
                    )
                    
                    # Log profile scores before storing
                    sorted_scores = sorted(profile_scores.items(), key=lambda x: x[1], reverse=True)
                    non_zero_scores = [(pt, score) for pt, score in sorted_scores if score > 0]
                    logger.info(f"Calculated profile scores for candidate_id={candidate_id}: {non_zero_scores[:5]}")  # Top 5
                    
                    # Store profile scores and check return value
                    success = db.insert_or_update_profile_scores(candidate_id, profile_scores)
                    if success:
                        logger.info(f"Successfully stored profile scores for candidate_id={candidate_id}")
                    else:
                        logger.error(f"FAILED to store profile scores for candidate_id={candidate_id}. Check database logs for details.")
                        # Log the scores that should have been stored for debugging
                        logger.error(f"Profile scores that failed to store: {non_zero_scores}")
                    
                    # Set sub_profile_type to the second highest scoring profile type
                    second_highest_profile = get_second_highest_profile_type(profile_scores)
                    if second_highest_profile:
                        db.update_resume(candidate_id, {'sub_profile_type': second_highest_profile})
                        # Update parsed_data so the API response reflects the correct value
                        parsed_data['sub_profile_type'] = second_highest_profile
                        logger.info(f"Set sub_profile_type={second_highest_profile} for candidate_id={candidate_id}")
                    else:
                        logger.warning(f"No second highest profile type found for candidate_id={candidate_id}. Profile scores: {non_zero_scores}")
                        # Update to None explicitly to ensure database consistency
                        db.update_resume(candidate_id, {'sub_profile_type': None})
                        parsed_data['sub_profile_type'] = None
                except Exception as e:
                    logger.error(f"Failed to store profile scores for candidate_id={candidate_id}: {e}", exc_info=True)

            # Index in Pinecone if enabled
            pinecone_indexed = False
            pinecone_error = None
            
            # Check Pinecone configuration
            if not ATSConfig.USE_PINECONE:
                logger.warning(f"Pinecone indexing skipped: USE_PINECONE={ATSConfig.USE_PINECONE}")
                pinecone_error = "Pinecone is disabled (USE_PINECONE=False)"
            elif not ATSConfig.PINECONE_API_KEY:
                logger.warning(f"Pinecone indexing skipped: PINECONE_API_KEY is missing")
                pinecone_error = "Pinecone API key is missing"
            else:
                try:
                    logger.info(f"Attempting to index resume {candidate_id} in Pinecone...")
                    
                    # Determine the correct index based on profile_type
                    profile_type = parsed_data.get('profile_type') or 'Generalist'
                    index_name = get_index_name_from_profile_type(profile_type)
                    logger.info(f"Profile type '{profile_type}' mapped to index '{index_name}'")
                    logger.info(f"Pinecone config: index={index_name}, dimension={ATSConfig.EMBEDDING_DIMENSION}")
                    
                    from enhanced_pinecone_manager import EnhancedPineconeManager
                    pinecone_manager = EnhancedPineconeManager(
                        api_key=ATSConfig.PINECONE_API_KEY,
                        index_name=index_name,  # Use the profile-specific index
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
                        'role_type': parsed_data.get('role_type') or 'Unknown',
                        'subrole_type': parsed_data.get('subrole_type') or 'Unknown',
                        'sub_profile_type': parsed_data.get('sub_profile_type') or 'Unknown',
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

                    # Upsert to Pinecone without namespace (using separate indexes)
                    pinecone_manager.upsert_vectors([vector_data], namespace=None)
                    pinecone_indexed = True
                    logger.info(f"Successfully indexed resume {candidate_id} in Pinecone index '{index_name}'")
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Failed to index resume {candidate_id} in Pinecone: {error_msg}", exc_info=True)
                    pinecone_error = error_msg
                    # Do not fail request for Pinecone errors

            # Clean up temporary file
            if os.path.exists(file_path):
                os.remove(file_path)

            # Fetch latest data from database to ensure we have the updated sub_profile_type
            # (it might have been updated after initial insert)
            latest_resume_data = None
            try:
                with create_ats_database() as db:
                    latest_resume_data = db.get_resume_by_id(candidate_id)
            except Exception as e:
                logger.warning(f"Failed to fetch latest resume data for response: {e}")
                # Fall back to parsed_data if database fetch fails
            
            # Use database values if available, otherwise fall back to parsed_data
            if latest_resume_data:
                # Database has the most up-to-date values (especially sub_profile_type)
                response_data = {
                    'status': 'success',
                    'message': 'Resume processed successfully',
                    'candidate_id': candidate_id,
                    'candidate_name': latest_resume_data.get('name') or parsed_data.get('name'),
                    'email': latest_resume_data.get('email') or parsed_data.get('email'),
                    'total_experience': latest_resume_data.get('total_experience') or parsed_data.get('total_experience'),
                    'primary_skills': latest_resume_data.get('primary_skills') or parsed_data.get('primary_skills'),
                    'domain': latest_resume_data.get('domain') or parsed_data.get('domain'),
                    'education': latest_resume_data.get('education') or parsed_data.get('education'),
                    'profile_type': latest_resume_data.get('profile_type') or parsed_data.get('profile_type'),
                    'role_type': latest_resume_data.get('role_type') or parsed_data.get('role_type'),
                    'subrole_type': latest_resume_data.get('subrole_type') or parsed_data.get('subrole_type'),
                    'sub_profile_type': latest_resume_data.get('sub_profile_type') or parsed_data.get('sub_profile_type'),
                    'embedding_dimensions': 'stored_in_pinecone_only',
                    'current_designation': parsed_data.get('current_designation') or 'Unknown',
                    'pinecone_indexed': pinecone_indexed,
                    'pinecone_error': pinecone_error if not pinecone_indexed else None,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Fallback to parsed_data if database fetch failed
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
                    'role_type': parsed_data.get('role_type'),
                    'subrole_type': parsed_data.get('subrole_type'),
                    'sub_profile_type': parsed_data.get('sub_profile_type'),
                    'embedding_dimensions': 'stored_in_pinecone_only',
                    'current_designation': parsed_data.get('current_designation') or 'Unknown',
                    'pinecone_indexed': pinecone_indexed,
                    'pinecone_error': pinecone_error if not pinecone_indexed else None,
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
        
        # Get only unindexed resumes from database
        with create_ats_database() as db:
            resumes = db.get_resumes_by_index_status(indexed=False)
        
        if not resumes:
            return jsonify({
                'status': 'success',
                'message': 'No unindexed resumes found in database. All resumes are already indexed.',
                'total_resumes': 0,
                'indexed_count': 0,
                'failed_count': 0,
                'skipped_count': 0,
                'timestamp': datetime.now().isoformat()
            }), 200
        
        logger.info(f"Found {len(resumes)} unindexed resumes to process")
        
        # Initialize Pinecone managers for different profile types
        # We'll create managers on-demand based on profile_type
        pinecone_managers = {}  # Cache of managers by index_name
        
        indexed_count = 0
        failed_count = 0
        skipped_count = 0
        # Group vectors by index_name for efficient batch upserting
        vectors_by_index = {}
        
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
                    'role_type': resume.get('role_type') or 'Unknown',
                    'subrole_type': resume.get('subrole_type') or 'Unknown',
                    'sub_profile_type': resume.get('sub_profile_type') or 'Unknown',
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
                
                # Determine index name from profile_type (using separate indexes, not namespaces)
                profile_type = resume.get('profile_type') or 'Generalist'
                index_name = get_index_name_from_profile_type(profile_type)
                
                # Get or create Pinecone manager for this index
                if index_name not in pinecone_managers:
                    from enhanced_pinecone_manager import EnhancedPineconeManager
                    pinecone_managers[index_name] = EnhancedPineconeManager(
                        api_key=ATSConfig.PINECONE_API_KEY,
                        index_name=index_name,
                        dimension=ATSConfig.EMBEDDING_DIMENSION
                    )
                    pinecone_managers[index_name].get_or_create_index()
                    logger.info(f"Initialized Pinecone manager for index '{index_name}'")
                
                # Group vectors by index_name
                if index_name not in vectors_by_index:
                    vectors_by_index[index_name] = []
                vectors_by_index[index_name].append(vector_data)
                
                # Batch upsert every 100 vectors per index
                if len(vectors_by_index[index_name]) >= 100:
                    try:
                        pinecone_managers[index_name].upsert_vectors(vectors_by_index[index_name], namespace=None)
                        logger.info(f"Upserted 100 vectors to index '{index_name}'")
                        
                        # Update database status for successfully indexed resumes
                        with create_ats_database() as db:
                            for vec in vectors_by_index[index_name]:
                                vec_candidate_id = int(vec['id'].replace('resume_', ''))
                                if db.update_pinecone_index_status(vec_candidate_id, indexed=True):
                                    indexed_count += 1
                                else:
                                    failed_count += 1
                        
                        vectors_by_index[index_name] = []
                    except Exception as e:
                        logger.error(f"Failed to upsert batch to index '{index_name}': {e}")
                        failed_count += len(vectors_by_index[index_name])
                        vectors_by_index[index_name] = []
                    
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
        
        # Upsert remaining vectors grouped by index_name
        for index_name, vectors in vectors_by_index.items():
            if vectors:
                try:
                    pinecone_managers[index_name].upsert_vectors(vectors, namespace=None)
                    logger.info(f"Upserted {len(vectors)} remaining vectors to index '{index_name}'")
                    
                    # Update database status for successfully indexed resumes
                    with create_ats_database() as db:
                        for vec in vectors:
                            vec_candidate_id = int(vec['id'].replace('resume_', ''))
                            db.update_pinecone_index_status(vec_candidate_id, indexed=True)
                            indexed_count += 1
                except Exception as e:
                    logger.error(f"Failed to upsert remaining vectors to index '{index_name}': {e}")
                    failed_count += len(vectors)
        
        # Calculate skipped count (resumes that were already indexed)
        with create_ats_database() as db:
            total_resumes = len(db.get_all_resumes(status='active', limit=100000))
            indexed_resumes = len(db.get_resumes_by_index_status(indexed=True, status='active', limit=100000))
            skipped_count = indexed_resumes
        
        return jsonify({
            'status': 'success',
            'message': 'Batch indexing completed',
            'total_resumes': total_resumes,
            'processed_count': len(resumes),
            'indexed_count': indexed_count,
            'failed_count': failed_count,
            'skipped_count': skipped_count,
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


def calculate_sql_score(
    candidate: Dict[str, Any],
    detected_main_role: Optional[str],
    detected_subrole: Optional[str],
    detected_profile_type: Optional[str],
    detected_skills: List[Dict[str, Any]],
    skill_score_columns: List[str],
    max_skill_score: float = 200.0
) -> float:
    """
    Calculate SQL-based relevance score (0-1) based on role, profile type, and skill matches.
    
    Args:
        candidate: Candidate data dictionary with role_type, profile_type, and skill scores
        detected_main_role: Detected role from query (e.g., "Software Engineer")
        detected_subrole: Detected subrole from query (e.g., "Frontend Developer")
        detected_profile_type: Detected profile type from query (e.g., "Java")
        detected_skills: List of detected skills with score_column info
        skill_score_columns: List of score column names to check
        max_skill_score: Maximum possible skill score for normalization (default 200)
    
    Returns:
        SQL relevance score between 0.0 and 1.0
    """
    # 1. Role match score (0-1)
    role_score = 0.5  # Default neutral if no role detected
    if detected_main_role:
        candidate_role = candidate.get('role_type', '')
        if candidate_role == detected_main_role:
            role_score = 1.0
        elif detected_subrole:
            # Check subrole match as bonus
            candidate_subrole = candidate.get('subrole_type', '')
            if candidate_subrole == detected_subrole:
                role_score = 1.0  # Perfect match with subrole
            else:
                role_score = 0.0  # Role detected but doesn't match
        else:
            role_score = 0.0  # Role detected but doesn't match
    
    # 2. Profile type match score (0-1) with fuzzy matching
    profile_score = 0.5  # Default neutral if no profile type detected
    if detected_profile_type:
        candidate_profile = candidate.get('profile_type', '').lower()
        detected_profile_lower = detected_profile_type.lower()
        
        if not candidate_profile:
            profile_score = 0.0
        elif candidate_profile == detected_profile_lower:
            # Exact match
            profile_score = 1.0
        elif detected_profile_lower in candidate_profile:
            # Partial match (e.g., "Java" in "Java, Python")
            profile_score = 0.8
        elif candidate_profile in detected_profile_lower:
            # Reverse partial match
            profile_score = 0.7
        else:
            # Check if profile type contains any part of detected profile
            profile_parts = candidate_profile.split(',')
            if any(detected_profile_lower in part.strip() for part in profile_parts):
                profile_score = 0.6
            else:
                profile_score = 0.0
    
    # 3. Skill score (0-1) - normalize from candidate_profile_scores
    skill_scores = []
    for skill_info in detected_skills:
        score_col = skill_info.get('score_column')
        if score_col:
            # Get raw score from candidate data (will be fetched from DB)
            raw_score = candidate.get(score_col, 0.0)
            # Convert to float (MySQL returns Decimal type, need to convert for division)
            raw_score = float(raw_score) if raw_score else 0.0
            
            if raw_score > 0:
                # Normalize: clamp to max_skill_score and convert to 0-1
                normalized = min(1.0, raw_score / max_skill_score)
                skill_scores.append(normalized)
    
    # Calculate average skill score, or use neutral if no skills found
    if skill_scores:
        # If multiple skills, weight first skill more (primary skill)
        if len(skill_scores) == 1:
            skill_score = skill_scores[0]
        else:
            # Weight primary skill 60%, others 40% average
            skill_score = (skill_scores[0] * 0.6) + (sum(skill_scores[1:]) / len(skill_scores[1:]) * 0.4)
    else:
        # No skill scores available - use neutral
        skill_score = 0.5
    
    # Combine with weights: role (20%), profile (30%), skill (50%)
    sql_score = (role_score * 0.2) + (profile_score * 0.3) + (skill_score * 0.5)
    
    # Clamp to 0-1 range
    return max(0.0, min(1.0, sql_score))


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
        
        # === STEP 1: Name Detection (with fallback to skill search) ===
        name_search_performed = False
        if looks_like_name(query):
            logger.info(f"Detected name search for: {query}")
            name_search_performed = True
            with create_ats_database() as db:
                # Conditionally add LIMIT clause to handle None limit
                limit_clause = "LIMIT %s" if limit is not None else ""
                query_lower = query.lower()
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
                        rm.created_at,
                        CASE 
                            -- Exact match (highest priority)
                            WHEN LOWER(rm.name) = %s THEN 10
                            -- Partial match (substring)
                            WHEN LOWER(rm.name) LIKE CONCAT('%%', %s, '%%') THEN 8
                            -- Full name SOUNDEX match
                            WHEN SOUNDEX(rm.name) = SOUNDEX(%s) THEN 7
                            -- Word-by-word SOUNDEX matching (NEW)
                            -- First word
                            WHEN SOUNDEX(SUBSTRING_INDEX(rm.name, ' ', 1)) = SOUNDEX(%s) THEN 6
                            -- Second word (if name has 2+ words)
                            WHEN (LENGTH(rm.name) - LENGTH(REPLACE(rm.name, ' ', '')) >= 1 
                                  AND SOUNDEX(SUBSTRING_INDEX(SUBSTRING_INDEX(rm.name, ' ', 2), ' ', -1)) = SOUNDEX(%s)) THEN 6
                            -- Third word (if name has 3+ words)
                            WHEN (LENGTH(rm.name) - LENGTH(REPLACE(rm.name, ' ', '')) >= 2
                                  AND SOUNDEX(SUBSTRING_INDEX(SUBSTRING_INDEX(rm.name, ' ', 3), ' ', -1)) = SOUNDEX(%s)) THEN 6
                            -- Last word
                            WHEN SOUNDEX(SUBSTRING_INDEX(rm.name, ' ', -1)) = SOUNDEX(%s) THEN 6
                            ELSE 0
                        END AS name_match_score
                    FROM resume_metadata rm
                    WHERE rm.status = 'active'
                      AND (
                          -- Exact match
                          LOWER(rm.name) = %s
                          -- Partial match (substring)
                          OR LOWER(rm.name) LIKE CONCAT('%%', %s, '%%')
                          -- Full name SOUNDEX
                          OR SOUNDEX(rm.name) = SOUNDEX(%s)
                          -- Word-by-word SOUNDEX (NEW)
                          -- First word
                          OR SOUNDEX(SUBSTRING_INDEX(rm.name, ' ', 1)) = SOUNDEX(%s)
                          -- Second word (if name has 2+ words)
                          OR (LENGTH(rm.name) - LENGTH(REPLACE(rm.name, ' ', '')) >= 1 
                              AND SOUNDEX(SUBSTRING_INDEX(SUBSTRING_INDEX(rm.name, ' ', 2), ' ', -1)) = SOUNDEX(%s))
                          -- Third word (if name has 3+ words)
                          OR (LENGTH(rm.name) - LENGTH(REPLACE(rm.name, ' ', '')) >= 2
                              AND SOUNDEX(SUBSTRING_INDEX(SUBSTRING_INDEX(rm.name, ' ', 3), ' ', -1)) = SOUNDEX(%s))
                          -- Last word
                          OR SOUNDEX(SUBSTRING_INDEX(rm.name, ' ', -1)) = SOUNDEX(%s)
                      )
                    ORDER BY name_match_score DESC, rm.total_experience DESC
                    {limit_clause}
                """
                # Parameters: 
                # - query_lower used for exact/partial matches (case-insensitive)
                # - query (original) used for SOUNDEX (case-insensitive function)
                # Total: 14 parameters (7 for CASE, 7 for WHERE)
                params = [
                    # CASE statement parameters (7)
                    query_lower,  # 1. Exact match
                    query_lower,  # 2. Partial match
                    query,        # 3. Full name SOUNDEX
                    query,        # 4. First word SOUNDEX
                    query,        # 5. Second word SOUNDEX
                    query,        # 6. Third word SOUNDEX
                    query,        # 7. Last word SOUNDEX
                    # WHERE clause parameters (7)
                    query_lower,  # 8. Exact match
                    query_lower,  # 9. Partial match
                    query,        # 10. Full name SOUNDEX
                    query,        # 11. First word SOUNDEX
                    query,        # 12. Second word SOUNDEX
                    query,        # 13. Third word SOUNDEX
                    query         # 14. Last word SOUNDEX
                ]
                if limit is not None:
                    params.append(limit)
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
                        'name_match_score': int(float(r.get('name_match_score', 0))),  # Name matching score (0-10)
                        'match_score': float(r.get('name_match_score', 0)) / 10.0,  # Normalized to 0-1 for consistency
                    })
                
                # FALLBACK LOGIC: If name search returns 0 results, continue to skill/role search
                if len(candidates) == 0:
                    logger.info(f"Name search returned 0 results for '{query}', falling back to skill/role search")
                    # Don't return - continue to skill/role search below
                    name_search_performed = False  # Reset flag so skill search proceeds
                else:
                    # Name search found results - return them
                    # Build the response for name search
                    response_data = {
                        'query': query,
                        'search_type': 'name_search',
                        'analysis': {'detected_as': 'name'},
                        'count': len(candidates),
                        'results': candidates,
                        'processing_time_ms': int((time.time() - start_time) * 1000),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # === SAVE CHAT HISTORY FOR NAME SEARCH ===
                    chat_role = data.get('role')
                    chat_sub_role = data.get('sub_role')
                    chat_profile_type = data.get('profile_type')
                    chat_sub_profile_type = data.get('sub_profile_type')
                    chat_candidate_id = data.get('candidate_id')
                    
                    try:
                        with create_ats_database() as history_db:
                            history_db.insert_chat_history(
                                chat_msg=query,
                                response=json.dumps(response_data, default=str),
                                candidate_id=chat_candidate_id,
                                role=chat_role,
                                sub_role=chat_sub_role,
                                profile_type=chat_profile_type,
                                sub_profile_type=chat_sub_profile_type
                            )
                    except Exception as history_error:
                        logger.warning(f"Failed to save chat history for name search: {history_error}")
                    
                    return jsonify(response_data), 200
        
        # === STEP 1.5: Irrelevant Detection (EARLY) ===
        cleaned_query = clean_partial_query(query)
        if is_irrelevant_query(cleaned_query):
            logger.info(f"Detected irrelevant query: {cleaned_query}")
            return jsonify({
                'error': 'Please provide a valid search query with technical terms (e.g., role, skill, technology).',
                'query': query,
                'query_type': 'irrelevant'
            }), 400
        
        # === STEP 2: Role & Sub Role Detection (4-level priority) ===
        detected_subrole = None
        detected_main_role = None
        detected_profile_type_from_role = None
        role_detected_explicitly = False  # Track if role was explicitly mentioned vs inferred
        
        # 2a. Check explicit subroles first (Frontend/Backend/Full Stack Developer)
        explicit_subrole = detect_explicit_subrole(cleaned_query)
        if explicit_subrole:
            logger.info(f"Detected explicit subrole: {explicit_subrole}")
            detected_subrole = explicit_subrole
            detected_main_role = "Software Engineer"  # Default for these subroles
            role_detected_explicitly = True  # Explicit subrole = role explicitly mentioned
        
        # 2b. Check technology + role pattern (java Developer, python Engineer)
        tech_role_info = None
        if not detected_main_role:
            tech_role_info = detect_tech_role_pattern(cleaned_query)
            if tech_role_info:
                logger.info(f"Detected tech-role pattern: {tech_role_info}")
                detected_main_role = tech_role_info['main_role']
                detected_profile_type_from_role = tech_role_info['profile_type']
                role_detected_explicitly = True  # Tech-role pattern = role explicitly mentioned
                # NOTE: subrole_type is NOT set for tech-role patterns
        
        # Helper function to check if query contains role keywords
        def has_role_keyword_in_query(query: str) -> bool:
            """Check if query contains explicit role keywords."""
            query_lower = normalize_text(query)
            role_keywords = ['developer', 'engineer', 'programmer', 'architect', 'manager', 
                           'analyst', 'administrator', 'specialist', 'consultant', 'lead',
                           'director', 'coordinator', 'supervisor', 'tester', 'qa']
            return any(keyword in query_lower for keyword in role_keywords)
        
        # 2c. Check role from role_extract.py hierarchy
        if not detected_main_role:
            role_from_hierarchy = detect_role_from_hierarchy(cleaned_query)
            if role_from_hierarchy:
                logger.info(f"Detected role from hierarchy: {role_from_hierarchy}")
                detected_main_role = role_from_hierarchy
                # Check if query contains role keywords to determine if explicit
                if has_role_keyword_in_query(cleaned_query):
                    role_detected_explicitly = True  # Role keyword found = explicit
        
        # 2d. Check role_processor table
        if not detected_main_role:
            role_from_processor = detect_role_from_processor_table(cleaned_query)
            if role_from_processor:
                logger.info(f"Detected role from processor table: {role_from_processor}")
                detected_main_role = role_from_processor
                # Check if query contains role keywords
                if has_role_keyword_in_query(cleaned_query):
                    role_detected_explicitly = True  # Role keyword found = explicit
        
        # 2e. Fallback: Use existing detect_subrole_from_query if still no role
        if not detected_main_role and not detected_subrole:
            role_info = detect_subrole_from_query(cleaned_query)
            if role_info:
                detected_subrole = role_info['sub_role']
                detected_main_role = role_info['main_role']
                detected_profile_type_from_role = role_info['profile_type']
                # Check if query contains role keywords
                if has_role_keyword_in_query(cleaned_query):
                    role_detected_explicitly = True  # Role keyword found = explicit
        
        # === STEP 3: Skill Detection (with boolean and comma-separated support) ===
        # Check if query is comma-separated (e.g., "django, mysql")
        is_comma_separated = ',' in cleaned_query and ' AND ' not in cleaned_query.upper() and ' OR ' not in cleaned_query.upper()
        
        # Parse query structure
        boolean_structure = None
        comma_separated_structure = None
        
        if is_comma_separated:
            # Parse comma-separated query (treats comma as AND)
            try:
                comma_separated_structure = parse_comma_separated_query(cleaned_query)
                logger.info(f"Detected comma-separated query: {comma_separated_structure}")
                # Convert comma-separated structure to boolean structure format for skill extraction
                boolean_structure = comma_separated_structure
            except Exception as e:
                logger.warning(f"Error parsing comma-separated query: {e}")
        else:
            # Parse boolean query (AND/OR)
            try:
                boolean_structure = parse_boolean_query(cleaned_query)
            except Exception as e:
                logger.warning(f"Error parsing boolean query: {e}")
        
        # Extract skills with boolean/comma-separated support
        detected_skills = extract_skills_from_query_enhanced(cleaned_query, boolean_structure)
        
        # OPTION 1: Use PRIMARY profile type score only for each skill
        # This prevents "mysql" from requiring both database_score AND fullstack_score
        # For both comma-separated ("django, mysql") and single skill ("mysql") queries
        primary_skill_scores = []
        
        if is_comma_separated and comma_separated_structure and comma_separated_structure.get('and_terms'):
            # Comma-separated query: Extract primary profile type for each term
            for term_group in comma_separated_structure['and_terms']:
                for term in term_group:
                    # Extract skills for this specific term
                    term_skills = extract_skills_from_query_enhanced(term, None)
                    if term_skills:
                        # Use FIRST skill's profile type (primary) only
                        primary_skill = term_skills[0]
                        primary_score_col = primary_skill.get('score_column')
                        if primary_score_col and primary_score_col not in primary_skill_scores:
                            primary_skill_scores.append(primary_score_col)
                            logger.info(f"Comma-separated query: Using primary score column '{primary_score_col}' for term '{term}'")
        else:
            # Single skill query: Use FIRST detected skill's primary profile type only
            # This handles cases like "mysql" where multiple profile types are detected
            # Priority: Exact match skills first, then substring matches
            if detected_skills:
                # Sort skills: exact matches first (skill name matches query), then others
                query_lower = normalize_text(cleaned_query)
                sorted_skills = sorted(detected_skills, key=lambda s: (
                    0 if normalize_text(s.get('skill', '')) == query_lower else 1,  # Exact match first
                    s.get('skill', '')  # Then alphabetically
                ))
                
                primary_skill = sorted_skills[0]
                primary_score_col = primary_skill.get('score_column')
                if primary_score_col:
                    primary_skill_scores.append(primary_score_col)
                    logger.info(f"Single skill query: Using primary score column '{primary_score_col}' for skill '{primary_skill.get('skill')}' (profile_type: {primary_skill.get('profile_type')})")
        
        # Override skill_score_columns with primary profile type scores only
        if primary_skill_scores:
            skill_score_columns = primary_skill_scores
            logger.info(f"Using primary score columns only (Option 1 - AND logic): {skill_score_columns}")
        else:
            # Fallback to original logic if no primary scores found
            skill_score_columns = [s['score_column'] for s in detected_skills if s.get('score_column')]
            logger.info(f"Fallback: Using all detected score columns: {skill_score_columns}")
        
        skill_profile_types = [s['profile_type'] for s in detected_skills]
        first_skill_profile_type = skill_profile_types[0] if skill_profile_types else None
        
        # === STEP 4: Profile Type Detection (Priority-based) ===
        if not detected_profile_type_from_role:
            profile_type = detect_profile_type_priority(
                detected_main_role,
                detected_skills,
                explicit_subrole
            )
            if profile_type:
                detected_profile_type_from_role = profile_type
                logger.info(f"Detected profile type: {profile_type}")
        
        # Fallback: Use profile_type from first skill if still not detected
        if not detected_profile_type_from_role and first_skill_profile_type:
            detected_profile_type_from_role = first_skill_profile_type
            logger.info(f"Using profile type from first skill: {first_skill_profile_type}")
        
        # Canonicalize profile types to ensure consistent matching with database
        if detected_profile_type_from_role:
            detected_profile_type_from_role = canonicalize_profile_type(detected_profile_type_from_role)
            logger.info(f"Canonicalized profile type: {detected_profile_type_from_role}")
        
        # Also canonicalize skill profile types
        canonicalized_skill_profile_types = []
        for pt in skill_profile_types:
            canonical_pt = canonicalize_profile_type(pt)
            if canonical_pt and canonical_pt not in canonicalized_skill_profile_types:
                canonicalized_skill_profile_types.append(canonical_pt)
        skill_profile_types = canonicalized_skill_profile_types
        
        # Infer sub-role from profile_type (only for skill-only queries, not for tech-role patterns)
        if not detected_subrole and detected_profile_type_from_role and not tech_role_info:
            inferred_role_info = infer_subrole_from_profile_type(detected_profile_type_from_role)
            if inferred_role_info:
                detected_subrole = inferred_role_info['sub_role']
                if not detected_main_role:
                    detected_main_role = inferred_role_info['main_role']
        
        # === STEP 5: Location Extraction ===
        detected_locations = extract_locations(cleaned_query)
        if detected_locations:
            logger.info(f"Detected locations: {detected_locations}")
        
        # === STEP 6: Input Validation ===
        # FIRST: Check if query contains any skill from TECH_SKILLS (6000+ skills)
        # This ensures skills in TECH_SKILLS are accepted even if extract_skills_from_query_enhanced() didn't detect them
        query_has_valid_skill = False
        try:
            query_lower = normalize_text(cleaned_query)
            # Check if any skill from TECH_SKILLS is in the query
            for skill in TECH_SKILLS:
                skill_lower = skill.lower()
                # Use word boundary matching to avoid false positives (e.g., "saas" in "saas platform")
                if skill_lower in query_lower:
                    query_has_valid_skill = True
                    logger.info(f"Query contains valid skill from TECH_SKILLS: {skill}")
                    break
        except Exception as e:
            logger.warning(f"Error checking TECH_SKILLS for validation: {e}")
        
        # Only show error if:
        # 1. No role detected AND
        # 2. No subrole detected AND
        # 3. No skills detected from extract_skills_from_query_enhanced() AND
        # 4. Query does NOT contain any skill from TECH_SKILLS
        if not detected_main_role and not detected_subrole and not detected_skills and not query_has_valid_skill:
            return jsonify({
                'error': 'Please refine the search and include at least one role or one skill.'
            }), 400
        
        # === STEP 6: Build SQL Query ===
        where_clauses = ["rm.status = 'active'"]
        params = []
        
        # Role filter: Only apply if role was EXPLICITLY mentioned in query (not inferred from skills)
        # Skill-only queries (e.g., "mysql", "mysql, django", "python or css") should NOT filter by role
        # Role queries (e.g., "java developer") should filter by role
        # Note: role_detected_explicitly flag is set during role detection above
        
        if detected_main_role and role_detected_explicitly:
            # Role was explicitly mentioned in query - apply role filter
            where_clauses.append("rm.role_type = %s")
            params.append(detected_main_role)
            logger.info(f"Role filter applied: role='{detected_main_role}' (explicitly mentioned in query)")
        elif detected_main_role and not role_detected_explicitly:
            # Role was inferred from skills only - skip role filter for skill-only queries
            logger.info(f"Role filter skipped: role='{detected_main_role}' was inferred from skills, query is skill-only")
        
        # Subrole filter (ONLY if explicit subrole detected)
        if detected_subrole and explicit_subrole:
            where_clauses.append("rm.subrole_type = %s")
            params.append(detected_subrole)
        
        # Profile Type filter (using canonicalized values)
        # For skill-only queries, make profile_type filter optional (use OR with skill matching)
        profile_type_filters = []
        if detected_profile_type_from_role:
            # Use canonicalized profile type for LIKE matching
            canonical_profile_type = canonicalize_profile_type(detected_profile_type_from_role)
            profile_type_filters.append("rm.profile_type LIKE %s")
            params.append(f"%{canonical_profile_type}%")
            logger.info(f"Adding profile_type filter: {canonical_profile_type}")
        
        # Also add profile types from skills (canonicalized)
        for pt in skill_profile_types:
            canonical_pt = canonicalize_profile_type(pt)
            # Avoid duplicates
            if canonical_pt and canonical_pt not in [detected_profile_type_from_role] if detected_profile_type_from_role else []:
                profile_type_filters.append("rm.profile_type LIKE %s")
                params.append(f"%{canonical_pt}%")
                logger.info(f"Adding skill profile_type filter: {canonical_pt}")
        
        # Score-based filtering and fetching (Hybrid Approach: Option 5)
        # Use score columns for ALL detected skills (more reliable than profile_type matching alone)
        # Only filter by score columns that actually exist in the database
        join_clause = ""
        valid_score_columns = [col for col in skill_score_columns if col]  # Filter out None values
        use_score_filtering = True  # Flag to control whether to filter by scores
        
        # Skill matching filters (for skill-only queries, check skills directly)
        skill_match_filters = []
        if detected_skills:
            # For skill-only queries, also check if skills appear in primary_skills/all_skills
            # This helps find candidates even if they don't have profile scores calculated
            for skill_info in detected_skills:
                skill_name = skill_info.get('skill', '').lower()
                if skill_name:
                    # Check in primary_skills, secondary_skills, and all_skills
                    skill_match_filters.append(
                        "(LOWER(rm.primary_skills) LIKE %s OR LOWER(rm.secondary_skills) LIKE %s OR LOWER(rm.all_skills) LIKE %s)"
                    )
                    skill_pattern = f"%{skill_name}%"
                    params.extend([skill_pattern, skill_pattern, skill_pattern])
                    logger.info(f"Adding skill match filter: {skill_name}")
        
        if valid_score_columns:
            # Use LEFT JOIN instead of INNER JOIN for skill-only queries
            # This includes candidates even if they don't have entries in candidate_profile_scores
            # For skill-only queries (no explicit role), use LEFT JOIN
            # For role queries, we can still use LEFT JOIN but filter by score if available
            if role_detected_explicitly:
                # Role query: Use INNER JOIN (strict matching)
                join_clause = "INNER JOIN candidate_profile_scores cps ON rm.candidate_id = cps.candidate_id"
            else:
                # Skill-only query: Use LEFT JOIN (include candidates without scores)
                join_clause = "LEFT JOIN candidate_profile_scores cps ON rm.candidate_id = cps.candidate_id"
                logger.info("Using LEFT JOIN for skill-only query (includes candidates without profile scores)")
            
            # Filter by score > 0 for each detected skill with valid score column
            # For skill-only queries, combine with skill matching (OR logic)
            if use_score_filtering:
                score_filters = []
                for col in valid_score_columns:
                    score_filters.append(f"cps.{col} > 0")
                    logger.info(f"Adding score filter: {col} > 0")
                
                # For skill-only queries: (score > 0) OR (skill in primary_skills/all_skills) OR (profile_type match)
                # For role queries: (score > 0) AND (profile_type match)
                if not role_detected_explicitly:
                    # Skill-only: Use OR logic - either has score OR has skill in text OR has matching profile_type
                    or_conditions = []
                    or_conditions.extend(score_filters)
                    if skill_match_filters:
                        or_conditions.extend(skill_match_filters)
                    if profile_type_filters:
                        or_conditions.extend(profile_type_filters)
                    
                    if or_conditions:
                        combined_filter = "(" + " OR ".join(or_conditions) + ")"
                        where_clauses.append(combined_filter)
                        logger.info(f"Using OR logic for skill-only query: (score > 0) OR (skill in skills text) OR (profile_type match)")
                else:
                    # Role query: Use AND logic with score filters
                    where_clauses.extend(score_filters)
                    # Also add profile_type as AND filter for role queries
                    if profile_type_filters:
                        where_clauses.append("(" + " OR ".join(profile_type_filters) + ")")
                        logger.info("Adding profile_type filter for role query")
        elif skill_match_filters:
            # No score columns but have skill matches - use skill matching with optional profile_type
            or_conditions = []
            or_conditions.extend(skill_match_filters)
            if profile_type_filters and not role_detected_explicitly:
                # For skill-only: Include profile_type in OR logic
                or_conditions.extend(profile_type_filters)
                where_clauses.append("(" + " OR ".join(or_conditions) + ")")
                logger.info("Using skill matching with profile_type (OR logic, no score columns available)")
            else:
                # For role queries or no profile_type: Use skill matching only
                where_clauses.append("(" + " OR ".join(skill_match_filters) + ")")
                logger.info("Using skill matching only (no score columns available)")
        elif profile_type_filters and role_detected_explicitly:
            # Role query with profile_type but no score columns or skill matches
            where_clauses.append("(" + " OR ".join(profile_type_filters) + ")")
            logger.info("Using profile_type filter only for role query")
        
        # Build final SQL
        # Always include skill scores in SELECT for SQL score calculation
        skill_score_select = ""
        if valid_score_columns:
            # Add skill score columns to SELECT
            skill_score_select = ", " + ", ".join([f"cps.{col}" for col in valid_score_columns])
        elif join_clause:
            # If join exists but no valid score columns, still join to get empty scores
            skill_score_select = ""
        
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
                {skill_score_select}
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
                candidate_data = {
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
                }
                
                # Add skill scores from candidate_profile_scores if available
                if valid_score_columns:
                    for col in valid_score_columns:
                        candidate_data[col] = r.get(col, 0.0)
                
                sql_candidates.append(candidate_data)
        
        # === STEP 7.5: Calculate SQL Scores for All Candidates ===
        sql_scores = {}
        for candidate in sql_candidates:
            sql_score = calculate_sql_score(
                candidate=candidate,
                detected_main_role=detected_main_role,
                detected_subrole=detected_subrole,
                detected_profile_type=detected_profile_type_from_role,
                detected_skills=detected_skills,
                skill_score_columns=skill_score_columns,
                max_skill_score=200.0
            )
            sql_scores[candidate['candidate_id']] = sql_score
            candidate['sql_score'] = round(sql_score, 4)  # Store for debugging/transparency
        
        # === STEP 8: Semantic Search (Hybrid Approach) ===
        semantic_applied = False
        final_candidates = sql_candidates
        semantic_scores = {}
        
        # Initialize match_score with SQL scores (will be updated if semantic search succeeds)
        for candidate in final_candidates:
            candidate_id = candidate['candidate_id']
            candidate['match_score'] = round(sql_scores.get(candidate_id, 0.5), 4)  # Use SQL score as default
        
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
                    # Handle None limit - use all candidates if limit is None
                    vector_top_k = min(limit * 3, 200) if limit else 200
                    
                    # Query Pinecone with candidate ID filter
                    # Note: EnhancedPineconeManager uses query_vectors method
                    vector_results = pinecone_manager.query_vectors(
                        query_vector=query_embedding,
                        top_k=vector_top_k,
                        include_metadata=True,
                        filter={'candidate_id': {'$in': candidate_ids}}
                    )
                    
                    # Handle both QueryResponse and dict return types
                    matches = vector_results.matches if hasattr(vector_results, 'matches') else (vector_results.get('matches', []) if isinstance(vector_results, dict) else [])
                    
                    if matches:
                        logger.info(f"✅ Pinecone search found {len(matches)} results")
                        
                        # Extract candidate IDs and scores from Pinecone
                        pinecone_candidate_ids = []
                        for match in matches:
                            candidate_id = match.metadata.get('candidate_id')
                            if candidate_id:
                                pinecone_candidate_ids.append(candidate_id)
                                semantic_scores[candidate_id] = match.score
                        
                        # Calculate hybrid match_score: SQL (30%) + Pinecone (70%)
                        for candidate in final_candidates:
                            candidate_id = candidate['candidate_id']
                            sql_score = sql_scores.get(candidate_id, 0.5)
                            pinecone_score = semantic_scores.get(candidate_id, 0.0)
                            
                            # Combine scores: SQL 30% + Pinecone 70%
                            if pinecone_score > 0:
                                hybrid_score = (sql_score * 0.3) + (pinecone_score * 0.7)
                                candidate['match_score'] = round(hybrid_score, 4)
                                candidate['semantic_score'] = round(pinecone_score, 4)  # Store for transparency
                            else:
                                # Only SQL score available
                                candidate['match_score'] = round(sql_score, 4)
                        
                        # Reorder final_candidates by hybrid match_score
                        final_candidates.sort(
                            key=lambda x: x.get('match_score', 0),
                            reverse=True
                        )
                        
                        semantic_applied = True
                        logger.info(f"✅ Semantic search completed: {len(final_candidates)} candidates re-ranked")
                    else:
                        logger.warning("No Pinecone results found, using SQL-only results")
                        semantic_applied = False
                else:
                    logger.warning("No candidate IDs from SQL results, skipping Pinecone search")
                    semantic_applied = False
                    
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}, falling back to SQL-only results")
                logger.error(traceback.format_exc())
                semantic_applied = False
                # Use SQL scores for SQL-only candidates
                for candidate in final_candidates:
                    candidate_id = candidate['candidate_id']
                    candidate['match_score'] = round(sql_scores.get(candidate_id, 0.5), 4)
        
        # === STEP 8: Semantic Search using Process_Search_Query (Query Correct Index) ===
        # This is a fallback/alternative approach - only use if first approach didn't work
        if not semantic_applied and use_semantic and ATSConfig.USE_PINECONE and ATSConfig.PINECONE_API_KEY:
            try:
                logger.info(f"Using Process_Search_Query to search correct Pinecone index for query: {query}")
                
                # Use Process_Search_Query to extract metadata and query the correct index
                from summary_AI_extraction import Process_Search_Query
                
                # Determine profile_type from detected profile types for better index selection
                search_profile_type = detected_profile_type_from_role or first_skill_profile_type
                
                # Call Process_Search_Query to get results from the correct index
                search_result = Process_Search_Query(
                    query=query,
                    profile_type=search_profile_type,  # Pass detected profile_type if available
                    top_k=limit * 3 if limit else 200,  # Get more results for better ranking
                    include_metadata=True
                )
                
                if search_result.get('success') and search_result.get('results'):
                    logger.info(f"✅ Process_Search_Query found {len(search_result['results'].matches)} results from index '{search_result.get('index_name')}'")
                    
                    # Extract candidate IDs and scores from Pinecone results
                    pinecone_candidate_ids = []
                    for match in search_result['results'].matches:
                        candidate_id = match.metadata.get('candidate_id')
                        if candidate_id:
                            pinecone_candidate_ids.append(candidate_id)
                            semantic_scores[candidate_id] = match.score
                    
                    logger.info(f"Found {len(pinecone_candidate_ids)} unique candidates from Pinecone index")
                    
                    # Fetch full candidate details from database using candidate IDs from Pinecone
                    if pinecone_candidate_ids:
                        with create_ats_database() as db:
                            # Build SQL query to fetch candidates by IDs with skill scores
                            placeholders = ','.join(['%s'] * len(pinecone_candidate_ids))
                            skill_score_select = ""
                            join_clause_pinecone = ""
                            if valid_score_columns:
                                skill_score_select = ", " + ", ".join([f"cps.{col}" for col in valid_score_columns])
                                join_clause_pinecone = "INNER JOIN candidate_profile_scores cps ON rm.candidate_id = cps.candidate_id"
                            
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
                                    {skill_score_select}
                                FROM resume_metadata rm
                                {join_clause_pinecone}
                                WHERE rm.status = 'active'
                                  AND rm.candidate_id IN ({placeholders})
                                ORDER BY FIELD(rm.candidate_id, {placeholders})
                            """
                            # Order by IDs in the same order as Pinecone results (by score)
                            ordered_ids = pinecone_candidate_ids.copy()
                            db.cursor.execute(sql, ordered_ids + ordered_ids)
                            results = db.cursor.fetchall()
                            
                            # Build candidates list and calculate hybrid scores
                            pinecone_candidates = []
                            for r in results:
                                candidate_id = r['candidate_id']
                                candidate = {
                                    'candidate_id': candidate_id,
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
                                }
                                
                                # Add skill scores if available
                                if valid_score_columns:
                                    for col in valid_score_columns:
                                        candidate[col] = r.get(col, 0.0)
                                
                                # Calculate SQL score for this candidate
                                sql_score = calculate_sql_score(
                                    candidate=candidate,
                                    detected_main_role=detected_main_role,
                                    detected_subrole=detected_subrole,
                                    detected_profile_type=detected_profile_type_from_role,
                                    detected_skills=detected_skills,
                                    skill_score_columns=skill_score_columns,
                                    max_skill_score=200.0
                                )
                                candidate['sql_score'] = round(sql_score, 4)
                                
                                # Get Pinecone score
                                pinecone_score = semantic_scores.get(candidate_id, 0.0)
                                candidate['semantic_score'] = round(pinecone_score, 4)
                                
                                # Calculate hybrid match_score: SQL (30%) + Pinecone (70%)
                                if pinecone_score > 0:
                                    hybrid_score = (sql_score * 0.3) + (pinecone_score * 0.7)
                                    candidate['match_score'] = round(hybrid_score, 4)
                                else:
                                    # Only SQL score available
                                    candidate['match_score'] = round(sql_score, 4)
                                
                                pinecone_candidates.append(candidate)
                            
                            # Replace final_candidates with Pinecone candidates
                            final_candidates = pinecone_candidates
                            
                            # Sort by hybrid match_score
                            final_candidates.sort(
                                key=lambda x: x.get('match_score', 0),
                                reverse=True
                            )
                            
                            # Apply limit if specified
                            if limit:
                                final_candidates = final_candidates[:limit]
                            
                            semantic_applied = True
                            logger.info(f"✅ Semantic search completed: {len(final_candidates)} candidates from Pinecone index '{search_result.get('index_name')}'")
                    else:
                        logger.warning("No candidate IDs found in Pinecone results")
                        # Fallback to SQL results
                        final_candidates = sql_candidates
                        for candidate in final_candidates:
                            candidate_id = candidate['candidate_id']
                            candidate['match_score'] = round(sql_scores.get(candidate_id, 0.5), 4)
                else:
                    error_msg = search_result.get('error', 'Unknown error')
                    logger.warning(f"Process_Search_Query failed: {error_msg}, falling back to SQL-only results")
                    # Fallback to SQL results
                    final_candidates = sql_candidates
                    for candidate in final_candidates:
                        candidate_id = candidate['candidate_id']
                        candidate['match_score'] = round(sql_scores.get(candidate_id, 0.5), 4)
                    
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}, falling back to SQL-only results")
                logger.error(traceback.format_exc())
                # Fallback to SQL-only results
                if not final_candidates:
                    final_candidates = sql_candidates
                semantic_applied = False
                # Use SQL scores for SQL-only candidates
                for candidate in final_candidates:
                    if 'match_score' not in candidate:
                        candidate_id = candidate['candidate_id']
                        candidate['match_score'] = round(sql_scores.get(candidate_id, 0.5), 4)
        
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
        
        # Determine query type
        if explicit_subrole:
            if detected_skills:
                query_type = 'role_skill_search'
            else:
                query_type = 'role_search'
        elif detected_main_role:
            if detected_skills:
                query_type = 'role_skill_search'
            else:
                query_type = 'role_search'
        elif detected_skills:
            query_type = 'skills_only'
        else:
            query_type = 'unknown'
        
        # Build analysis response
        analysis = {
            'query_type': query_type,
            'detected_subrole': detected_subrole,
            'explicit_subrole_detected': explicit_subrole is not None,
            'detected_main_role': detected_main_role,
            'detected_profile_type': detected_profile_type_from_role,
            'detected_skills': [s['skill'] for s in detected_skills],
            'skill_profile_types': skill_profile_types,
            'skill_score_columns': skill_score_columns,
            'detected_locations': detected_locations,
            'cleaned_query': cleaned_query if 'cleaned_query' in locals() else query,
            'boolean_structure': boolean_structure if 'boolean_structure' in locals() else None,
            'semantic_search_applied': semantic_applied,
            'boolean_filter_applied': boolean_applied,
            'sql_candidates_count': len(sql_candidates),
            'vector_candidates_count': vector_candidates_count if semantic_applied else len(sql_candidates),
            'boolean_filtered_count': boolean_filtered_count if boolean_applied else None,
            'final_candidates_count': len(final_candidates),
        }
        
        # Determine search type
        if boolean_applied and semantic_applied:
            search_type = 'pinecone_vector_boolean'
        elif semantic_applied:
            search_type = 'pinecone_vector'  # Changed from 'sql_vector' since we query Pinecone first
        else:
            search_type = 'sql_only'
        
        # Build the response
        response_data = {
            'query': query,
            'search_type': search_type,
            'analysis': analysis,
            'count': len(final_candidates),
            'results': final_candidates,
            'processing_time_ms': int((time.time() - start_time) * 1000),
            'timestamp': datetime.now().isoformat()
        }
        
        # === SAVE CHAT HISTORY ===
        # Extract role information from request or detected analysis
        chat_role = data.get('role') or detected_main_role
        chat_sub_role = data.get('sub_role') or detected_subrole
        chat_profile_type = data.get('profile_type') or detected_profile_type_from_role
        chat_sub_profile_type = data.get('sub_profile_type')
        chat_candidate_id = data.get('candidate_id')  # Optional: if search is for a specific candidate
        
        try:
            with create_ats_database() as history_db:
                history_db.insert_chat_history(
                    chat_msg=query,
                    response=json.dumps(response_data, default=str),
                    candidate_id=chat_candidate_id,
                    role=chat_role,
                    sub_role=chat_sub_role,
                    profile_type=chat_profile_type,
                    sub_profile_type=chat_sub_profile_type
                )
        except Exception as history_error:
            # Log error but don't fail the main request
            logger.warning(f"Failed to save chat history: {history_error}")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error in /api/searchResumes: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500



# ============================================================================
# Role Mapping Structures for /api/searchResumes
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
    "mysql": {"profile_type": "Database", "score_column": "database_score"},
    "postgresql": {"profile_type": "Database", "score_column": "database_score"},
    "mongodb": {"profile_type": "Database", "score_column": "database_score"},
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


def detect_subrole_frontend_backend(text: str) -> Optional[str]:
    """
    Detect sub_role: Frontend, Backend, or Full Stack Developer.
    Returns one of: 'Frontend', 'Backend', 'Full Stack Developer', or None
    """
    text_lower = normalize_text(text)
    
    # Frontend keywords
    frontend_keywords = [
        'frontend', 'front-end', 'front end', 'ui developer', 'ui/ux', 'user interface',
        'react', 'angular', 'vue', 'javascript', 'typescript', 'html', 'css', 'sass',
        'scss', 'less', 'bootstrap', 'tailwind', 'material-ui', 'jquery', 'webpack',
        'babel', 'gulp', 'grunt', 'responsive design', 'mobile-first'
    ]
    
    # Backend keywords
    backend_keywords = [
        'backend', 'back-end', 'back end', 'server-side', 'server side', 'api developer',
        'rest api', 'graphql', 'microservices', 'spring boot', 'django', 'flask',
        'express', 'node.js', 'asp.net', '.net core', 'java', 'python', 'php',
        'ruby', 'database', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql',
        'oracle', 'redis', 'elasticsearch'
    ]
    
    # Full Stack keywords
    fullstack_keywords = [
        'full stack', 'fullstack', 'full-stack', 'end-to-end', 'end to end',
        'mern', 'mean', 'mevn', 'lamp', 'lemp'
    ]
    
    # Count matches
    frontend_count = sum(1 for keyword in frontend_keywords if keyword in text_lower)
    backend_count = sum(1 for keyword in backend_keywords if keyword in text_lower)
    fullstack_count = sum(1 for keyword in fullstack_keywords if keyword in text_lower)
    
    # Check for explicit full stack mention first
    if fullstack_count > 0 or (frontend_count > 0 and backend_count > 0):
        return 'Full Stack Developer'
    
    # If only frontend keywords
    if frontend_count > 0 and backend_count == 0:
        return 'Frontend'
    
    # If only backend keywords
    if backend_count > 0 and frontend_count == 0:
        return 'Backend'
    
    # Default: if both present but no explicit full stack, return Full Stack
    if frontend_count > 0 and backend_count > 0:
        return 'Full Stack Developer'
    
    return None


def extract_specific_technologies(text: str, required_skills: str = '') -> List[str]:
    """
    Extract specific technologies for profile_sub_type.
    Returns list of technologies like: ['React', 'SQL', 'Angular', 'Node.js']
    """
    text_lower = normalize_text(text + ' ' + str(required_skills))
    
    # Specific technology keywords (case-sensitive for output)
    technology_map = {
        # Frontend
        'react': 'React',
        'angular': 'Angular',
        'vue': 'Vue',
        'typescript': 'TypeScript',
        'javascript': 'JavaScript',
        'jquery': 'jQuery',
        'next.js': 'Next.js',
        'nuxt': 'Nuxt',
        'svelte': 'Svelte',
        
        # Backend/Database
        'sql': 'SQL',
        'postgresql': 'PostgreSQL',
        'mysql': 'MySQL',
        'mongodb': 'MongoDB',
        'redis': 'Redis',
        'oracle': 'Oracle',
        'sql server': 'SQL Server',
        'dynamodb': 'DynamoDB',
        'cassandra': 'Cassandra',
        
        # Frameworks
        'spring boot': 'Spring Boot',
        'django': 'Django',
        'flask': 'Flask',
        'express': 'Express',
        'node.js': 'Node.js',
        'asp.net': 'ASP.NET',
        '.net core': '.NET Core',
        'laravel': 'Laravel',
        'rails': 'Rails',
        
        # BI/Data Tools
        'power bi': 'Power BI',
        'tableau': 'Tableau',
        'qlik': 'Qlik',
        'looker': 'Looker',
        'ssis': 'SSIS',
        'ssrs': 'SSRS',
        'cognos': 'Cognos',
        'microstrategy': 'MicroStrategy',
        'spotfire': 'Spotfire',
        'excel': 'Excel',
        'sql': 'SQL',
        'python': 'Python',
        # 'r ': 'R',  # Removed - causes false positives in profile_sub_type
        'sas': 'SAS',
        'spss': 'SPSS',
        
        # Tools/Others
        'docker': 'Docker',
        'kubernetes': 'Kubernetes',
        'aws': 'AWS',
        'azure': 'Azure',
        'gcp': 'GCP',
        'terraform': 'Terraform',
        'jenkins': 'Jenkins',
        'git': 'Git'
    }
    
    detected_techs = []
    for keyword, tech_name in technology_map.items():
        if keyword in text_lower and tech_name not in detected_techs:
            detected_techs.append(tech_name)
    
    # Return top 3 most relevant
    return detected_techs[:3]


def looks_like_name(text: str) -> bool:
    """
    Detect if input looks like a candidate name.
    Uses comprehensive technical term detection to avoid false positives.
    """
    if not text or not text.strip():
        return False
    
    text_lower = normalize_text(text)
    
    # FIRST: Check if it's a known technical term (comprehensive check)
    # This prevents technical queries from being misidentified as names
    
    # 1. Check against MASTER_SKILL_TO_PROFILE_AND_SCORE keys
    if MASTER_SKILL_TO_PROFILE_AND_SCORE:
        if any(skill in text_lower for skill in MASTER_SKILL_TO_PROFILE_AND_SCORE.keys()):
            return False  # It's a skill, not a name
    
    # 2. Check against additional skills (power platform, ios, django, etc.)
    additional_skill_keywords = [
        "power platform", "powerplatform", "power apps", "power automate",
        "power bi", "dataverse", "power fx", "canvas app", "model-driven app",
        "ios", "django", "flask", "spring boot", "springboot"
    ]
    if any(kw in text_lower for kw in additional_skill_keywords):
        return False  # It's a technical term, not a name
    
    # 3. Check against PROFILE_TYPE_RULES keywords (from profile_type_utils)
    try:
        from profile_type_utils import PROFILE_TYPE_RULES
        for profile_type, keywords in PROFILE_TYPE_RULES:
            # Check if any keyword from this profile type matches
            if any(kw in text_lower for kw in keywords):
                return False  # It's a technical term, not a name
    except ImportError:
        # If import fails, continue with other checks
        pass
    
    # 4. Check against original tech keywords (for backward compatibility)
    tech_keywords = [
        "java", "python", "react", "sql", "developer", "engineer", 
        "manager", "analyst", ".net", "asp", "c#", "javascript",
        # Common short skills that look like names
        "css", "html", "xml", "json", "api", "ui", "ux", "jsx", "tsx",
        "aws", "gcp", "api", "rest", "soap", "graphql", "http", "https",
        "sdk", "ide", "cli", "gui", "sso", "oauth", "jwt", "rpc",
        "dns", "tcp", "udp", "ip", "url", "uri", "csv", "pdf", "xls"
    ]
    if any(kw in text_lower for kw in tech_keywords):
        return False  # It's a technical term, not a name
    
    # THEN: Check name pattern (only if no technical terms found)
    words = text.strip().split()
    if 1 <= len(words) <= 4:
        if all(word.isalpha() for word in words):
            # If we reach here, it passed all technical term checks
            # and matches name pattern, so it might be a name
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


# ============================================================================
# Query Understanding Engine Helper Functions
# ============================================================================

# Explicit subroles that should be detected (ONLY these three)
EXPLICIT_SUBROLES = [
    "Frontend Developer", "Front-End Developer", "Front End Developer",
    "Backend Developer", "Back-End Developer", "Back End Developer",
    "Full Stack Developer", "Full-Stack Developer", "Fullstack Developer"
]

# Stop words for partial query cleaning
STOP_WORDS = [
    "get", "the", "profile", "of", "show", "candidates", "for", "find",
    "search", "look", "with", "having", "who", "are", "is", "a", "an",
    "in", "at", "on", "from", "to", "and", "or", "not"
]

# Irrelevant query patterns (non-technical phrases)
IRRELEVANT_PATTERNS = [
    r'\bcommunication\b',
    r'\bfor the team\b',
    r'\bin the company\b',
    r'\bto handle\b',
    r'\bability to\b',
    r'\bgood\b.*\bcommunication\b',
    r'\bteam\s+player\b',
    r'\bwork\s+ethic\b'
]

# Technology + Role pattern keywords
ROLE_TITLE_KEYWORDS = ['developer', 'engineer', 'programmer', 'architect']


def clean_partial_query(query: str) -> str:
    """
    Clean partial queries by removing stop words.
    Example: "get the profile of power platform" → "power platform"
    """
    if not query:
        return query
    
    # Remove stop words
    words = query.split()
    cleaned_words = [w for w in words if w.lower() not in STOP_WORDS]
    
    # Rejoin
    cleaned = ' '.join(cleaned_words)
    
    # Remove extra spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned if cleaned else query


def is_irrelevant_query(query: str) -> bool:
    """
    Check if query is irrelevant (non-technical).
    Uses comprehensive technical term detection including TECH_SKILLS, 
    MASTER_SKILL_TO_PROFILE_AND_SCORE, PROFILE_TYPE_RULES, and domain keywords.
    """
    if not query:
        return True
    
    query_lower = normalize_text(query)
    
    # FIRST: Check against TECH_SKILLS (comprehensive skill list from skill_extractor.py)
    # This includes 6000+ skills including "saas", "paas", "iaas", etc.
    # Direct check - very fast set membership
    if any(skill.lower() in query_lower for skill in TECH_SKILLS):
        return False  # Has technical skill, not irrelevant
    
    # SECOND: Check MASTER_SKILL_TO_PROFILE_AND_SCORE
    if MASTER_SKILL_TO_PROFILE_AND_SCORE:
        if any(skill in query_lower for skill in MASTER_SKILL_TO_PROFILE_AND_SCORE.keys()):
            return False  # Has master skill, not irrelevant
    
    # THIRD: Check additional skills (power platform, ios, django, etc.)
    additional_skills = [
        "power platform", "powerplatform", "power apps", "power automate",
        "power bi", "dataverse", "power fx", "canvas app", "model-driven app",
        "ios", "django", "flask", "spring boot", "springboot"
    ]
    if any(skill in query_lower for skill in additional_skills):
        return False  # Has additional skill, not irrelevant
    
    # FOURTH: Check PROFILE_TYPE_RULES keywords (comprehensive profile type keywords)
    try:
        from profile_type_utils import PROFILE_TYPE_RULES
        for profile_type, keywords in PROFILE_TYPE_RULES:
            # Check if any keyword from this profile type matches
            if any(kw in query_lower for kw in keywords):
                return False  # Has profile type keyword, not irrelevant
    except ImportError:
        logger.warning("profile_type_utils.PROFILE_TYPE_RULES not available, skipping PROFILE_TYPE_RULES check")
    
    # FIFTH: Check domain keywords (SaaS, PaaS, IaaS, etc. from domain_extraction)
    try:
        from domain_extraction import CLOUD_KEYWORDS, IT_KEYWORDS
        if any(kw in query_lower for kw in CLOUD_KEYWORDS):
            return False  # Has cloud-related term, not irrelevant
        if any(kw in query_lower for kw in IT_KEYWORDS):
            return False  # Has IT-related term, not irrelevant
    except ImportError:
        logger.warning("domain_extraction keywords not available, skipping domain keyword check")
    
    # SIXTH: Check common technical acronyms and terms
    common_tech_terms = [
        'saas', 'paas', 'iaas', 'api', 'sdk', 'sso', 'oauth', 'rest', 'graphql',
        'crm', 'erp', 'scrum', 'agile', 'kanban', 'ci/cd', 'cicd', 'sdlc',
        'microservices', 'serverless', 'containerization', 'orchestration',
        'devops', 'mlops', 'dataops', 'gitops'
    ]
    if any(term in query_lower for term in common_tech_terms):
        return False  # Has common tech term, not irrelevant
    
    # SEVENTH: Check original tech keywords (backward compatibility)
    tech_keywords = [
        'java', 'python', 'sql', 'javascript', 'react', 'angular', 'vue',
        'node', 'c#', '.net', 'asp.net', 'php', 'ruby', 'go', 'rust',
        'developer', 'engineer', 'programmer', 'architect', 'analyst',
        'database', 'mysql', 'postgresql', 'mongodb', 'redis',
        'docker', 'kubernetes', 'aws', 'azure', 'gcp',
        'django', 'flask', 'spring', 'express', 'laravel',
        'power platform', 'salesforce', 'sap', 'devops'
    ]
    if any(kw in query_lower for kw in tech_keywords):
        return False  # Has tech keyword, not irrelevant
    
    # EIGHTH: Check for irrelevant patterns (non-technical phrases)
    # Only mark as irrelevant if it matches irrelevant patterns AND has no technical terms
    for pattern in IRRELEVANT_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            # Double-check: if it matches irrelevant pattern, it's likely irrelevant
            return True  # Matches irrelevant pattern
    
    # If we reach here, no technical terms found and no irrelevant patterns matched
    # This means it's a generic query that might still be valid (e.g., "developer")
    # So we'll be lenient and NOT mark it as irrelevant
    # Only mark as irrelevant if it's clearly non-technical
    return False  # Default to NOT irrelevant (let skill/role detection handle it)


def detect_explicit_subrole(query: str) -> Optional[str]:
    """
    Detect explicit subroles: Frontend Developer, Backend Developer, Full Stack Developer.
    Returns the exact subrole string if found, None otherwise.
    """
    query_normalized = normalize_text(query)
    
    for subrole in EXPLICIT_SUBROLES:
        subrole_normalized = normalize_text(subrole)
        if subrole_normalized in query_normalized:
            # Return the canonical form (first in list)
            if "frontend" in subrole_normalized or "front-end" in subrole_normalized or "front end" in subrole_normalized:
                return "Frontend Developer"
            elif "backend" in subrole_normalized or "back-end" in subrole_normalized or "back end" in subrole_normalized:
                return "Backend Developer"
            elif "full stack" in subrole_normalized or "fullstack" in subrole_normalized or "full-stack" in subrole_normalized:
                return "Full Stack Developer"
    
    return None


def detect_tech_role_pattern(query: str) -> Optional[Dict[str, str]]:
    """
    Detect Technology + Role pattern: "java Developer", "python Engineer", etc.
    
    Returns:
        {
            'main_role': 'Software Engineer',
            'profile_type': 'Java',
            'detection_type': 'tech_role_pattern'
        }
    
    NOTE: Does NOT set subrole_type (only explicit subroles do that)
    """
    query_normalized = normalize_text(query)
    
    # Check each master skill
    for skill_key, mapping in MASTER_SKILL_TO_PROFILE_AND_SCORE.items():
        profile_type = mapping['profile_type']
        
        # Check for pattern: skill + role_title
        for role_title in ROLE_TITLE_KEYWORDS:
            pattern = f"{skill_key} {role_title}"
            if pattern in query_normalized:
                # Construct main_role (default to Software Engineer)
                main_role = "Software Engineer"
                
                # Try to get main_role from SUBROLE_TO_MAIN_ROLE if available
                if SUBROLE_TO_MAIN_ROLE:
                    # Create a potential subrole name
                    potential_subrole = f"{profile_type} Developer"
                    main_role_from_map = SUBROLE_TO_MAIN_ROLE.get(potential_subrole.lower())
                    if main_role_from_map:
                        main_role = main_role_from_map
                
                return {
                    'main_role': main_role,
                    'profile_type': profile_type,
                    'detection_type': 'tech_role_pattern'
                }
    
    return None


def detect_role_from_hierarchy(query: str) -> Optional[str]:
    """
    Detect role using role_extract.py detect_all_roles().
    Returns role_type if found.
    """
    try:
        from role_extract import detect_all_roles
        roles = detect_all_roles(query)
        if roles:
            # Return the highest priority role (first in sorted list)
            return roles[0][1]  # (priority, role_type, subrole)
    except Exception as e:
        logger.warning(f"Error detecting role from hierarchy: {e}")
    
    return None


def detect_role_from_processor_table(query: str) -> Optional[str]:
    """
    Detect role using role_processor table.
    Returns normalized role if found.
    """
    try:
        from role_processor import RoleProcessor
        with RoleProcessor(config=ATSConfig.get_mysql_config()) as rp:
            # Try to find normalized role
            normalized = rp.get_normalized_role(query)
            if normalized:
                return normalized
            
            # Try with cleaned query
            cleaned = clean_partial_query(query)
            if cleaned != query:
                normalized = rp.get_normalized_role(cleaned)
                if normalized:
                    return normalized
    except Exception as e:
        logger.warning(f"Error detecting role from processor table: {e}")
    
    return None


def extract_additional_skills(term: str) -> List[Dict[str, str]]:
    """
    Extract additional skills not in MASTER_SKILL_TO_PROFILE_AND_SCORE.
    Handles: "power platform", "ios", "django", "saas", "paas", "iaas", etc.
    """
    additional_skills = []
    term_lower = normalize_text(term)
    
    # Power Platform variations
    if 'power platform' in term_lower or 'powerplatform' in term_lower:
        additional_skills.append({
            'skill': 'power platform',
            'profile_type': 'Microsoft Power Platform',
            'score_column': 'microsoft_power_platform_score'
        })
    
    # iOS
    if term_lower in ['ios', 'ios developer', 'ios development']:
        additional_skills.append({
            'skill': 'ios',
            'profile_type': 'Mobile Development',
            'score_column': 'mobile_development_score'
        })
    
    # Django
    if 'django' in term_lower:
        additional_skills.append({
            'skill': 'django',
            'profile_type': 'Python',
            'score_column': 'python_score'
        })
    
    # Flask
    if 'flask' in term_lower:
        additional_skills.append({
            'skill': 'flask',
            'profile_type': 'Python',
            'score_column': 'python_score'
        })
    
    # Spring Boot
    if 'spring boot' in term_lower or 'springboot' in term_lower:
        additional_skills.append({
            'skill': 'spring boot',
            'profile_type': 'Java',
            'score_column': 'java_score'
        })
    
    # Cloud service models (SaaS, PaaS, IaaS)
    if term_lower in ['saas', 'software as a service']:
        additional_skills.append({
            'skill': 'saas',
            'profile_type': 'Cloud / Infra',
            'score_column': 'cloud_infra_score'
        })
    
    if term_lower in ['paas', 'platform as a service']:
        additional_skills.append({
            'skill': 'paas',
            'profile_type': 'Cloud / Infra',
            'score_column': 'cloud_infra_score'
        })
    
    if term_lower in ['iaas', 'infrastructure as a service']:
        additional_skills.append({
            'skill': 'iaas',
            'profile_type': 'Cloud / Infra',
            'score_column': 'cloud_infra_score'
        })
    
    # Check TECH_SKILLS for other skills not already detected
    # Only check if skill wasn't already added above
    already_detected_skills = {s['skill'].lower() for s in additional_skills}
    
    if term_lower not in already_detected_skills:
        # Check if term is in TECH_SKILLS but not in master list
        if term_lower in TECH_SKILLS and term_lower not in MASTER_SKILL_TO_PROFILE_AND_SCORE:
            # Try to infer profile type from PROFILE_TYPE_RULES
            try:
                    from profile_type_utils import PROFILE_TYPE_RULES
                    # Map profile type to score column (only valid columns that exist in database)
                    profile_to_score = {
                        'Java': 'java_score',
                        '.Net': 'dotnet_score',
                        'Python': 'python_score',
                        'JavaScript': 'javascript_score',
                        'Full Stack': 'fullstack_score',
                        'DevOps': 'devops_score',
                        'Data Engineering': 'data_engineering_score',
                        'Data Science': 'data_science_score',
                        'SAP': 'sap_score',
                        'ERP': 'erp_score',
                        'Cloud / Infra': 'cloud_infra_score',
                        'Database': 'database_score',
                        'Testing / QA': 'testing_qa_score',
                        'Mobile Development': 'mobile_development_score',
                        'Salesforce': 'salesforce_score',
                        'Microsoft Power Platform': 'microsoft_power_platform_score',
                        'Business Intelligence (BI)': 'business_intelligence_score',
                        'RPA': 'rpa_score',
                        'Cyber Security': 'cyber_security_score',
                        'Low Code / No Code': 'low_code_no_code_score',
                        'Integration / APIs': 'integration_apis_score',
                        'UI/UX': 'ui_ux_score',
                        'Support': 'support_score',
                        'Business Development': 'business_development_score'
                    }
                    
                    # Map profile types without score columns to related profile types that have score columns
                    profile_type_fallback = {
                        'Dart': 'Mobile Development',  # Dart -> Mobile Development
                        'Flutter': 'Mobile Development',  # Flutter -> Mobile Development
                        'Objective-C': 'Mobile Development',  # Objective-C -> Mobile Development
                        'Go / Golang': 'Python',  # Go -> Python (closest match)
                        'Ruby': 'Python',  # Ruby -> Python (closest match)
                        'PHP': 'JavaScript',  # PHP -> JavaScript (web development)
                        'Rust': 'Python',  # Rust -> Python (closest match)
                        'Scala': 'Java',  # Scala -> Java (JVM-based)
                        'C/C++': 'Java',  # C/C++ -> Java (systems programming)
                    }
                    
                    for profile_type, keywords in PROFILE_TYPE_RULES:
                        if term_lower in keywords:
                            # Try to get score column for this profile type
                            score_column = profile_to_score.get(profile_type)
                            
                            # If no score column, try fallback profile type
                            if not score_column:
                                fallback_profile = profile_type_fallback.get(profile_type)
                                if fallback_profile:
                                    score_column = profile_to_score.get(fallback_profile)
                                    # Use fallback profile type for score, but keep original for profile_type LIKE
                                    additional_skills.append({
                                        'skill': term_lower,
                                        'profile_type': profile_type,  # Original profile type for LIKE filter
                                        'score_column': score_column  # Fallback score column
                                    })
                                    break
                            
                            # If we have a valid score column (either direct or from fallback)
                            if score_column:
                                additional_skills.append({
                                    'skill': term_lower,
                                    'profile_type': profile_type,
                                    'score_column': score_column
                                })
                                break  # Found match, stop searching
                            else:
                                # Profile type found but no score column and no fallback
                                # Add skill but without score column (will use profile_type LIKE only)
                                additional_skills.append({
                                    'skill': term_lower,
                                    'profile_type': profile_type,
                                    'score_column': None  # No score column available
                                })
                                break  # Found match, stop searching
            except ImportError:
                pass
    
    return additional_skills


def extract_skills_from_query_enhanced(query: str, parsed_boolean: Optional[Dict] = None) -> List[Dict[str, str]]:
    """
    Extract skills from query with boolean query support.
    Handles boolean queries by extracting skills from each term.
    """
    all_skills = []
    
    # If boolean structure provided, extract from each term
    if parsed_boolean and parsed_boolean.get('and_terms'):
        all_terms = []
        for or_group in parsed_boolean['and_terms']:
            all_terms.extend(or_group)
    else:
        # No boolean structure, treat whole query as single term
        all_terms = [query]
    
    # Extract skills from each term
    for term in all_terms:
        term_clean = normalize_text(term)
        
        # Use existing extract_master_skills function
        detected_skills = extract_master_skills(term)
        all_skills.extend(detected_skills)
        
        # Also check for additional skills not in MASTER_SKILL dict
        additional_skills = extract_additional_skills(term_clean)
        all_skills.extend(additional_skills)
    
    # Deduplicate skills (keep first occurrence)
    seen = set()
    unique_skills = []
    for skill in all_skills:
        skill_key = skill.get('skill', '').lower()
        if skill_key not in seen:
            seen.add(skill_key)
            unique_skills.append(skill)
    
    return unique_skills


def detect_profile_type_priority(role_type: Optional[str], skills: List[Dict], explicit_subrole: Optional[str]) -> Optional[str]:
    """
    Detect profile type with priority:
    1. From role (if role has associated profile_type)
    2. From skills (most common profile_type)
    3. From explicit subrole (if any)
    """
    profile_type = None
    
    # Priority 1: From role (if role has associated profile_type)
    # This would need to be implemented based on your role mapping
    # For now, we'll skip this and go to skills
    
    # Priority 2: From skills (most common profile_type)
    if not profile_type and skills:
        profile_types = [s.get('profile_type') for s in skills if s.get('profile_type')]
        if profile_types:
            # Get most common profile_type
            profile_type_counter = Counter(profile_types)
            profile_type = profile_type_counter.most_common(1)[0][0]
    
    # Priority 3: From explicit subrole (if any)
    # Explicit subroles don't directly map to profile_type
    # But we can infer from context
    # For now, skip this
    
    return profile_type


def extract_locations(query: str) -> List[str]:
    """
    Extract location patterns from query.
    Example: "developer in hyderabad" → ["hyderabad"]
    """
    location_pattern = r'\b(?:in|at|from|near)\s+([a-z\s]+?)(?:\s|$|AND|OR)'
    locations = re.findall(location_pattern, normalize_text(query))
    return [loc.strip() for loc in locations if loc.strip()]


@app.route('/api/profileRankingByJD', methods=['POST'])
def profile_ranking_by_jd():
    """
    Profile ranking by job description.
    
    Accepts job description and ranks candidates against it.
    Returns ranked list of candidates with match scores.
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json() or {}
        
        # Get job_id (optional - for tracking)
        job_id = data.get('job_id', f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Get job metadata from database if job_id is provided
        job_metadata = None
        if job_id and job_id.startswith('job_') == False:  # If it's not an auto-generated ID
            try:
                with create_ats_database() as db:
                    job_metadata = db.get_job_description(job_id)
            except Exception as e:
                logger.warning(f"Could not fetch job metadata for job_id {job_id}: {e}")
        
        # Get job description - use provided text or from metadata
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
            
            # Convert extracted skills to list if needed
            if isinstance(extracted_skills, str):
                extracted_skills = [s.strip() for s in extracted_skills.split(',') if s.strip()]
            elif not isinstance(extracted_skills, list):
                extracted_skills = []
            
            # Filter out non-technical words that might have been incorrectly extracted
            # Words like "dashboards", "datasets" are tasks/features, not skills
            # Also filter out single-letter skills like "R" that might be false positives
            non_skill_words = {'dashboards', 'dashboard', 'datasets', 'dataset', 'insights', 'insight', 
                             'strategies', 'strategy', 'decisions', 'decision', 'teams', 'team',
                             'r', 'R', 'accuracy', 'documentation', 'reporting', 'communication',
                             'leadership', 'teamwork', 'collaboration', 'problem solving', 'analytical thinking'}  # Filter out non-technical words and soft skills
            # Filter out non-skill words and single-letter skills (especially "R" which is often false positive)
            extracted_skills = [s for s in extracted_skills 
                              if s.strip().lower() not in non_skill_words and len(s.strip()) > 1]
            
            # If BI/Data Analyst keywords detected, add common BI skills
            job_desc_lower = job_description.lower()
            bi_keywords_detected = any(kw in job_desc_lower for kw in ['dashboard', 'dataset', 'data visualization', 
                                                                      'data-driven', 'insights', 'business intelligence', 
                                                                      'bi', 'reporting', 'analytics', 'data analysis',
                                                                      'data analyst', 'business analyst'])
            
            if bi_keywords_detected:
                # Common BI/Data Analyst skills to check for/add
                # NOTE: "R" is EXCLUDED - it's handled separately with strict regex patterns to avoid false positives
                bi_skills_to_check = ['SQL', 'Excel', 'Power BI', 'Tableau', 'Python', 'SAS', 'SPSS', 
                                     'data visualization', 'statistical analysis', 'data mining', 'ETL']
                
                # First, check if these skills are explicitly mentioned in the job description
                # Use word boundary matching to avoid false positives (e.g., "reporting" matching "r")
                import re
                for skill in bi_skills_to_check:
                    skill_lower = skill.lower()
                    # Use word boundary regex to ensure exact match, not substring
                    # Escape special characters in skill name for regex
                    skill_pattern = r'\b' + re.escape(skill_lower) + r'\b'
                    if re.search(skill_pattern, job_desc_lower, re.IGNORECASE):
                        # Add to extracted_skills if not already present (case-insensitive check)
                        if not any(s.lower() == skill_lower for s in extracted_skills):
                            extracted_skills.append(skill)
                
                # Remove "R" completely - it's often a false positive from words like "reporting", "Power BI", etc.
                # We don't add "R" even if explicitly mentioned to avoid confusion
                extracted_skills = [s for s in extracted_skills 
                                  if s.strip().lower() not in ['r', 'R']]
                
                # If very few or no real skills were extracted, add default BI skills for Data Analyst roles
                # This helps when job description is generic and doesn't mention specific tools
                # Filter out single-letter false positives
                real_skills = [s for s in extracted_skills if len(s.strip()) > 1 and s.lower() not in {'r'}]
                if len(real_skills) < 3:
                    # Default essential BI skills for Data Analyst (always add these for BI roles)
                    default_bi_skills = ['SQL', 'Excel', 'data visualization']
                    for skill in default_bi_skills:
                        if not any(s.lower() == skill.lower() for s in extracted_skills):
                            extracted_skills.append(skill)
            
            # Convert extracted skills to string if needed
            if not required_skills:
                # Use first 10 skills as required/primary
                required_skills = ', '.join(extracted_skills[:10]) if extracted_skills else ''
            
            # ALWAYS extract secondary skills from remaining skills if preferred_skills not provided
            if not preferred_skills and extracted_skills:
                if len(extracted_skills) > 10:
                    # Use skills after the first 10 as preferred/secondary
                    preferred_skills = ', '.join(extracted_skills[10:])
                elif len(extracted_skills) > 5:
                    # If we have 6-10 skills, split: first 5 primary, rest secondary
                    preferred_skills = ', '.join(extracted_skills[5:])
                else:
                    # If we have 5 or fewer skills, use last 2-3 as secondary (if any)
                    if len(extracted_skills) > 2:
                        preferred_skills = ', '.join(extracted_skills[2:])
                    else:
                        preferred_skills = ''
            
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
        
        # Extract job metadata (role, sub_role, profile_type, profile_sub_type, primary_skills) using Python only (NO AI)
        extracted_job_metadata = {}
        logger.info("Extracting job metadata using Python-only extraction (no AI)...")
        
        # Build searchable text from job description and skills
        searchable_text = job_description
        if required_skills_list:
            searchable_text += ' ' + ', '.join(required_skills_list)
        if preferred_skills_list:
            searchable_text += ' ' + ', '.join(preferred_skills_list)
        
        # 1. Extract ROLE using role_extract.py
        job_title = data.get('job_title', '')
        if job_title:
            extracted_job_metadata['role'] = job_title
        else:
            # Use role_extract.py to detect role from job description
            detected_role = detect_role_only(searchable_text)
            if detected_role:
                extracted_job_metadata['role'] = detected_role
            else:
                # Default fallback
                extracted_job_metadata['role'] = 'Software Engineer'
        
        # 2. Extract SUB_ROLE using role_extract.py
        # Use detect_role_subrole to get both role and subrole, then extract subrole
        role_subrole_result = detect_role_subrole(searchable_text)
        if role_subrole_result:
            detected_role_from_pair, detected_subrole = role_subrole_result
            # For Data Analyst / BI roles, sub_role doesn't apply
            searchable_lower_for_subrole = searchable_text.lower()
            if 'data analyst' in searchable_lower_for_subrole or 'business analyst' in searchable_lower_for_subrole or 'business intelligence' in searchable_lower_for_subrole:
                extracted_job_metadata['sub_role'] = None
            else:
                extracted_job_metadata['sub_role'] = detected_subrole
        else:
            # If no subrole detected, check for Frontend/Backend keywords as fallback
            searchable_lower_for_subrole = searchable_text.lower()
            if 'data analyst' in searchable_lower_for_subrole or 'business analyst' in searchable_lower_for_subrole or 'business intelligence' in searchable_lower_for_subrole:
                extracted_job_metadata['sub_role'] = None
            else:
                # Default to Backend if no clear indication
                extracted_job_metadata['sub_role'] = 'Backend'
        
        # 3. Extract PROFILE_TYPE (Java, Python, .Net, JavaScript, etc.)
        # First check for Data Analyst / BI specific keywords
        searchable_lower = searchable_text.lower()
        profile_type_detected = None
        
        # Check if this is a Data Analyst role first
        is_data_analyst_role = ('data analyst' in searchable_lower or 
                                'business analyst' in searchable_lower or 
                                'business intelligence' in searchable_lower or
                                'bi analyst' in searchable_lower)
        
        # Check for BI/Data Science keywords
        bi_keywords = ['dashboard', 'dashboards', 'dataset', 'datasets', 'data visualization', 
                      'data-driven', 'insights', 'business intelligence', 'bi', 'reporting',
                      'analytics', 'data analysis']
        
        # Strong Data Science indicators (ML/AI specific)
        data_science_keywords = ['machine learning', 'deep learning', 'ml engineer', 'ai engineer',
                                'artificial intelligence', 'neural network', 'tensorflow', 'pytorch',
                                'computer vision', 'nlp', 'natural language processing']
        
        # Always try to detect technology-based profile_type first (Java, Python, .Net, etc.)
        # Use standard inference from skills and job description
        inferred_profile_types = infer_profile_type_from_requirements(required_skills_list, job_description)
        if inferred_profile_types:
            profile_type_detected = inferred_profile_types[0]
        else:
            # Fallback: use detect_profile_types_from_text
            detected_profiles = detect_profile_types_from_text(job_description, ', '.join(required_skills_list) if required_skills_list else '')
            if detected_profiles:
                profile_type_detected = detected_profiles[0]
            else:
                # Only use BI/Data Science if no technology profile detected
                if is_data_analyst_role or any(keyword in searchable_lower for keyword in bi_keywords):
                    # Check if it's more Data Science (strong ML/AI indicators)
                    if any(kw in searchable_lower for kw in data_science_keywords):
                        profile_type_detected = 'Data Science'
                    else:
                        profile_type_detected = 'Business Intelligence (BI)'
                else:
                    profile_type_detected = 'Generalist'
        
        extracted_job_metadata['profile_type'] = profile_type_detected
        
        # 4. Extract PROFILE_SUB_TYPE (specific technologies: React, SQL, Angular, etc.)
        # Take the second highest (second technology) if available, otherwise first
        specific_techs = extract_specific_technologies(searchable_text, ', '.join(required_skills_list) if required_skills_list else '')
        if specific_techs:
            # Remove "R" from profile_sub_type (it's often a false positive)
            specific_techs = [tech for tech in specific_techs if tech.strip() not in ['R', 'r']]
            if len(specific_techs) >= 2:
                # Take second highest (second technology)
                extracted_job_metadata['profile_sub_type'] = specific_techs[1]
            elif len(specific_techs) == 1:
                # If only one technology, use it
                extracted_job_metadata['profile_sub_type'] = specific_techs[0]
            else:
                extracted_job_metadata['profile_sub_type'] = None
        else:
            extracted_job_metadata['profile_sub_type'] = None
        
        # 5. Extract PRIMARY_SKILLS (use required_skills_list)
        extracted_job_metadata['primary_skills'] = required_skills_list[:15] if required_skills_list else []
        
        logger.info(f"Extracted job metadata (Python-only): {extracted_job_metadata}")
        
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
                'education_required': job_requirements['education_required'],
                # Add extracted metadata
                'role': extracted_job_metadata.get('role', ''),
                'sub_role': extracted_job_metadata.get('sub_role', ''),
                'profile_type': extracted_job_metadata.get('profile_type', ''),
                'profile_sub_type': extracted_job_metadata.get('profile_sub_type', ''),
                'primary_skills': extracted_job_metadata.get('primary_skills', [])
            }
            
            # Try to insert (will update if job_id already exists)
            try:
                db.insert_job_description(jd_data, jd_embedding)
                logger.info(f"Successfully stored job description with metadata for job_id: {job_id}")
                
                # Also insert into job_description table (singular)
                try:
                    # Prepare primary_skills - use from extracted_job_metadata (already a list from Python extraction)
                    primary_skills_for_desc = extracted_job_metadata.get('primary_skills', [])
                    if isinstance(primary_skills_for_desc, str):
                        primary_skills_for_desc = [s.strip() for s in primary_skills_for_desc.split(',') if s.strip()]
                    elif not isinstance(primary_skills_for_desc, list):
                        primary_skills_for_desc = []
                    
                    # Prepare secondary_skills from preferred_skills_list
                    # preferred_skills_list is already extracted above (line ~2493-2498)
                    secondary_skills_for_desc = preferred_skills_list.copy() if preferred_skills_list else []
                    
                    # If still empty, try to extract from job description text directly
                    if not secondary_skills_for_desc:
                        logger.info("preferred_skills_list is empty, attempting to extract secondary skills from job description...")
                        from resume_parser import extract_skills_from_text
                        all_extracted_skills = extract_skills_from_text(job_description)
                        if isinstance(all_extracted_skills, list) and len(all_extracted_skills) > 10:
                            # Use skills after first 10 as secondary
                            secondary_skills_for_desc = all_extracted_skills[10:]
                            logger.info(f"Extracted {len(secondary_skills_for_desc)} secondary skills from job description")
                        elif isinstance(all_extracted_skills, str):
                            skills_list = [s.strip() for s in all_extracted_skills.split(',') if s.strip()]
                            if len(skills_list) > 10:
                                secondary_skills_for_desc = skills_list[10:]
                                logger.info(f"Extracted {len(secondary_skills_for_desc)} secondary skills from job description")
                    
                    logger.info(f"Final secondary_skills_for_desc: {len(secondary_skills_for_desc)} skills - {secondary_skills_for_desc[:5] if secondary_skills_for_desc else 'EMPTY'}")
                    
                    job_desc_metadata = {
                        'role': extracted_job_metadata.get('role', ''),
                        'sub_role': extracted_job_metadata.get('sub_role', ''),
                        'profile_type': extracted_job_metadata.get('profile_type', ''),
                        'profile_sub_type': extracted_job_metadata.get('profile_sub_type', ''),
                        'primary_skills': primary_skills_for_desc,
                        'secondary_skills': secondary_skills_for_desc
                    }
                    logger.info(f"Attempting to insert into job_description table with metadata:")
                    logger.info(f"  - primary_skills: {len(primary_skills_for_desc)} skills")
                    logger.info(f"  - secondary_skills: {len(secondary_skills_for_desc)} skills")
                    success = db.insert_into_job_description(job_desc_metadata)
                    if success:
                        logger.info(f"Successfully inserted into job_description table for job_id: {job_id}")
                    else:
                        logger.warning(f"Failed to insert into job_description table (returned False)")
                except Exception as e:
                    logger.error(f"Error inserting into job_description table: {e}")
                    logger.error(traceback.format_exc())
            except Exception as e:
                logger.error(f"Error storing job description: {e}")
        
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
                'extracted_job_metadata': {
                    'role': extracted_job_metadata.get('role', ''),
                    'sub_role': extracted_job_metadata.get('sub_role', ''),
                    'profile_type': extracted_job_metadata.get('profile_type', ''),
                    'profile_sub_type': extracted_job_metadata.get('profile_sub_type', ''),
                    'primary_skills': extracted_job_metadata.get('primary_skills', []),
                    'secondary_skills': preferred_skills_list
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
            'extracted_job_metadata': {
                'role': extracted_job_metadata.get('role', ''),
                'sub_role': extracted_job_metadata.get('sub_role', ''),
                'profile_type': extracted_job_metadata.get('profile_type', ''),
                'profile_sub_type': extracted_job_metadata.get('profile_sub_type', ''),
                'primary_skills': extracted_job_metadata.get('primary_skills', []),
                'secondary_skills': preferred_skills_list
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
        
        # Extract role, sub_role, profile_type, and profile_sub_type from job description
        extracted_job_metadata = {}
        job_description_text = job_requirements.get('job_description', '')
        if not job_description_text:
            # Try to get from job_requirements as text
            job_description_text = str(job_requirements.get('job_description', ''))
        
        # Build searchable text from job requirements for detection
        searchable_text = job_description_text
        if required_skills:
            searchable_text += ' ' + (required_skills if isinstance(required_skills, str) else ', '.join(required_skills))
        if preferred_skills:
            searchable_text += ' ' + (preferred_skills if isinstance(preferred_skills, str) else ', '.join(preferred_skills))
        
        # 1. Extract ROLE using role_extract.py
        job_title = job_requirements.get('job_title', '')
        if job_title:
            extracted_job_metadata['role'] = job_title
        else:
            # Use role_extract.py to detect role from job description
            detected_role = detect_role_only(searchable_text)
            if detected_role:
                extracted_job_metadata['role'] = detected_role
            else:
                # Default fallback
                extracted_job_metadata['role'] = 'Software Engineer'
        
        # 2. Extract SUB_ROLE using role_extract.py
        # Use detect_role_subrole to get both role and subrole, then extract subrole
        role_subrole_result = detect_role_subrole(searchable_text)
        if role_subrole_result:
            detected_role_from_pair, detected_subrole = role_subrole_result
            # Map subrole to standard format: Backend, Frontend, or Full Stack
            if detected_subrole:
                # Normalize subrole names to standard format
                subrole_lower = detected_subrole.lower()
                if 'backend' in subrole_lower or 'back-end' in subrole_lower:
                    extracted_job_metadata['sub_role'] = 'Backend'
                elif 'frontend' in subrole_lower or 'front-end' in subrole_lower or 'ui' in subrole_lower:
                    extracted_job_metadata['sub_role'] = 'Frontend'
                elif 'full stack' in subrole_lower or 'fullstack' in subrole_lower:
                    extracted_job_metadata['sub_role'] = 'Full Stack'
                else:
                    # Use detected subrole as-is, or default to Backend
                    extracted_job_metadata['sub_role'] = 'Backend'
            else:
                extracted_job_metadata['sub_role'] = 'Backend'
        else:
            # If no subrole detected, use keyword matching as fallback
            searchable_lower_for_subrole = searchable_text.lower()
            # Check for Frontend/Backend keywords
            frontend_keywords = ['frontend', 'front-end', 'react', 'angular', 'vue', 'ui', 'javascript', 'html', 'css']
            backend_keywords = ['backend', 'back-end', 'api', 'database', 'sql', 'server', 'spring', 'django']
            
            frontend_count = sum(1 for kw in frontend_keywords if kw in searchable_lower_for_subrole)
            backend_count = sum(1 for kw in backend_keywords if kw in searchable_lower_for_subrole)
            
            if frontend_count > 0 and backend_count > 0:
                extracted_job_metadata['sub_role'] = 'Full Stack'
            elif frontend_count > 0:
                extracted_job_metadata['sub_role'] = 'Frontend'
            else:
                # Default to Backend if no clear indication
                extracted_job_metadata['sub_role'] = 'Backend'
        
        # 3. Extract PROFILE_TYPE (Java, Python, .Net, JavaScript, etc.)
        required_skills_list = required_skills.split(',') if isinstance(required_skills, str) else required_skills
        if not isinstance(required_skills_list, list):
            required_skills_list = []
        
        # First check for Data Analyst / BI specific keywords
        searchable_lower_profile = searchable_text.lower()
        profile_type_detected = None
        
        # Check if this is a Data Analyst role first
        is_data_analyst_role = ('data analyst' in searchable_lower_profile or 
                                'business analyst' in searchable_lower_profile or 
                                'business intelligence' in searchable_lower_profile or
                                'bi analyst' in searchable_lower_profile)
        
        # Check for BI/Data Science keywords
        bi_keywords = ['dashboard', 'dashboards', 'dataset', 'datasets', 'data visualization', 
                      'data-driven', 'insights', 'business intelligence', 'bi', 'reporting',
                      'analytics', 'data analysis']
        
        # Strong Data Science indicators (ML/AI specific)
        data_science_keywords = ['machine learning', 'deep learning', 'ml engineer', 'ai engineer',
                                'artificial intelligence', 'neural network', 'tensorflow', 'pytorch',
                                'computer vision', 'nlp', 'natural language processing']
        
        # Always try to detect technology-based profile_type first (Java, Python, .Net, etc.)
        # Use standard inference from skills and job description
        inferred_profile_types = infer_profile_type_from_requirements(required_skills_list, job_description_text)
        if inferred_profile_types:
            profile_type_detected = inferred_profile_types[0]
        else:
            # Fallback: use detect_profile_types_from_text
            detected_profiles = detect_profile_types_from_text(job_description_text, str(required_skills))
            if detected_profiles:
                profile_type_detected = detected_profiles[0]
            else:
                # Only use BI/Data Science if no technology profile detected
                if is_data_analyst_role or any(keyword in searchable_lower_profile for keyword in bi_keywords):
                    # Check if it's more Data Science (strong ML/AI indicators)
                    if any(kw in searchable_lower_profile for kw in data_science_keywords):
                        profile_type_detected = 'Data Science'
                    else:
                        profile_type_detected = 'Business Intelligence (BI)'
                else:
                    profile_type_detected = None
        
        extracted_job_metadata['profile_type'] = profile_type_detected
        
        # 4. Extract PROFILE_SUB_TYPE (specific technologies: React, SQL, Angular, etc.)
        # Take the second highest (second technology) if available, otherwise first
        specific_techs = extract_specific_technologies(searchable_text, required_skills)
        if specific_techs:
            # Remove "R" from profile_sub_type (it's often a false positive)
            specific_techs = [tech for tech in specific_techs if tech.strip() not in ['R', 'r']]
            if len(specific_techs) >= 2:
                # Take second highest (second technology)
                extracted_job_metadata['profile_sub_type'] = specific_techs[1]
            elif len(specific_techs) == 1:
                # If only one technology, use it
                extracted_job_metadata['profile_sub_type'] = specific_techs[0]
            else:
                extracted_job_metadata['profile_sub_type'] = None
        else:
            extracted_job_metadata['profile_sub_type'] = None
        
        # Extract primary_skills and secondary_skills for response
        if isinstance(required_skills, str):
            primary_skills_list = [s.strip() for s in required_skills.split(',') if s.strip()][:10]
        else:
            primary_skills_list = required_skills[:10] if isinstance(required_skills, list) else []
        
        if isinstance(preferred_skills, str):
            secondary_skills_list = [s.strip() for s in preferred_skills.split(',') if s.strip()]
        else:
            secondary_skills_list = preferred_skills if isinstance(preferred_skills, list) else []
        
        # Store extracted metadata in SQL database (job_description table)
        try:
            with create_ats_database() as db:
                metadata_to_store = {
                    'role': extracted_job_metadata.get('role'),
                    'sub_role': extracted_job_metadata.get('sub_role'),
                    'profile_type': extracted_job_metadata.get('profile_type'),
                    'profile_sub_type': extracted_job_metadata.get('profile_sub_type'),
                    'primary_skills': primary_skills_list,  # Pass as list, method will convert
                    'secondary_skills': secondary_skills_list  # Pass as list, method will convert
                }
                
                logger.info(f"Attempting to insert into job_description table with metadata:")
                logger.info(f"  - primary_skills: {len(primary_skills_list)} skills")
                logger.info(f"  - secondary_skills: {len(secondary_skills_list)} skills")
                # Insert into job_description table (singular)
                success = db.insert_into_job_description(metadata_to_store)
                if success:
                    logger.info(f"Successfully stored job metadata in job_description table with {len(primary_skills_list)} primary and {len(secondary_skills_list)} secondary skills")
                else:
                    logger.warning("Failed to store job metadata in job_description table (returned False)")
        except Exception as e:
            logger.error(f"Error storing job metadata in job_description table: {e}")
            logger.error(traceback.format_exc())
            # Don't fail the entire request if database insert fails
        
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
            'extracted_job_metadata': {
                'role': extracted_job_metadata.get('role'),
                'sub_role': extracted_job_metadata.get('sub_role'),
                'profile_type': extracted_job_metadata.get('profile_type'),
                'profile_sub_type': extracted_job_metadata.get('profile_sub_type'),
                'primary_skills': ', '.join(primary_skills_list) if primary_skills_list else '',
                'secondary_skills': ', '.join(secondary_skills_list) if secondary_skills_list else ''
            },
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

