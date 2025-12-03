"""
AI-powered Summary Extraction Module for Resume Parsing.

Extracts professional summary from resumes using AI.
Also provides search query processing for Pinecone-based resume search.
"""

import json
import logging
import os
from typing import Any, Optional, Dict, List

logger = logging.getLogger(__name__)

# AI prompt specifically for summary extraction
SUMMARY_EXTRACTION_PROMPT = """🧠 ROLE / PERSONA

You are an Expert Resume Parser specialized in extracting professional summaries from unstructured resume text.

Your only task is to create and return a concise professional summary in JSON format.

🎯 EXTRACTION RULES

1. summary – Provide a concise 2–3 line professional summary describing:
   - Experience in years
   - Domain focus
   - Technical strengths

Rules for Extraction:
- Look for existing summary sections (Summary, Professional Summary, Objective, Profile)
- If no summary exists, create one based on experience, skills, and domain
- Focus on quantifiable professional traits
- Avoid generic phrases like "Hardworking individual" or "Team player"
- Keep it concise (2-3 sentences, max 200 words)
- Focus on technical expertise, domain experience, and key achievements

Examples:
✅ Good: "Software Engineer with 5+ years of experience in Java and Spring Boot with exposure to banking domain. Strong expertise in microservices architecture and cloud technologies."
❌ Bad: "Hardworking individual seeking opportunities" (too generic)

💡 OUTPUT FORMAT

Return a single valid JSON object with only the summary field:

{
  "summary": "Software Engineer with 5+ years of experience in Java and Spring Boot with exposure to banking domain."
}

If no valid summary can be created, return:
{
  "summary": null
}

🧭 TONE AND STYLE

Objective, analytical, and strictly data-driven.
Focus on quantifiable professional traits and technical expertise.
Output must be clean JSON, no markdown formatting.

Resume Text:
{resume_text}
"""


def extract_summary_with_ai(
    text: str,
    ai_client: Any,
    ai_model: str
) -> Optional[str]:
    """
    Extract or generate professional summary from resume text using AI.
    
    Args:
        text: Resume text to extract summary from
        ai_client: AI client (Ollama, OpenAI, or Azure OpenAI)
        ai_model: Model name to use
        
    Returns:
        Extracted/generated summary string or None if not found
    """
    if not ai_client:
        logger.warning("AI client not available for summary extraction")
        return None
    
    try:
        # Prepare the prompt with resume text
        prompt = SUMMARY_EXTRACTION_PROMPT.replace("{resume_text}", text[:10000])
        
        # Determine service type for logging
        from ats_config import ATSConfig
        service_type = "OLLAMA" if ATSConfig.USE_OLLAMA else ("AZURE OPENAI" if ATSConfig.AZURE_OPENAI_ENDPOINT else "OPENAI")
        logger.info(f"Extracting summary using {service_type} (model: {ai_model})")
        
        # Call AI API
        response = ai_client.chat.completions.create(
            model=ai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Slightly higher for summary generation
            max_tokens=300,   # Summary needs more tokens
            response_format={"type": "json_object"}  # Ensure JSON output
        )
        
        logger.info(f"AI summary extraction response received, length: {len(response.choices[0].message.content)} chars")
        
        # Parse JSON response
        response_content = response.choices[0].message.content.strip()
        
        # Try to fix common JSON issues
        if not response_content.startswith('{'):
            start_idx = response_content.find('{')
            if start_idx != -1:
                response_content = response_content[start_idx:]
            else:
                logger.error(f"No JSON object found in AI response. First 200 chars: {response_content[:200]}")
                return None
        
        # Parse JSON
        try:
            ai_result = json.loads(response_content)
        except json.JSONDecodeError as je:
            logger.error(f"JSON decode error at position {je.pos}: {je.msg}")
            return None
        
        # Get extracted summary
        summary = ai_result.get('summary') or ''
        
        if not summary or summary.lower() == 'null':
            logger.warning("AI did not extract a summary")
            return None
        
        # Basic validation - summary should not be empty or too short
        summary = summary.strip()
        if len(summary) < 10:
            return None
        
        logger.info(f"AI extracted summary: {summary[:100]}...")
        return summary
            
    except Exception as e:
        logger.error(f"AI summary extraction failed: {e}")
        return None


# AI prompt for extracting search metadata from queries
SEARCH_METADATA_EXTRACTION_PROMPT = """🧠 ROLE / PERSONA

You are an Expert Search Query Parser for a Resume Search and Classification system.

Your job is to extract structured metadata from natural language search queries to enable precise resume matching.

🎯 EXTRACTION RULES

Extract the following metadata from the search query:

1. profile_type - The primary technology/domain profile (MUST be one of: java, python, dotnet, business analyst, others)
   - Look for keywords: "java", "python", ".net", "dotnet", "business analyst", "ba"
   - If no clear profile type found, use "others"
   - Normalize variations: ".net" or "dotnet" → "dotnet", "business analyst" or "ba" → "business analyst"

2. role_type - The job role/position type (e.g., "software engineer", "backend developer", "developer", "fresher")
   - Extract role keywords: developer, engineer, analyst, manager, fresher, senior, lead, etc.
   - Can be null if not specified

3. location - Geographic location mentioned (e.g., "Hyderabad", "Bangalore", "Mumbai")
   - Extract city names, states, or regions
   - Can be null if not specified

4. experience - Years of experience mentioned (e.g., "5 years", "2 years", "fresher")
   - Extract numeric values with "years" or "yrs"
   - For "fresher", set to 0
   - Can be null if not specified

5. skill_keywords - Technical skills or keywords mentioned (e.g., ["API", "Spring Boot", "REST"])
   - Extract technical terms, frameworks, tools
   - Return as array of strings
   - Can be empty array if not specified

💡 OUTPUT FORMAT

Return a single valid JSON object with all extracted metadata:

{
  "profile_type": "java",
  "role_type": "developer",
  "location": "hyderabad",
  "experience": 5,
  "skill_keywords": ["spring boot", "api"]
}

If a field is not found, use null (or empty array for skill_keywords):

{
  "profile_type": "python",
  "role_type": null,
  "location": null,
  "experience": null,
  "skill_keywords": []
}

🧭 EXAMPLES

Query: "java developer in hyderabad"
{
  "profile_type": "java",
  "role_type": "developer",
  "location": "hyderabad",
  "experience": null,
  "skill_keywords": []
}

Query: "python backend 5 years"
{
  "profile_type": "python",
  "role_type": "backend developer",
  "location": null,
  "experience": 5,
  "skill_keywords": []
}

Query: ".net API developer"
{
  "profile_type": "dotnet",
  "role_type": "developer",
  "location": null,
  "experience": null,
  "skill_keywords": ["api"]
}

Query: "business analyst fresher"
{
  "profile_type": "business analyst",
  "role_type": "fresher",
  "location": null,
  "experience": 0,
  "skill_keywords": []
}

🧭 TONE AND STYLE

Objective, analytical, and strictly data-driven.
Output must be clean JSON, no markdown formatting.

Search Query:
{search_query}
"""


def get_index_name_from_profile_type(profile_type: str) -> str:
    """
    Map profile_type to Pinecone index name.
    
    Args:
        profile_type: Profile type string (e.g., 'java', 'python', 'dotnet', 'business analyst')
    
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
    
    # Map to index names (matching the indexes created in create_all_pinecone_indexes.py)
    index_mapping = {
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
        'others': 'others',
        'other': 'others',
        'generalist': 'others',
        'general': 'others'
    }
    
    # Direct match
    if profile_lower in index_mapping:
        return index_mapping[profile_lower]
    
    # Partial match (contains keyword)
    for key, index_name in index_mapping.items():
        if key in profile_lower or profile_lower in key:
            return index_name
    
    # Default to others
    return 'others'


def extract_search_metadata_with_ai(
    query: str,
    ai_client: Any,
    ai_model: str
) -> Optional[Dict[str, Any]]:
    """
    Extract search metadata from query using AI.
    
    Args:
        query: Search query string
        ai_client: AI client (Ollama, OpenAI, or Azure OpenAI)
        ai_model: Model name to use
        
    Returns:
        Dictionary with extracted metadata or None if extraction fails
    """
    if not ai_client:
        logger.warning("AI client not available for search metadata extraction")
        return None
    
    try:
        # Prepare the prompt with search query
        prompt = SEARCH_METADATA_EXTRACTION_PROMPT.replace("{search_query}", query[:1000])
        
        # Determine service type for logging
        from ats_config import ATSConfig
        service_type = "OLLAMA" if ATSConfig.USE_OLLAMA else ("AZURE OPENAI" if ATSConfig.AZURE_OPENAI_ENDPOINT else "OPENAI")
        logger.info(f"Extracting search metadata using {service_type} (model: {ai_model})")
        
        # Call AI API
        response = ai_client.chat.completions.create(
            model=ai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # Lower temperature for more consistent extraction
            max_tokens=300,
            response_format={"type": "json_object"}  # Ensure JSON output
        )
        
        logger.info(f"AI search metadata extraction response received, length: {len(response.choices[0].message.content)} chars")
        
        # Parse JSON response
        response_content = response.choices[0].message.content.strip()
        
        # Try to fix common JSON issues
        if not response_content.startswith('{'):
            start_idx = response_content.find('{')
            if start_idx != -1:
                response_content = response_content[start_idx:]
            else:
                logger.error(f"No JSON object found in AI response. First 200 chars: {response_content[:200]}")
                return None
        
        # Parse JSON
        try:
            metadata = json.loads(response_content)
        except json.JSONDecodeError as je:
            logger.error(f"JSON decode error at position {je.pos}: {je.msg}")
            return None
        
        # Validate required fields
        if 'profile_type' not in metadata:
            logger.warning("AI did not extract profile_type")
            metadata['profile_type'] = 'others'
        
        logger.info(f"AI extracted metadata: profile_type={metadata.get('profile_type')}, "
                   f"role_type={metadata.get('role_type')}, location={metadata.get('location')}, "
                   f"experience={metadata.get('experience')}")
        
        return metadata
            
    except Exception as e:
        logger.error(f"AI search metadata extraction failed: {e}")
        return None


def Process_Search_Query(
    query: str,
    profile_type: Optional[str] = None,
    ai_client: Any = None,
    ai_model: str = None,
    top_k: int = 10,
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Process search query to extract metadata and query Pinecone from the correct index.
    
    This function:
    1. Extracts search metadata from query using AI (if profile_type not provided)
    2. Maps profile_type to the correct Pinecone index name
    3. Connects to the correct Pinecone index
    4. Generates query embedding
    5. Queries Pinecone with metadata filters
    
    Args:
        query: Search query string (e.g., "java developer in hyderabad")
        profile_type: Optional profile type to override AI extraction (e.g., "java", "python", "dotnet", "business analyst")
        ai_client: AI client for metadata extraction (if None, will try to get from ats_api)
        ai_model: AI model name (if None, will use from ATSConfig)
        top_k: Number of results to return (default: 10)
        include_metadata: Whether to include metadata in results (default: True)
    
    Returns:
        Dictionary containing:
        - success: bool
        - metadata: extracted search metadata
        - index_name: Pinecone index name used
        - results: Pinecone query results
        - error: error message if failed
    """
    try:
        from ats_config import ATSConfig
        
        # Get AI client and model if not provided
        if ai_client is None:
            try:
                if ATSConfig.USE_OLLAMA:
                    from ollama_client import get_ollama_openai_client
                    ai_client = get_ollama_openai_client(
                        base_url=ATSConfig.OLLAMA_BASE_URL,
                        model=ATSConfig.OLLAMA_MODEL
                    )
                    ai_model = ATSConfig.OLLAMA_MODEL
                elif ATSConfig.AZURE_OPENAI_ENDPOINT:
                    from openai import AzureOpenAI
                    ai_client = AzureOpenAI(
                        api_key=ATSConfig.AZURE_OPENAI_API_KEY,
                        api_version=ATSConfig.AZURE_OPENAI_API_VERSION,
                        azure_endpoint=ATSConfig.AZURE_OPENAI_ENDPOINT
                    )
                    ai_model = ATSConfig.AZURE_OPENAI_MODEL
                elif ATSConfig.OPENAI_API_KEY:
                    from openai import OpenAI
                    ai_client = OpenAI(api_key=ATSConfig.OPENAI_API_KEY)
                    ai_model = ATSConfig.OPENAI_MODEL
                else:
                    ai_client = None
                    ai_model = None
            except Exception as e:
                logger.warning(f"Could not initialize AI client: {e}")
                ai_client = None
                ai_model = None
        
        # Step 1: Extract metadata from query using AI (if profile_type not provided)
        if profile_type:
            # Use provided profile_type
            extracted_metadata = {
                'profile_type': profile_type.lower().strip(),
                'role_type': None,
                'location': None,
                'experience': None,
                'skill_keywords': []
            }
            logger.info(f"Using provided profile_type: {profile_type}")
        else:
            # Extract metadata using AI
            if ai_client:
                extracted_metadata = extract_search_metadata_with_ai(query, ai_client, ai_model)
                if not extracted_metadata:
                    logger.warning("AI metadata extraction failed, using defaults")
                    extracted_metadata = {
                        'profile_type': 'others',
                        'role_type': None,
                        'location': None,
                        'experience': None,
                        'skill_keywords': []
                    }
            else:
                logger.warning("AI client not available, using default profile_type: others")
                extracted_metadata = {
                    'profile_type': 'others',
                    'role_type': None,
                    'location': None,
                    'experience': None,
                    'skill_keywords': []
                }
        
        # Step 2: Map profile_type to namespace (using single index with namespaces)
        # Import the namespace mapping function from ats_api
        try:
            from ats_api import get_namespace_from_profile_type
        except ImportError:
            # Fallback to local function if import fails
            namespace = get_index_name_from_profile_type(extracted_metadata['profile_type'])
        else:
            namespace = get_namespace_from_profile_type(extracted_metadata['profile_type'])
        
        # Use the main index name from config (single index with namespaces)
        index_name = ATSConfig.PINECONE_INDEX_NAME
        logger.info(f"Mapped profile_type '{extracted_metadata['profile_type']}' to namespace '{namespace}' in index '{index_name}'")
        
        # Step 3: Initialize Pinecone client
        try:
            from pinecone import Pinecone
        except Exception as e:
            logger.error(f"Failed to import Pinecone: {e}")
            return {
                'success': False,
                'error': f'Pinecone import failed: {str(e)}. Please ensure pinecone package is installed correctly.',
                'metadata': extracted_metadata,
                'index_name': index_name,
                'namespace': namespace
            }
        
        pinecone_api_key = ATSConfig.PINECONE_API_KEY
        if not pinecone_api_key:
            return {
                'success': False,
                'error': 'PINECONE_API_KEY not configured',
                'metadata': extracted_metadata,
                'index_name': index_name,
                'namespace': namespace
            }
        
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Step 4: Check if main index exists (we use a single index with namespaces)
        existing_indexes = pc.list_indexes()
        existing_names = [idx.name for idx in existing_indexes]
        
        if index_name not in existing_names:
            logger.warning(f"Index '{index_name}' does not exist. Available indexes: {existing_names}")
            # Try to use the first available index as fallback
            if existing_names:
                index_name = existing_names[0]
                logger.info(f"Using fallback index: {index_name}")
            else:
                return {
                    'success': False,
                    'error': f"Index '{index_name}' does not exist. Available indexes: {existing_names}. Please create the main index first.",
                    'metadata': extracted_metadata,
                    'index_name': index_name,
                    'namespace': namespace
                }
        
        # Step 5: Connect to the main index (will query specific namespace)
        index = pc.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}, will query namespace: {namespace}")
        
        # Step 6: Generate query embedding
        try:
            from ats_api import embedding_service
            query_embedding = embedding_service.generate_embedding(query)
            logger.info(f"Generated query embedding with {len(query_embedding)} dimensions")
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return {
                'success': False,
                'error': f"Failed to generate embedding: {str(e)}",
                'metadata': extracted_metadata,
                'index_name': index_name
            }
        
        # Step 7: Build Pinecone filter from metadata
        pinecone_filter = {}
        
        # Add role_type filter if available
        if extracted_metadata.get('role_type'):
            # Note: This assumes role_type is stored in metadata. Adjust based on your schema
            pinecone_filter['role_type'] = {'$eq': extracted_metadata['role_type'].lower()}
        
        # Add location filter if available
        if extracted_metadata.get('location'):
            pinecone_filter['location'] = {'$eq': extracted_metadata['location'].lower()}
        
        # Add experience filter if available
        if extracted_metadata.get('experience') is not None:
            exp = extracted_metadata['experience']
            # Filter for candidates with experience >= specified (or == 0 for fresher)
            if exp == 0:
                pinecone_filter['total_experience'] = {'$eq': 0}
            else:
                pinecone_filter['total_experience'] = {'$gte': exp}
        
        # Add skill keywords filter if available
        if extracted_metadata.get('skill_keywords') and len(extracted_metadata['skill_keywords']) > 0:
            # Filter by primary_skills containing any of the keywords
            # Note: Pinecone filters support $in for arrays
            skill_filter = {'$in': [skill.lower() for skill in extracted_metadata['skill_keywords']]}
            pinecone_filter['primary_skills'] = skill_filter
        
        # Step 8: Query Pinecone with namespace
        query_params = {
            'vector': query_embedding,
            'top_k': top_k,
            'include_metadata': include_metadata,
            'namespace': namespace  # Query specific namespace within the index
        }
        
        if pinecone_filter:
            query_params['filter'] = pinecone_filter
            logger.info(f"Querying namespace '{namespace}' with filter: {pinecone_filter}")
        else:
            logger.info(f"Querying namespace '{namespace}' without filter")
        
        try:
            results = index.query(**query_params)
            logger.info(f"Query returned {len(results.matches)} results from namespace '{namespace}'")
            
            return {
                'success': True,
                'metadata': extracted_metadata,
                'index_name': index_name,
                'namespace': namespace,
                'results': results,
                'query': query,
                'top_k': top_k,
                'filter_applied': bool(pinecone_filter)
            }
        except Exception as e:
            logger.error(f"Pinecone query failed: {e}")
            return {
                'success': False,
                'error': f"Pinecone query failed: {str(e)}",
                'metadata': extracted_metadata,
                'index_name': index_name,
                'namespace': namespace
            }
            
    except Exception as e:
        logger.error(f"Process_Search_Query failed: {e}")
        return {
            'success': False,
            'error': f"Process_Search_Query failed: {str(e)}",
            'metadata': extracted_metadata if 'extracted_metadata' in locals() else {},
            'index_name': index_name if 'index_name' in locals() else 'unknown',
            'namespace': namespace if 'namespace' in locals() else 'unknown'
        }

