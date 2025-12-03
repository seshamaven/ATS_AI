"""
AI-powered Job Description Extraction Module.

Extracts role, sub_role, profile_type, profile_sub_type, and primary_skills from job descriptions using AI.
"""

import json
import logging
from typing import Any, Optional, Dict, List

logger = logging.getLogger(__name__)

# AI prompt for job description extraction
JOB_DESCRIPTION_EXTRACTION_PROMPT = """🧠 ROLE / PERSONA

You are an Expert Job Description Parser specialized in extracting structured information from job descriptions.

Your task is to extract and return the following fields in JSON format:
1. role - The main job title/role (e.g., "Software Engineer", "Senior Software Engineer", "Business Analyst", "Data Scientist")
2. sub_role - The sub-role category: "Frontend", "Backend", or "Full Stack" (only these three options)
3. profile_type - The primary technology profile type (e.g., "Java", "Python", ".Net", "JavaScript", "SAP", "DevOps", "Data Science")
4. profile_sub_type - Specific technologies/frameworks (e.g., "React", "Spring Boot", "Django", "Angular", "SQL", "AWS")
5. primary_skills - List of the most important technical skills required (top 10-15 skills)

🎯 EXTRACTION RULES

1. ROLE:
   - Extract the main job title from the job description
   - Examples: "Software Engineer", "Senior Backend Developer", "Data Analyst", "DevOps Engineer"
   - If multiple titles mentioned, use the primary one

2. SUB_ROLE:
   - Must be EXACTLY one of: "Frontend", "Backend", or "Full Stack"
   - Frontend: If mentions React, Angular, Vue, HTML, CSS, JavaScript frontend work
   - Backend: If mentions server-side, APIs, databases, microservices, backend services
   - Full Stack: If mentions both frontend and backend, or full stack explicitly
   - Default to "Backend" if unclear

3. PROFILE_TYPE:
   - Identify the primary technology stack profile
   - Common types: "Java", "Python", ".Net", "JavaScript", "SAP", "DevOps", "Data Science", "Data Engineering", "Cloud / Infra", "Testing / QA", "Mobile Development", "Salesforce", "RPA", "Cyber Security"
   - Choose the most dominant technology stack mentioned

4. PROFILE_SUB_TYPE:
   - Extract specific technologies, frameworks, or tools
   - Examples: "React", "Spring Boot", "Django", "Angular", "SQL", "AWS", "Docker", "Kubernetes"
   - Can be multiple technologies separated by comma
   - Focus on the most important ones (top 3-5)

5. PRIMARY_SKILLS:
   - Extract the most important technical skills required
   - Should be a list of skill names (strings)
   - Include programming languages, frameworks, tools, platforms
   - Limit to top 10-15 most important skills
   - Examples: ["Java", "Spring Boot", "MySQL", "REST API", "Microservices", "Docker"]

💡 OUTPUT FORMAT

Return a single valid JSON object with all fields:

{
  "role": "Software Engineer",
  "sub_role": "Backend",
  "profile_type": "Java",
  "profile_sub_type": "Spring Boot, MySQL, REST API",
  "primary_skills": ["Java", "Spring Boot", "MySQL", "REST API", "Microservices", "Docker", "Git"]
}

If a field cannot be determined, use null or empty string as appropriate.

🧭 TONE AND STYLE

Objective, analytical, and strictly data-driven.
Extract only what is explicitly mentioned or clearly implied.
Output must be clean JSON, no markdown formatting.

Job Description:
{job_description}
"""


def extract_job_metadata_with_ai(
    job_description: str,
    ai_client: Any,
    ai_model: str
) -> Optional[Dict[str, Any]]:
    """
    Extract role, sub_role, profile_type, profile_sub_type, and primary_skills from job description using AI.
    
    Args:
        job_description: Job description text to extract from
        ai_client: AI client (Ollama, OpenAI, or Azure OpenAI)
        ai_model: Model name to use
        
    Returns:
        Dictionary with extracted fields or None if extraction failed
    """
    if not ai_client:
        logger.warning("AI client not available for job description extraction")
        return None
    
    try:
        # Prepare the prompt with job description
        prompt = JOB_DESCRIPTION_EXTRACTION_PROMPT.replace("{job_description}", job_description[:15000])
        
        # Determine service type for logging
        from ats_config import ATSConfig
        service_type = "OLLAMA" if ATSConfig.USE_OLLAMA else ("AZURE OPENAI" if ATSConfig.AZURE_OPENAI_ENDPOINT else "OPENAI")
        logger.info(f"Extracting job metadata using {service_type} (model: {ai_model})")
        
        # Call AI API
        response = ai_client.chat.completions.create(
            model=ai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for consistent results
            max_tokens=1000,  # Enough for structured extraction
            response_format={"type": "json_object"}  # Ensure JSON output
        )
        
        logger.info(f"AI job metadata extraction response received, length: {len(response.choices[0].message.content)} chars")
        
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
            logger.error(f"Response content: {response_content[:500]}")
            return None
        
        # Extract and validate fields
        extracted = {
            'role': ai_result.get('role', '').strip() if ai_result.get('role') else '',
            'sub_role': ai_result.get('sub_role', '').strip() if ai_result.get('sub_role') else '',
            'profile_type': ai_result.get('profile_type', '').strip() if ai_result.get('profile_type') else '',
            'profile_sub_type': ai_result.get('profile_sub_type', '').strip() if ai_result.get('profile_sub_type') else '',
            'primary_skills': ai_result.get('primary_skills', [])
        }
        
        # Validate sub_role - must be one of the allowed values
        if extracted['sub_role'] and extracted['sub_role'] not in ['Frontend', 'Backend', 'Full Stack']:
            logger.warning(f"Invalid sub_role '{extracted['sub_role']}', defaulting to 'Backend'")
            extracted['sub_role'] = 'Backend'
        
        # Ensure primary_skills is a list
        if not isinstance(extracted['primary_skills'], list):
            if extracted['primary_skills']:
                extracted['primary_skills'] = [str(extracted['primary_skills'])]
            else:
                extracted['primary_skills'] = []
        
        # Clean and validate primary_skills
        extracted['primary_skills'] = [s.strip() for s in extracted['primary_skills'] if s and isinstance(s, str) and s.strip()]
        
        logger.info(f"AI extracted job metadata: role={extracted['role']}, sub_role={extracted['sub_role']}, profile_type={extracted['profile_type']}, profile_sub_type={extracted['profile_sub_type']}, primary_skills_count={len(extracted['primary_skills'])}")
        return extracted
            
    except Exception as e:
        logger.error(f"AI job metadata extraction failed: {e}", exc_info=True)
        return None

