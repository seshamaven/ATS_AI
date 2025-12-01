"""
AI-powered Designation/Job Title Extraction Module for Resume Parsing.

Extracts current/most recent job designation from resumes using AI.
"""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# AI prompt specifically for designation extraction
DESIGNATION_EXTRACTION_PROMPT = """🧠 ROLE / PERSONA

You are an Expert Resume Parser specialized in extracting the current or most recent job title/designation from unstructured resume text.

Your only task is to identify and return the candidate's CURRENT or MOST RECENT JOB TITLE/DESIGNATION in JSON format.

🎯 EXTRACTION RULES

1. current_designation – Extract the most recent job title/designation.

Rules for Extraction:
- Look in the Experience/Work History section
- Find the most recent job entry
- Extract the job title/designation (e.g., "Software Engineer", "Senior Developer", "Project Manager")
- Job titles are typically Title Case
- Reject company names, locations, or other non-title text

Examples:
✅ Correct: Software Engineer, Senior Software Developer, Project Manager, Data Scientist
❌ Incorrect: Infosys, Microsoft, Bangalore (these are companies/locations)

💡 OUTPUT FORMAT

Return a single valid JSON object with only the current_designation field:

{
  "current_designation": "Software Engineer"
}

If no valid designation is found, return:
{
  "current_designation": null
}

🧭 TONE AND STYLE

Objective, analytical, and strictly data-driven.
Do not infer, assume, or explain — extract only.
Output must be clean JSON, no markdown formatting.

Resume Text:
{resume_text}
"""


def extract_designation_with_ai(
    text: str,
    ai_client: Any,
    ai_model: str
) -> Optional[str]:
    """
    Extract current/most recent job designation from resume text using AI.
    
    Args:
        text: Resume text to extract designation from
        ai_client: AI client (Ollama, OpenAI, or Azure OpenAI)
        ai_model: Model name to use
        
    Returns:
        Extracted designation string or None if not found
    """
    if not ai_client:
        logger.warning("AI client not available for designation extraction")
        return None
    
    try:
        # Prepare the prompt with resume text
        prompt = DESIGNATION_EXTRACTION_PROMPT.replace("{resume_text}", text[:10000])
        
        # Determine service type for logging
        from ats_config import ATSConfig
        service_type = "OLLAMA" if ATSConfig.USE_OLLAMA else ("AZURE OPENAI" if ATSConfig.AZURE_OPENAI_ENDPOINT else "OPENAI")
        logger.info(f"Extracting designation using {service_type} (model: {ai_model})")
        
        # Call AI API
        response = ai_client.chat.completions.create(
            model=ai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for consistent results
            max_tokens=200,   # Designation doesn't need many tokens
            response_format={"type": "json_object"}  # Ensure JSON output
        )
        
        logger.info(f"AI designation extraction response received, length: {len(response.choices[0].message.content)} chars")
        
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
        
        # Get extracted designation
        current_designation = ai_result.get('current_designation') or ''
        
        if not current_designation or current_designation.lower() == 'null':
            logger.warning("AI did not extract a designation")
            return None
        
        # Basic validation - designation should not be empty or too short
        current_designation = current_designation.strip()
        if len(current_designation) < 2:
            return None
        
        logger.info(f"AI extracted designation: {current_designation}")
        return current_designation
            
    except Exception as e:
        logger.error(f"AI designation extraction failed: {e}")
        return None

