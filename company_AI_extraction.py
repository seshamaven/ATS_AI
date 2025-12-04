"""
AI-powered Company Extraction Module for Resume Parsing.

Extracts current/most recent company name from resumes using AI.
"""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# AI prompt specifically for company extraction
COMPANY_EXTRACTION_PROMPT = """🧠 ROLE / PERSONA

You are an Expert Resume Parser specialized in extracting the current or most recent employer/company name from unstructured resume text.

Your only task is to identify and return the candidate's CURRENT or MOST RECENT COMPANY NAME in JSON format.

🎯 EXTRACTION RULES

1. current_company – Extract the current or most recent employer name.

Rules for Extraction:
- Look in the Experience/Work History section
- Find the most recent job entry
- Extract the company/organization name
- Company names are typically capitalized (e.g., "Infosys", "Microsoft", "TCS")
- May include suffixes like "Inc", "LLC", "Ltd", "Corp", "Pvt Ltd"
- Reject job titles, designations, or role descriptions

Examples:
✅ Correct: Infosys, Microsoft Corporation, Tata Consultancy Services, Wipro Technologies
❌ Incorrect: Software Engineer, Senior Developer, Project Manager (these are job titles)

💡 OUTPUT FORMAT

Return a single valid JSON object with only the current_company field:

{
  "current_company": "Infosys"
}

If no valid company is found, return:
{
  "current_company": null
}

🧭 TONE AND STYLE

Objective, analytical, and strictly data-driven.
Do not infer, assume, or explain — extract only.
Output must be clean JSON, no markdown formatting.

Resume Text:
{resume_text}
"""


def extract_company_with_ai(
    text: str,
    ai_client: Any,
    ai_model: str
) -> Optional[str]:
    """
    Extract current/most recent company name from resume text using AI.
    
    Args:
        text: Resume text to extract company from
        ai_client: AI client (Ollama, OpenAI, or Azure OpenAI)
        ai_model: Model name to use
        
    Returns:
        Extracted company name string or None if not found
    """
    if not ai_client:
        logger.warning("AI client not available for company extraction")
        return None
    
    try:
        # Prepare the prompt with resume text
        prompt = COMPANY_EXTRACTION_PROMPT.replace("{resume_text}", text[:10000])
        
        # Determine service type for logging
        from ats_config import ATSConfig
        service_type = "OLLAMA" if ATSConfig.USE_OLLAMA else ("AZURE OPENAI" if ATSConfig.AZURE_OPENAI_ENDPOINT else "OPENAI")
        logger.info(f"Extracting company using {service_type} (model: {ai_model})")
        
        # Call AI API
        response = ai_client.chat.completions.create(
            model=ai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for consistent results
            max_tokens=200,   # Company name doesn't need many tokens
            response_format={"type": "json_object"}  # Ensure JSON output
        )
        
        logger.info(f"AI company extraction response received, length: {len(response.choices[0].message.content)} chars")
        
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
        
        # Get extracted company
        current_company = ai_result.get('current_company') or ''
        
        if not current_company or current_company.lower() == 'null':
            logger.warning("AI did not extract a company")
            return None
        
        # Basic validation - company should not be empty or too short
        current_company = current_company.strip()
        if len(current_company) < 2:
            return None
        
        logger.info(f"AI extracted company: {current_company}")
        return current_company
            
    except Exception as e:
        logger.error(f"AI company extraction failed: {e}")
        return None

