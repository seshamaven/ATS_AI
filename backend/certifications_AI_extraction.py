"""
AI-powered Certifications Extraction Module for Resume Parser.

Extracts professional certifications from resumes using AI.
"""

import json
import logging
from typing import Any, Optional, List

logger = logging.getLogger(__name__)

# AI prompt specifically for certifications extraction
CERTIFICATIONS_EXTRACTION_PROMPT = """🧠 ROLE / PERSONA

You are an Expert Resume Parser specialized in extracting professional certifications from unstructured resume text.

Your only task is to identify and return all professional or vendor certifications in JSON format.

🎯 EXTRACTION RULES

1. certifications – Capture all professional or vendor certifications.

Rules for Extraction:
- Look in Certifications, Certificates, or Professional Development sections
- Extract vendor certifications (e.g., AWS Certified Developer, Microsoft Azure, Google Cloud)
- Extract professional certifications (e.g., PMP, Scrum Master, Six Sigma)
- Extract technology-specific certifications (e.g., Oracle Certified Professional, Cisco CCNA)
- Return as an array of certification strings
- Include the full certification name when available

Examples:
✅ Correct: ["AWS Certified Developer", "PMP", "Microsoft Azure Fundamentals", "Oracle Certified Professional Java SE"]
❌ Incorrect: ["Bachelor's Degree", "Master's Degree"] (these are education, not certifications)

💡 OUTPUT FORMAT

Return a single valid JSON object with only the certifications field as an array:

{
  "certifications": ["AWS Certified Developer", "PMP"]
}

If no valid certifications are found, return:
{
  "certifications": []
}

🧭 TONE AND STYLE

Objective, analytical, and strictly data-driven.
Do not infer, assume, or explain — extract only.
Output must be clean JSON, no markdown formatting.

Resume Text:
{resume_text}
"""


def extract_certifications_with_ai(
    text: str,
    ai_client: Any,
    ai_model: str
) -> Optional[List[str]]:
    """
    Extract professional certifications from resume text using AI.
    
    Args:
        text: Resume text to extract certifications from
        ai_client: AI client (Ollama, OpenAI, or Azure OpenAI)
        ai_model: Model name to use
        
    Returns:
        List of certification strings or None if extraction failed
    """
    if not ai_client:
        logger.warning("AI client not available for certifications extraction")
        return None
    
    try:
        # Prepare the prompt with resume text
        prompt = CERTIFICATIONS_EXTRACTION_PROMPT.replace("{resume_text}", text[:10000])
        
        # Determine service type for logging
        from ats_config import ATSConfig
        service_type = "OLLAMA" if ATSConfig.USE_OLLAMA else ("AZURE OPENAI" if ATSConfig.AZURE_OPENAI_ENDPOINT else "OPENAI")
        logger.info(f"Extracting certifications using {service_type} (model: {ai_model})")
        
        # Call AI API
        response = ai_client.chat.completions.create(
            model=ai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for consistent results
            max_tokens=1000,   # Certifications list might need more tokens
            response_format={"type": "json_object"}  # Ensure JSON output
        )
        
        logger.info(f"AI certifications extraction response received, length: {len(response.choices[0].message.content)} chars")
        
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
        
        # Get extracted certifications
        certifications = ai_result.get('certifications', [])
        
        # Ensure it's a list
        if not isinstance(certifications, list):
            if certifications:
                certifications = [certifications]
            else:
                certifications = []
        
        # Filter out empty strings and validate
        certifications = [c.strip() for c in certifications if c and isinstance(c, str) and c.strip()]
        
        if not certifications:
            logger.warning("AI did not extract any certifications")
            return []
        
        logger.info(f"AI extracted certifications: {certifications}")
        return certifications
            
    except Exception as e:
        logger.error(f"AI certifications extraction failed: {e}")
        return None

