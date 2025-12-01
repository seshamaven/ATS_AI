"""
AI-powered Summary Extraction Module for Resume Parsing.

Extracts professional summary from resumes using AI.
"""

import json
import logging
from typing import Any, Optional

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

