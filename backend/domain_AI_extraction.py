"""
AI-powered Domain Extraction Module for Resume Parsing.

Extracts professional domains/industries from resumes using AI.
"""

import json
import logging
from typing import Any, Optional, List

logger = logging.getLogger(__name__)

# Domain list for reference
DOMAINS_LIST = [
    "Information Technology", "Software Development", "Cloud Computing", "Cybersecurity", 
    "Data Science", "Blockchain", "Internet of Things", "Banking", "Finance", "Insurance", 
    "FinTech", "Healthcare", "Pharmaceuticals", "Biotechnology", "Manufacturing", "Automotive", 
    "Energy", "Construction", "Retail", "E-commerce", "Logistics", "Telecommunications",
    "Media & Entertainment", "Advertising & Marketing", "Education Technology", "Public Sector", 
    "Real Estate", "Hospitality", "Travel & Tourism", "Agriculture", "Legal & Compliance", 
    "Human Resources", "Environmental & Sustainability"
]

# AI prompt specifically for domain extraction
DOMAIN_EXTRACTION_PROMPT = """🧠 ROLE / PERSONA

You are an Expert Resume Parser specialized in extracting professional domains/industries from unstructured resume text.

Your only task is to identify and return the candidate's professional domains/industries in JSON format.

🎯 EXTRACTION RULES

1. domain – Extract professional domains or industries (not education fields).

Rules for Extraction:
- Look for industry keywords, company types, project domains, and business contexts
- Use the provided domain list as reference
- If technical keywords (Python, Java, AWS, etc.) appear → include "Information Technology"
- Include multiple relevant business domains (e.g., "Banking", "Finance")
- Ignore educational degrees — only professional domains count
- Return as an array of domain strings

Available Domains:
{domains_list}

Examples:
✅ Correct: ["Information Technology", "Banking"], ["Healthcare", "Pharmaceuticals"]
❌ Incorrect: ["B.Tech"], ["Computer Science"] (these are education fields, not business domains)

💡 OUTPUT FORMAT

Return a single valid JSON object with only the domain field as an array:

{
  "domain": ["Information Technology", "Banking"]
}

If no valid domains are found, return:
{
  "domain": []
}

🧭 TONE AND STYLE

Objective, analytical, and strictly data-driven.
Do not infer, assume, or explain — extract only.
Output must be clean JSON, no markdown formatting.

Resume Text:
{resume_text}
"""


def extract_domain_with_ai(
    text: str,
    ai_client: Any,
    ai_model: str
) -> Optional[List[str]]:
    """
    Extract professional domains/industries from resume text using AI.
    
    Args:
        text: Resume text to extract domains from
        ai_client: AI client (Ollama, OpenAI, or Azure OpenAI)
        ai_model: Model name to use
        
    Returns:
        List of domain strings or None if extraction failed
    """
    if not ai_client:
        logger.warning("AI client not available for domain extraction")
        return None
    
    try:
        # Prepare the prompt with resume text and domains list
        domains_str = ", ".join(DOMAINS_LIST)
        prompt = DOMAIN_EXTRACTION_PROMPT.replace("{domains_list}", domains_str)
        prompt = prompt.replace("{resume_text}", text[:10000])
        
        # Determine service type for logging
        from ats_config import ATSConfig
        service_type = "OLLAMA" if ATSConfig.USE_OLLAMA else ("AZURE OPENAI" if ATSConfig.AZURE_OPENAI_ENDPOINT else "OPENAI")
        logger.info(f"Extracting domains using {service_type} (model: {ai_model})")
        
        # Call AI API
        response = ai_client.chat.completions.create(
            model=ai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for consistent results
            max_tokens=500,   # Domains list might need more tokens
            response_format={"type": "json_object"}  # Ensure JSON output
        )
        
        logger.info(f"AI domain extraction response received, length: {len(response.choices[0].message.content)} chars")
        
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
        
        # Get extracted domains
        domain_list = ai_result.get('domain', [])
        
        # Ensure it's a list
        if not isinstance(domain_list, list):
            if domain_list:
                domain_list = [domain_list]
            else:
                domain_list = []
        
        # Filter out empty strings and validate
        domain_list = [d.strip() for d in domain_list if d and isinstance(d, str) and d.strip()]
        
        if not domain_list:
            logger.warning("AI did not extract any domains")
            return []
        
        logger.info(f"AI extracted domains: {domain_list}")
        return domain_list
            
    except Exception as e:
        logger.error(f"AI domain extraction failed: {e}")
        return None

