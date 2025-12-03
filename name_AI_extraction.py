"""
AI-powered Name Extraction Module for Resume Parsing.

Extracts candidate names from resumes using AI with comprehensive validation
and fallback mechanisms.
"""

import re
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# AI prompt specifically for name extraction
NAME_EXTRACTION_PROMPT = """🧠 ROLE / PERSONA

You are an Expert Resume Parser specialized in extracting candidate names from unstructured resume text.

Your only task is to identify and return the candidate's ACTUAL PERSONAL NAME in JSON format.

🎯 EXTRACTION RULES

1. full_name – Identify the candidate's ACTUAL PERSONAL NAME (e.g., "Daniel Mindlin", "John Smith", "MUPPANA APARNA DEVI"). 

   CRITICAL: Do NOT confuse section headers or labels (like "Education", "Experience", "Skills", "Contact Information") with the person's name.
   The candidate's name is typically at the top of the resume, often centered or on the left.
   It is NEVER a section header. If the first prominent text is a section header, look for the actual name below or above it.

Rules for Extraction:
- Name is usually on line 1 or 2 (Title Case or ALL CAPS).
- Should contain 2–4 alphabetic words only.
- Reject anything containing punctuation, digits, or organization names.
- Stop searching after the first 3 lines.

Reject:
- Section headers (Education, Experience, Skills, etc.)
- Degrees, Job Titles, or Organization Names.
- Locations (City, State patterns)
- Email addresses or phone numbers

Examples:
✅ Correct: Daniel Mindlin, VARRE DHANA LAKSHMI DURGA, MUPPANA APARNA DEVI
❌ Incorrect: Education, Infosys, B.Tech in EEE, Rajahmundry, Andhra Pradesh

💡 OUTPUT FORMAT

Return a single valid JSON object with only the full_name field:

{
  "full_name": "John M. Smith"
}

If no valid name is found, return:
{
  "full_name": null
}

🧭 TONE AND STYLE

Objective, analytical, and strictly data-driven.
Do not infer, assume, or explain — extract only.
Output must be clean JSON, no markdown formatting.

Resume Text (look for name in FIRST FEW LINES):
{resume_text}
"""


def validate_extracted_name(full_name: str) -> bool:
    """
    Validate if the extracted name is valid (not a section header, degree, etc.).
    
    Args:
        full_name: The name extracted by AI
        
    Returns:
        True if valid, False if invalid
    """
    if not full_name or not isinstance(full_name, str):
        return False
    
    full_name = full_name.strip()
    if not full_name:
        return False
    
    # Common section headers and academic degrees that should NEVER be names
    invalid_names = ['education', 'experience', 'skills', 'contact', 'objective', 
                   'summary', 'qualifications', 'work history', 'professional summary',
                   'references', 'certifications', 'projects', 'achievements', 'unknown']
    
    # Check if it's in invalid names list
    if full_name.lower() in invalid_names:
        return False
    
    # Check for academic degree patterns (B.A., M.S., PhD, etc.)
    degree_patterns = [
        r'\b([BM]\.?[AS]\.?|MBA|PhD|MD|JD|B\.?Tech|M\.?Tech)\b',
        r'\bin\s+[A-Z][a-z]+',
        r'degree|diploma|certificate'
    ]
    
    for pattern in degree_patterns:
        if re.search(pattern, full_name, re.IGNORECASE):
            return False
    
    # Check if it looks like an academic degree format (e.g., "B.A. in History")
    if any(keyword in full_name.lower() for keyword in ['in ', 'degree', 'major', 'minor']):
        return False
    
    # Check if it's a sentence fragment from bullet points (e.g., "full course load.")
    sentence_fragment_keywords = ['while maintaining', 'while working', 'while attending', 
                                  'while completing', 'full course', 'as part of', 
                                  'in order to', 'for the', 'that', 'load.', 'completed in']
    if any(phrase in full_name.lower() for phrase in sentence_fragment_keywords):
        return False
    
    # Check if it ends with a period or comma (likely a sentence fragment)
    if full_name.endswith('.') or full_name.endswith(','):
        return False
    
    # Check if it looks like a location pattern (City, State)
    if ',' in full_name:
        parts = [p.strip() for p in full_name.split(',')]
        if len(parts) == 2:
            part1, part2 = parts[0], parts[1]
            # If both parts are 1-3 words and look like location names
            if (1 <= len(part1.split()) <= 3 and 1 <= len(part2.split()) <= 3 and
                all(word.replace('.', '').isalpha() for word in part1.split()) and
                all(word.replace('.', '').isalpha() for word in part2.split())):
                # Check if second part looks like a state/province/country
                part2_lower = part2.lower()
                location_indicators = [
                    'andhra pradesh', 'telangana', 'karnataka', 'tamil nadu', 'maharashtra',
                    'gujarat', 'rajasthan', 'west bengal', 'uttar pradesh', 'punjab',
                    'haryana', 'delhi', 'hyderabad', 'bangalore', 'mumbai', 'chennai',
                    'kolkata', 'pune', 'rajahmundry', 'visakhapatnam', 'vijayawada',
                    'india', 'usa', 'united states', 'uk', 'united kingdom'
                ]
                if any(indicator in part2_lower for indicator in location_indicators):
                    return False
                # Also check if it's a common location pattern (two capitalized words separated by comma)
                if (part1[0].isupper() and part2[0].isupper() and 
                    len(part1) > 3 and len(part2) > 3):
                    return False
    
    # Check if it contains email or phone patterns
    if '@' in full_name or re.search(r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', full_name):
        return False
    
    # Check if it's a reasonable name length (2-4 words, reasonable character count)
    words = full_name.split()
    if not (2 <= len(words) <= 4):
        return False
    
    if len(full_name) > 70:
        return False
    
    return True


def extract_name_with_ai(
    text: str,
    ai_client: Any,
    ai_model: str,
    fallback_extractor: Optional[Any] = None,
    nlp: Optional[Any] = None
) -> Optional[str]:
    """
    Extract candidate name from resume text using AI with validation and fallback.
    
    Args:
        text: Resume text to extract name from
        ai_client: AI client (Ollama, OpenAI, or Azure OpenAI)
        ai_model: Model name to use
        fallback_extractor: Optional function/method for regex-based fallback extraction
        nlp: Optional spaCy NLP model for final fallback
        
    Returns:
        Extracted name string or None if not found
    """
    if not ai_client:
        logger.warning("AI client not available for name extraction")
        return None
    
    try:
        # Prepare the prompt with resume text (focus on first 5000 chars for name extraction)
        prompt = NAME_EXTRACTION_PROMPT.replace("{resume_text}", text[:5000])
        
        # Determine service type for logging
        from ats_config import ATSConfig
        service_type = "OLLAMA" if ATSConfig.USE_OLLAMA else ("AZURE OPENAI" if ATSConfig.AZURE_OPENAI_ENDPOINT else "OPENAI")
        logger.info(f"Extracting name using {service_type} (model: {ai_model})")
        
        # Call AI API
        response = ai_client.chat.completions.create(
            model=ai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for consistent results
            max_tokens=500,   # Name extraction doesn't need many tokens
            response_format={"type": "json_object"}  # Ensure JSON output
        )
        
        logger.info(f"AI name extraction response received, length: {len(response.choices[0].message.content)} chars")
        
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
            logger.error(f"Problematic JSON around error: {response_content[max(0, je.pos-100):je.pos+100]}")
            return None
        
        # Get extracted name
        full_name = ai_result.get('full_name') or ''
        
        if not full_name or full_name.lower() == 'null':
            logger.warning("AI did not extract a name")
            return None
        
        # Validate extracted name
        if validate_extracted_name(full_name):
            logger.info(f"AI extracted valid name: {full_name}")
            return full_name
        else:
            logger.warning(f"AI extracted invalid name '{full_name}' (section header/academic degree/location). Trying fallback...")
            
            # Try regex fallback if provided
            if fallback_extractor:
                try:
                    fallback_name = fallback_extractor(text)
                    if fallback_name and fallback_name != 'Unknown' and validate_extracted_name(fallback_name):
                        logger.info(f"Fallback extraction found valid name: {fallback_name}")
                        return fallback_name
                except Exception as e:
                    logger.warning(f"Fallback extraction failed: {e}")
            
            # Last resort: try NLP if available
            if nlp:
                try:
                    logger.warning("Trying NLP fallback...")
                    doc = nlp(text[:500])
                    for ent in doc.ents:
                        if ent.label_ == "PERSON" and len(ent.text.split()) <= 4:
                            if validate_extracted_name(ent.text):
                                logger.info(f"Found name via NLP: {ent.text}")
                                return ent.text
                except Exception as e:
                    logger.warning(f"NLP fallback failed: {e}")
            
            return None
            
    except Exception as e:
        logger.error(f"AI name extraction failed: {e}")
        return None

