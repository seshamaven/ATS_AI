"""
Rule-based Name Extraction Module for Resume Parsing.

Extracts candidate names from resumes using rule-based pattern matching
without AI dependencies.
"""

import re
import sys
import logging
from typing import Optional, Any, Set, Dict

# Import COMMON_TITLES from designation_extraction
try:
    from designation_extraction import COMMON_TITLES
except ImportError:
    # Fallback if import fails
    COMMON_TITLES = set()

logger = logging.getLogger(__name__)


def strip_prefixes(line):
    """Strip common prefixes like 'Resume of', 'CV', 'Profile'."""
    line_lower = line.lower().strip()
    prefixes = [
        r'^resume\s+of\s+',
        r'^cv\s+of\s+',
        r'^profile\s+of\s+',
        r'^resume\s*:?\s*',
        r'^cv\s*:?\s*',
        r'^profile\s*:?\s*',
    ]
    for prefix in prefixes:
        line = re.sub(prefix, '', line, flags=re.IGNORECASE)
    return line.strip()


def should_skip_line(line):
    """
    Check if line should be skipped based on various criteria.
    
    Returns True if line should be skipped, False otherwise.
    """
    line_lower = line.lower().strip()
    
    # Skip empty lines
    if not line_lower:
        return True
    
    # Skip lines starting with '.' (e.g., ".Net Developer")
    if line.strip().startswith('.'):
        return True
    
    # Skip lines containing digits
    if re.search(r'\d', line):
        return True
    
    # Skip lines containing '@', 'linkedin', 'http', '.com'
    skip_patterns = ['@', 'linkedin', 'http', '.com', 'www.', 'github.com']
    if any(pattern in line_lower for pattern in skip_patterns):
        return True
    
    # Skip lines containing COMMON_TITLES from designation_extraction
    if COMMON_TITLES:
        line_lower_normalized = re.sub(r'[^\w\s]', ' ', line_lower)  # Normalize punctuation
        line_lower_normalized = re.sub(r'\s+', ' ', line_lower_normalized).strip()
        
        # Check if any common title is in the line
        for title in COMMON_TITLES:
            title_lower = title.lower().strip()
            # Check if title appears as a whole word/phrase in the line
            if title_lower in line_lower or f' {title_lower} ' in line_lower or line_lower.startswith(title_lower) or line_lower.endswith(title_lower):
                return True
    
    # Skip lines containing summary keywords
    summary_keywords = ['summary', 'professional summary', 'objective', 'profile summary']
    if any(keyword in line_lower for keyword in summary_keywords):
        return True
    
    return False


def is_certification_or_acronym(word):
    """
    Check if a word is likely a certification or acronym (not part of name).
    
    Args:
        word: Word to check
        
    Returns:
        True if word is a certification/acronym, False otherwise
    """
    if not word or not isinstance(word, str):
        return False
    
    word = word.strip()
    if not word:
        return False
    
    # Common certifications and acronyms (all uppercase, 2-5 letters)
    common_certs = {
        'pmp', 'csm', 'cspo', 'pmi', 'aws', 'azure', 'gcp', 'itil', 'scrum', 
        'agile', 'six', 'sigma', 'lean', 'mba', 'phd', 'ms', 'bs', 'ba', 'ma',
        'cissp', 'ceh', 'cisa', 'cisco', 'ccna', 'ccnp', 'mcp', 'mcsd', 'ocp',
        'java', 'python', 'sql', 'html', 'css', 'js', 'api', 'rest', 'soap',
        'pmp', 'csm', 'cspo', 'pmi', 'aws', 'azure', 'gcp', 'itil', 'scrum'
    }
    
    # If it's a known certification/acronym
    if word.upper() in common_certs:
        return True
    
    # If it's all uppercase and 2-5 letters (likely acronym)
    if word.isupper() and 2 <= len(word) <= 5 and word.isalpha():
        return True
    
    return False


def is_valid_name(name):
    """
    Check if a name is valid according to the rules.
    
    Rules:
    - Must have 2 to 4 words
    - Words must be either Title Case (Sesha Reddy) OR ALL CAPS (M SESHA REDDY)
    - Contain only alphabets and spaces
    """
    if not name or not isinstance(name, str):
        return False
    
    name = name.strip()
    if not name:
        return False
    
    # Split into words
    words = name.split()
    
    # Must have 2 to 4 words
    if not (2 <= len(words) <= 4):
        return False
    
    # Check each word
    all_title_case = True
    all_uppercase = True
    
    for word in words:
        # Must contain only alphabets (no digits, special chars except spaces)
        if not word.isalpha():
            return False
        
        # Check if Title Case (first letter uppercase, rest lowercase)
        # Handle single-letter words (e.g., "M" in "M SESHA REDDY")
        if len(word) == 1:
            # Single letter must be uppercase to be valid
            if not word.isupper():
                return False
            # Single uppercase letter is valid for both Title Case and ALL CAPS
        else:
            # Multi-letter word: check Title Case
            if not (word[0].isupper() and word[1:].islower()):
                all_title_case = False
        
        # Check if ALL CAPS
        if not word.isupper():
            all_uppercase = False
    
    # Name must be either all Title Case OR all ALL CAPS
    return all_title_case or all_uppercase


def normalize_for_skill_match(text: str) -> str:
    """
    Normalize text the same way skill_extractor normalizes skills.
    Matches skill_extractor._normalize_skills() logic EXACTLY.
    
    skill_extractor._normalize_skills() does: skill.lower().strip()
    (No special char removal, no whitespace normalization beyond strip)
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text (lowercase and stripped only)
    """
    if not text:
        return ""
    # EXACT match to skill_extractor._normalize_skills(): just lowercase and strip
    normalized = text.lower().strip()
    return normalized


def is_skill_match(potential_name: str, extracted_skills: Set[str]) -> bool:
    """
    Check if potential name matches extracted skills.
    
    Returns True if name should be rejected (it's a skill).
    
    Rules:
    - Exact match: If normalized name exactly matches any skill → reject
    - Partial match: If ≥50% of words in name match skills → reject
    - Fallback: Check against TECH_SKILLS directly (always check as safety net)
    - Explicit tool names: Check against common tool names list
    
    Args:
        potential_name: Potential candidate name to check
        extracted_skills: Set of normalized skills from skill_extractor
        
    Returns:
        True if name should be rejected (it's a skill), False otherwise
    """
    if not potential_name:
        return False
    
    normalized_name = normalize_for_skill_match(potential_name)
    
    # ALWAYS check against TECH_SKILLS as a safety net (even if extracted_skills exists)
    # This catches cases where skills aren't extracted from formal sections
    try:
        from skill_extractor import TECH_SKILLS
        # Normalize TECH_SKILLS for comparison
        tech_skills_normalized = {skill.lower().strip() for skill in TECH_SKILLS}
        
        # Check exact match against TECH_SKILLS
        if normalized_name in tech_skills_normalized:
            logger.info(f"Rejected '{potential_name}' - exact match in TECH_SKILLS: '{normalized_name}'")
            return True
        
        # Check if ≥50% of words match TECH_SKILLS
        name_words = normalized_name.split()
        if name_words:
            tech_skill_word_matches = sum(1 for word in name_words if word in tech_skills_normalized)
            match_ratio = tech_skill_word_matches / len(name_words)
            if match_ratio >= 0.5:
                logger.info(f"Rejected '{potential_name}' - {tech_skill_word_matches}/{len(name_words)} words match TECH_SKILLS ({match_ratio:.0%})")
                return True
    except ImportError:
        pass  # skill_extractor not available, skip fallback
    
    # Check against extracted_skills if available
    if extracted_skills:
        # Exact match check
        if normalized_name in extracted_skills:
            logger.info(f"Rejected '{potential_name}' - exact skill match: '{normalized_name}'")
            return True
        
        # Check if ≥50% of words match skills
        name_words = normalized_name.split()
        if name_words:
            skill_word_matches = sum(1 for word in name_words if word in extracted_skills)
            match_ratio = skill_word_matches / len(name_words)
            
            if match_ratio >= 0.5:
                logger.info(f"Rejected '{potential_name}' - {skill_word_matches}/{len(name_words)} words match skills ({match_ratio:.0%})")
                return True
    
    return False


def find_section_boundaries(text: str) -> Dict[str, any]:
    """
    Find line numbers where sections start.
    
    Returns:
        Dictionary with:
        - experience_start: Line number where EXPERIENCE section starts (None if not found)
        - skills_sections: List of line numbers where SKILLS sections start
    """
    lines = text.split('\n')
    boundaries = {
        'experience_start': None,
        'skills_sections': []
    }
    
    # Patterns for EXPERIENCE sections (stop scanning here)
    experience_patterns = [
        r'^PROFESSIONAL\s+EXPERIENCE\s*$',
        r'^WORK\s+EXPERIENCE\s*$',
        r'^EXPERIENCE\s*$',
        r'^EMPLOYMENT\s+HISTORY\s*$',
        r'^CAREER\s+HISTORY\s*$'
    ]
    
    # Patterns for SKILLS sections (skip these lines)
    skill_patterns = [
        r'^TECHNICAL\s+SKILLS\s*$',
        r'^SKILLS\s*$',
        r'^TOOLS\s*$',
        r'^TECHNOLOGIES\s*$',
        r'^KEY\s+SKILLS\s*$',
        r'^CORE\s+SKILLS\s*$',
        r'^PROFESSIONAL\s+SKILLS\s*$'
    ]
    
    for i, line in enumerate(lines):
        line_upper = line.strip().upper()
        
        # Check for EXPERIENCE section (stop scanning here)
        if not boundaries['experience_start']:
            for pattern in experience_patterns:
                if re.match(pattern, line_upper):
                    boundaries['experience_start'] = i
                    logger.debug(f"Found EXPERIENCE section at line {i}: '{line.strip()}'")
                    break
        
        # Check for SKILLS sections (skip these lines)
        for pattern in skill_patterns:
            if re.match(pattern, line_upper):
                boundaries['skills_sections'].append(i)
                logger.debug(f"Found SKILLS section at line {i}: '{line.strip()}'")
                break
    
    return boundaries


def extract_name(text, extracted_skills: Optional[Set[str]] = None):
    """
    Extract candidate's full name from text using rule-based pattern matching.
    
    Rules:
    1. Scan only the first 8-10 lines of the resume (or until EXPERIENCE section)
    2. Strip prefixes like "Resume of", "CV", "Profile"
    3. Skip lines that contain COMMON_TITLES, '@', 'linkedin', 'http', '.com', 
       start with '.', or contain digits
    4. Skip lines that match extracted skills (exact match or ≥50% word match)
    5. Skip lines in SKILLS sections
    6. Accept names only if they have 2-4 words, are Title Case or ALL CAPS,
       and contain only alphabets and spaces
    7. Once a valid name is found, STOP processing further lines
    
    Args:
        text: Resume text to extract name from
        extracted_skills: Optional set of normalized skills to filter against
        
    Returns:
        Extracted name string or None if not found
    """
    if not text:
        return None
    
    # Find section boundaries
    boundaries = find_section_boundaries(text)
    
    lines = text.split('\n')
    
    # Determine max line to scan:
    # - If EXPERIENCE section found, stop there (but scan up to it)
    # - If no EXPERIENCE found, scan up to 25 lines (increased from 10 to handle names after SUMMARY)
    # - Continue scanning even if SUMMARY appears (per requirements)
    if boundaries['experience_start']:
        max_line = boundaries['experience_start']
    else:
        # No EXPERIENCE section found - scan more lines to catch names after SUMMARY
        max_line = min(25, len(lines))  # Increased limit to handle names after SUMMARY
    
    # Analyze lines up to max_line
    for i, line in enumerate(lines[:max_line]):
        # Skip if line is in SKILLS section
        if i in boundaries['skills_sections']:
            logger.debug(f"Skipping line {i} - in SKILLS section")
            continue
        original_line = line.strip()
        
        # Skip empty lines
        if not original_line:
            continue
        
        # Skip lines starting with '.' (e.g., ".Net Developer")
        if original_line.startswith('.'):
            continue
        
        # Skip lines containing digits, '@', 'linkedin', 'http', '.com'
        skip_patterns = ['@', 'linkedin', 'http', '.com', 'www.', 'github.com']
        if any(pattern in original_line.lower() for pattern in skip_patterns) or re.search(r'\d', original_line):
            continue
        
        # Strip prefixes like "Resume of", "CV", "Profile"
        line = strip_prefixes(original_line)
        
        # Skip if line is empty after stripping
        if not line:
            continue
        
        # Handle lines with commas - take only the part before first comma
        # (name is usually before first comma, rest might be designation/certifications)
        if ',' in line:
            line = line.split(',')[0].strip()
            if not line:
                continue
        
        # Now check if the remaining line contains COMMON_TITLES (after stripping prefixes and commas)
        # This prevents skipping lines like "Resume of Sesha Reddy" where "Sesha Reddy" is the name
        should_skip_due_to_title = False
        if COMMON_TITLES:
            line_lower = line.lower()
            # Check if the entire line (or significant part) matches a common title
            # Only skip if the line IS a common title, not just contains one
            line_normalized = re.sub(r'[^\w\s]', ' ', line_lower)
            line_normalized = re.sub(r'\s+', ' ', line_normalized).strip()
            
            # Skip if the line exactly matches or is very similar to a common title
            for title in COMMON_TITLES:
                title_lower = title.lower().strip()
                # Skip if line exactly matches title or line is just the title
                if line_normalized == title_lower or line_lower == title_lower:
                    should_skip_due_to_title = True
                    break
                # Skip if line starts with title followed by only punctuation/whitespace
                if line_lower.startswith(title_lower) and len(line_lower) <= len(title_lower) + 2:
                    should_skip_due_to_title = True
                    break
        
        if should_skip_due_to_title:
            continue
        
        # Check if the line (or first part of it) is a valid name
        words = line.split()
        
        # Try different name lengths (4 to 2 words, prefer longer names)
        for name_length in range(min(4, len(words)), 1, -1):
            potential_name = ' '.join(words[:name_length])
            
            # Check if potential name is valid
            if not is_valid_name(potential_name):
                logger.debug(f"Skipping '{potential_name}' - failed name validation")
                continue
            
            # CRITICAL: Check if potential name matches extracted skills or TECH_SKILLS
            # This prevents tools like "Visual Studio" from being extracted as names
            if is_skill_match(potential_name, extracted_skills):
                logger.info(f"Rejected potential name '{potential_name}' (line {i+1}) - matches skills/tools")
                continue
            
            # NEW: Check if next line is also a skill (reject if so)
            if i + 1 < len(lines) and extracted_skills:
                next_line = lines[i + 1].strip()
                if next_line and is_skill_match(next_line, extracted_skills):
                    logger.info(f"Rejected potential name '{potential_name}' (line {i+1}) - next line is a skill: '{next_line}'")
                    continue
            
            # Found valid name - return immediately and stop processing
            logger.info(f"Found valid candidate name on line {i+1}: '{potential_name}'")
            return potential_name
    
    return None


def validate_extracted_name(full_name: str) -> bool:
    """
    Validate if the extracted name is valid.
    
    Uses the same validation logic as is_valid_name().
    
    Args:
        full_name: The name extracted
        
    Returns:
        True if valid, False if invalid
    """
    return is_valid_name(full_name)


def extract_name_with_ai(
    text: str,
    ai_client: Any = None,
    ai_model: str = None,
    fallback_extractor: Optional[Any] = None,
    nlp: Optional[Any] = None,
    extracted_skills: Optional[Set[str]] = None
) -> Optional[str]:
    """
    Extract candidate name from resume text using rule-based extraction.
    
    This function maintains compatibility with the old AI-based interface but uses
    rule-based extraction instead. AI parameters are ignored.
    
    Args:
        text: Resume text to extract name from
        ai_client: Ignored (maintained for compatibility)
        ai_model: Ignored (maintained for compatibility)
        fallback_extractor: Optional function/method for regex-based fallback extraction
        nlp: Optional spaCy NLP model for final fallback
        extracted_skills: Optional set of normalized skills to filter against
        
    Returns:
        Extracted name string or None if not found
    """
    try:
        # Use rule-based extraction as primary method (with skill filtering)
        extracted_name = extract_name(text, extracted_skills=extracted_skills)
        
        if extracted_name and validate_extracted_name(extracted_name):
            logger.info(f"Rule-based extraction found valid name: {extracted_name}")
            return extracted_name
        
        # Try fallback extractor if provided and rule-based extraction failed
        if fallback_extractor:
            try:
                fallback_name = fallback_extractor(text)
                if fallback_name and fallback_name != 'Unknown':
                    # Clean the fallback name (remove extra whitespace, newlines, etc.)
                    fallback_name = ' '.join(fallback_name.split()).strip()
                    
                    # Validate the fallback name
                    if validate_extracted_name(fallback_name):
                        logger.info(f"Fallback extraction found valid name: {fallback_name}")
                        return fallback_name
                    else:
                        logger.debug(f"Fallback extraction found name but validation failed: '{fallback_name}'")
                        # If validation fails but name looks reasonable (2-4 words, no URLs/emails), still use it
                        # This handles edge cases where validation is too strict
                        words = fallback_name.split()
                        if (2 <= len(words) <= 4 and 
                            len(fallback_name) < 70 and
                            not re.search(r'https?://|www\.|\.com|@', fallback_name, re.IGNORECASE) and
                            not re.search(r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', fallback_name) and
                            all(word.replace('.', '').replace(',', '').replace("'", '').replace('-', '').isalpha() for word in words)):
                            logger.info(f"Fallback extraction found reasonable name (validation bypassed): {fallback_name}")
                            return fallback_name
            except Exception as e:
                logger.warning(f"Fallback extraction failed: {e}")
        
        # Last resort: try NLP if available
        if nlp:
            try:
                logger.info("Trying NLP fallback...")
                doc = nlp(text[:500])
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        ent_text = ent.text.strip()
                        
                        # Reject if contains URL patterns, email, phone, etc.
                        if (re.search(r'https?://|www\.|\.com|\.org|\.net|\.io|linkedin\.com|github\.com', ent_text, re.IGNORECASE) or
                            '@' in ent_text or
                            re.search(r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', ent_text) or
                            re.search(r'[•\|\/\\]', ent_text)):
                            logger.debug(f"Rejected NLP entity (contains invalid patterns): {ent_text}")
                            continue
                        
                        # Validate the extracted name using our validation rules
                        if validate_extracted_name(ent_text):
                            logger.info(f"Found name via NLP: {ent_text}")
                            return ent_text
                        else:
                            logger.debug(f"Rejected NLP entity (failed validation): {ent_text}")
            except Exception as e:
                logger.warning(f"NLP fallback failed: {e}")
        
        logger.warning("Name extraction failed with all methods")
        return None
        
    except Exception as e:
        logger.error(f"Name extraction failed: {e}")
        return None


def main():
    """Main function to read input and extract name."""
    # Read from stdin or file
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = sys.stdin.read()
    
    name = extract_name(text)
    
    if name:
        print(name)
    else:
        # If no name found, output nothing (or empty string)
        pass


if __name__ == '__main__':
    main()
