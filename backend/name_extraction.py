"""
Rule-based Name Extraction Module for Resume Parsing.

Extracts candidate names from resumes using rule-based pattern matching
without AI dependencies.
"""

import re
import sys
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


def clean_line(line):
    """Remove commas, hyphens, brackets, tabs, and multiple spaces."""
    # Replace tabs with spaces
    line = line.replace('\t', ' ')
    # Remove commas, hyphens, brackets
    line = re.sub(r'[,\(\)\[\]\{\}]', '', line)
    line = re.sub(r'-', ' ', line)
    # Replace multiple spaces with single space
    line = re.sub(r'\s+', ' ', line)
    # Strip leading/trailing whitespace
    return line.strip()


def contains_ignore_keywords(line):
    """Check if line contains keywords to ignore."""
    ignore_keywords = [
        'resume', 'cv', 'curriculum', 'profile', 'summary',
        'engineer', 'manager', 'developer', 'consultant', 'analyst',
        'director', 'lead', 'senior',
        'email', 'phone', 'mobile', 'whatsapp',
        'linkedin', 'link', 'github', 'contact', 'location', 'objective'
    ]
    line_lower = line.lower()
    return any(keyword in line_lower for keyword in ignore_keywords)


def has_invalid_patterns(line):
    """Check for digits, @, or phone number patterns."""
    # Check for digits
    if re.search(r'\d', line):
        return True
    # Check for @ symbol
    if '@' in line:
        return True
    # Check for phone number patterns (e.g., +1234567890, (123) 456-7890, etc.)
    phone_patterns = [
        r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}',
        r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
    ]
    for pattern in phone_patterns:
        if re.search(pattern, line):
            return True
    return False


def has_metadata_indicators(text):
    """Check if text contains metadata indicators like URLs, colons with labels, etc."""
    text_lower = text.lower()
    # Check for URL patterns
    if re.search(r'https?://|www\.|\.com|\.org|\.net|\.io|linkedin\.com|github\.com', text_lower):
        return True
    # Check for colon-separated labels (e.g., "Link :", "Email :")
    if re.search(r'\w+\s*:\s*\w', text_lower):
        return True
    return False


def is_certification_or_acronym(word):
    """Check if a word is likely a certification or acronym (not part of name)."""
    # Common certifications and acronyms (all uppercase, 2-5 letters)
    common_certs = {
        'pmp', 'csm', 'cspo', 'pmi', 'aws', 'azure', 'gcp', 'itil', 'scrum', 
        'agile', 'six', 'sigma', 'lean', 'mba', 'phd', 'ms', 'bs', 'ba', 'ma',
        'cissp', 'ceh', 'cisa', 'cisco', 'ccna', 'ccnp', 'mcp', 'mcsd', 'ocp',
        'java', 'python', 'sql', 'html', 'css', 'js', 'api', 'rest', 'soap'
    }
    
    # If it's a known certification/acronym
    if word.upper() in common_certs:
        return True
    
    # If it's all uppercase and 2-5 letters (likely acronym)
    if word.isupper() and 2 <= len(word) <= 5 and word.isalpha():
        return True
    
    return False


def is_valid_name(line):
    """Check if line is a valid name according to rules."""
    if not line:
        return False
    
    # Split into words
    words = line.split()
    
    # Must have 1 to 4 words
    if len(words) < 1 or len(words) > 4:
        return False
    
    # Check each word
    for word in words:
        # Word can contain letters, periods, or apostrophes
        # Pattern: letters, periods, apostrophes only
        if not re.match(r'^[a-zA-Z\.\']+$', word):
            return False
    
    return True


def extract_name(text):
    """
    Extract candidate's full name from text using rule-based pattern matching.
    
    Args:
        text: Resume text to extract name from
        
    Returns:
        Extracted name string or None if not found
    """
    if not text:
        return None
        
    lines = text.split('\n')
    
    # Analyze only first 8 lines
    for i, line in enumerate(lines[:8]):
        # Clean the line
        cleaned = clean_line(line)
        
        if not cleaned:
            continue
        
        # Handle commas - split on commas and only process the first part (name is usually before first comma)
        # Also check if remaining parts are certifications/acronyms and exclude them
        parts = line.split(',')
        if parts:
            # Take first part as potential name
            line_to_process = parts[0].strip()
            # Check if remaining parts are certifications - if so, we can be confident first part is the name
            remaining_parts = [p.strip() for p in parts[1:]]
            has_certifications = False
            if remaining_parts:
                # Check if ALL remaining parts are certifications/acronyms
                all_are_certs = True
                for part in remaining_parts:
                    part_words = part.split()
                    if part_words:
                        # Check if all words in this part are certifications/acronyms
                        if not all(is_certification_or_acronym(word) for word in part_words if word):
                            all_are_certs = False
                            break
                    else:
                        all_are_certs = False
                        break
                has_certifications = all_are_certs
        else:
            line_to_process = cleaned
            has_certifications = False
        
        line_to_process = clean_line(line_to_process)
        
        # Try to extract name from the beginning of the line
        # (Don't skip the whole line if it contains ignore keywords - 
        #  the name might be at the start before the keywords)
        # Split into words and try first 1-4 words as potential name
        words = line_to_process.split()
        
        # Try different lengths of name (4 to 1 words, prefer longer names)
        for name_length in range(min(4, len(words)), 0, -1):
            if name_length == 0:
                break
            potential_name = ' '.join(words[:name_length])
            remaining_text = ' '.join(words[name_length:])
            
            # Check if potential name is valid
            if not is_valid_name(potential_name):
                continue
            
            # Don't accept if the potential name itself contains ignore keywords
            if contains_ignore_keywords(potential_name):
                continue
            
            # Check if the next word after potential name is a certification/acronym
            # If so, stop here - we've found the name
            if name_length < len(words):
                next_word = words[name_length]
                if is_certification_or_acronym(next_word):
                    return potential_name
            
            # If we detected certifications after comma in original line, and we have a valid name,
            # return it immediately (don't check remaining text)
            if has_certifications and is_valid_name(potential_name) and not contains_ignore_keywords(potential_name):
                return potential_name
            
            # If no remaining text, whole line is the name (but double-check it's not an ignore keyword)
            if not remaining_text:
                # Make sure the name itself isn't just an ignore keyword
                if not contains_ignore_keywords(potential_name):
                    return potential_name
                # If it's an ignore keyword, continue to next iteration
                continue
            
            # Check if remaining text contains ignore keywords, invalid patterns, or metadata indicators
            # If the remaining text has these, it's likely the name is at the start and rest is metadata
            if (contains_ignore_keywords(remaining_text) or 
                has_invalid_patterns(remaining_text) or 
                has_metadata_indicators(remaining_text)):
                # Name is valid and rest has metadata - this is likely the name
                return potential_name
        
        # If no name found by extracting from start, check if whole line is valid
        # (but skip if it has invalid patterns or ignore keywords)
        # CRITICAL: If line has commas, only check first part (before first comma)
        if ',' in line:
            # Already processed first part above, skip whole line check
            continue
        
        if not has_invalid_patterns(cleaned):
            if is_valid_name(cleaned) and not contains_ignore_keywords(cleaned):
                # Final check: ensure no certifications in the name
                words = cleaned.split()
                # If any word is a certification, exclude it and everything after
                for i, word in enumerate(words):
                    if is_certification_or_acronym(word):
                        # Return only the part before the certification
                        if i > 0:
                            name_before_cert = ' '.join(words[:i])
                            if is_valid_name(name_before_cert) and not contains_ignore_keywords(name_before_cert):
                                return name_before_cert
                        break
                return cleaned
    
    return None


def validate_extracted_name(full_name: str) -> bool:
    """
    Validate if the extracted name is valid (not a section header, degree, etc.).
    
    Args:
        full_name: The name extracted
        
    Returns:
        True if valid, False if invalid
    """
    if not full_name or not isinstance(full_name, str):
        return False
    
    full_name = full_name.strip()
    if not full_name:
        return False
    
    # CRITICAL: Check if name contains certifications/acronyms - reject if found
    words = full_name.split()
    for word in words:
        if is_certification_or_acronym(word):
            return False  # Name contains certification - invalid
    
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
    ai_client: Any = None,
    ai_model: str = None,
    fallback_extractor: Optional[Any] = None,
    nlp: Optional[Any] = None
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
        
    Returns:
        Extracted name string or None if not found
    """
    try:
        # Use rule-based extraction as primary method
        extracted_name = extract_name(text)
        
        if extracted_name and validate_extracted_name(extracted_name):
            logger.info(f"Rule-based extraction found valid name: {extracted_name}")
            return extracted_name
        
        # Try fallback extractor if provided and rule-based extraction failed
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
                logger.info("Trying NLP fallback...")
                doc = nlp(text[:500])
                for ent in doc.ents:
                    if ent.label_ == "PERSON" and len(ent.text.split()) <= 4:
                        if validate_extracted_name(ent.text):
                            logger.info(f"Found name via NLP: {ent.text}")
                            return ent.text
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
