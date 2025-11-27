import re
import logging
from typing import List, Optional

# Database imports
try:
    from ats_database import ATSDatabase
except ImportError:
    ATSDatabase = None

logger = logging.getLogger(__name__)


class EducationExtractor:
    """
    Standalone Education Extractor - Extracts only real degrees with full degree names and specializations.
    Ignores trainings, certifications, workshops.
    
    Usage:
        extractor = EducationExtractor(resume_text)
        education_list = extractor.extract()
    """
    
    # Valid education section headings
    EDUCATION_HEADINGS = [
        r'education', r'educational background', r'academics', r'qualification',
        r'qualifications', r'academic details', r'academic summary',
        r'scholastic profile', r'educational credentials', r'education summary',
        r'education & training'
    ]
    
    # Valid experience section headings (used to find section boundaries)
    VALID_EXPERIENCE_HEADINGS = [
        r'work experience', r'professional experience', r'experience', 
        r'employment history', r'job history', r'internships', 
        r'apprenticeship', r'freelance work', r'contract work'
    ]
    
    # Skills section headings (used to find section boundaries)
    SKILLS_HEADINGS = [
        r'skills', r'key skills', r'technical skills', r'technical skill', r'core skills', r'skill set',
        r'technical expertise', r'technologies', r'programming languages', r'competencies', r'proficiencies'
    ]
    
    # Education priority order (lower number = higher priority)
    EDUCATION_PRIORITY = {
        # PhD / Doctorate - Highest (Priority 1)
        'phd': 1, 'ph.d': 1, 'ph d': 1, 'doctorate': 1, 'doctor of philosophy': 1,
        
        # Master's (Priority 2)
        'm.tech': 2, 'mtech': 2, 'm tech': 2, 'm.e': 2, 'me': 2,
        'm.sc': 2, 'msc': 2, 'm sc': 2, 'm.a': 2, 'ma': 2,
        'm.com': 2, 'mcom': 2, 'mba': 2, 'm.b.a': 2, 'mca': 2, 'm.c.a': 2,
        'ms': 2, 'master': 2, 'master of technology': 2, 'master of engineering': 2,
        'master of science': 2, 'master of arts': 2, 'master of commerce': 2,
        'master of business administration': 2, 'master of computer applications': 2,
        
        # Bachelor's (Priority 3)
        'b.tech': 3, 'btech': 3, 'b tech': 3, 'b.e': 3, 'be': 3,
        'b.sc': 3, 'bsc': 3, 'b sc': 3, 'b.a': 3, 'ba': 3,
        'b.com': 3, 'bcom': 3, 'bba': 3, 'b.b.a': 3, 'bca': 3, 'b.c.a': 3,
        'bachelor': 3, 'bachelor of technology': 3, 'bachelor of engineering': 3,
        'bachelor of science': 3, 'bachelor of arts': 3, 'bachelor of commerce': 3,
        'bachelor of business administration': 3, 'bachelor of computer applications': 3,
        'bachelor of computer science': 3, 'bachelor of electrical': 3, 'bachelor of electronics': 3,
        'bachelor of mechanical': 3, 'bachelor of civil': 3, 'bachelor of information technology': 3,
        
        # Diploma / Polytechnic (Priority 4)
        'diploma': 4, 'polytechnic': 4, 'pgdca': 4, 'pg diploma': 4,
        'post graduate diploma': 4,
        
        # 12th / Intermediate (Priority 5)
        '12th': 5, 'hsc': 5, 'intermediate': 5, '10+2': 5,
        'a-levels': 5, 'a levels': 5, 'cbse 12': 5, 'icse 12': 5,
        
        # 10th / SSC - Lowest (Priority 6)
        '10th': 6, 'ssc': 6, 'sslc': 6, 'cbse 10': 6, 'icse 10': 6,
        'o-levels': 6, 'o levels': 6, 'igcse': 6,
    }
    
    def __init__(self, resume_text: str, store_in_db: bool = False, candidate_id: Optional[int] = None):
        """
        Initialize the education extractor with resume text.
        
        Args:
            resume_text: The resume text to parse
            store_in_db: If True, store extracted education in database
            candidate_id: Candidate ID for database storage (required if store_in_db=True)
        """
        self.resume_text = resume_text
        self.lines = [line.strip() for line in resume_text.split('\n') if line.strip()]
        self.store_in_db = store_in_db
        self.candidate_id = candidate_id
    
    def _is_extraction_valid(self, education_list: List[str]) -> bool:
        """
        Validate if Python extraction is correct.
        
        Args:
            education_list: List of extracted education degrees
            
        Returns:
            True if extraction is valid, False otherwise
        """
        if not education_list:
            logger.warning("Python extraction returned empty list")
            return False
        
        # Check if any entry contains valid degree patterns
        degree_patterns = [
            r'\bB\.?\s*Tech\b', r'\bM\.?\s*Tech\b', r'\bB\.?E\.?\b', r'\bM\.?E\.?\b',
            r'\bBachelor\b', r'\bMaster\b', r'\bB\.?\s*Sc\.?\b', r'\bM\.?\s*Sc\.?\b',
            r'\bB\.?\s*Com\.?\b', r'\bM\.?\s*Com\.?\b', r'\bB\.?\s*A\.?\b', r'\bM\.?\s*A\.?\b',
            r'\bM\.?\s*C\.?\s*A\.?\b', r'\bB\.?\s*C\.?\s*A\.?\b', r'\bM\.?\s*B\.?\s*A\.?\b',
            r'\bB\.?\s*B\.?\s*A\.?\b', r'\bPh\.?\s*D\.?\b', r'\bDiploma\b', r'\bP\.?\s*G\.?\s*D\.?\s*C\.?\s*A\.?\b'
        ]
        
        for edu in education_list:
            edu_lower = edu.lower()
            # Check if it's too short (likely invalid)
            if len(edu.strip()) < 5:
                continue
            
            # Check if it contains a valid degree pattern
            for pattern in degree_patterns:
                if re.search(pattern, edu, re.IGNORECASE):
                    logger.info(f"Python extraction validated: Found valid degree in '{edu}'")
                    return True
        
        logger.warning(f"Python extraction invalid: No valid degree patterns found in {education_list}")
        return False
    
    def _get_education_priority(self, education: str) -> int:
        """
        Get priority rank for an education entry.
        Lower number = higher priority (PhD=1, 10th=6).
        
        Args:
            education: Education string to check
            
        Returns:
            Priority number (1-6), or 99 if not found
        """
        edu_lower = education.lower()
        
        # Check each priority keyword
        for keyword, priority in self.EDUCATION_PRIORITY.items():
            if keyword in edu_lower:
                return priority
        
        return 99  # Unknown education type
    
    def _sort_by_priority(self, education_list: List[str]) -> List[str]:
        """
        Sort education list by priority (highest degree first).
        
        Args:
            education_list: List of education strings
            
        Returns:
            Sorted list with highest education first
        """
        return sorted(education_list, key=lambda x: self._get_education_priority(x))
    
    def get_highest_education(self) -> Optional[str]:
        """
        Extract and return only the highest education degree.
        
        Returns:
            Highest education string, or None if not found
        """
        education_list = self.extract()
        if education_list:
            return education_list[0]  # Already sorted by priority
        return None
    
    def _store_in_database(self, education: str, education_details: str):
        """
        Store extracted education in database.
        
        Args:
            education: Highest education degree
            education_details: Full education details
        """
        if not self.store_in_db:
            return
        
        if not ATSDatabase or not self.candidate_id:
            logger.warning("Database storage requested but ATSDatabase not available or candidate_id missing")
            return
        
        try:
            db = ATSDatabase()
            if not db.connect():
                logger.error("Failed to connect to database for education storage")
                return
            
            # Update education fields for the candidate
            query = """
                UPDATE resume_metadata 
                SET education = %s, education_details = %s
                WHERE candidate_id = %s
            """
            
            db.cursor.execute(query, (education, education_details, self.candidate_id))
            db.connection.commit()
            logger.info(f"Stored education in database for candidate_id: {self.candidate_id}")
            db.disconnect()
            
        except Exception as e:
            logger.error(f"Failed to store education in database: {e}")
    
    def find_section(self, headings: List[str]) -> Optional[int]:
        """
        Find the line index where a section starts.
        
        Args:
            headings: List of heading patterns to search for
            
        Returns:
            Line index where section starts, or None if not found
        """
        for i, line in enumerate(self.lines):
            line_lower = line.lower().strip()
            # Remove trailing colons, dashes, etc. for matching
            line_clean = re.sub(r'[:–\-•\s]+$', '', line_lower)
            for heading in headings:
                # Check if line starts with heading or contains it as a major section header
                # Section headers are usually standalone or at the start
                heading_clean = heading.strip()
                if re.search(rf'^{re.escape(heading_clean)}\b', line_clean, re.IGNORECASE) or \
                   (len(line_clean.split()) <= 3 and heading_clean in line_clean):
                    return i
        return None
    
    def extract(self) -> List[str]:
        """
        Extract only REAL degrees with full degree names and specializations.
        Ignores trainings, certifications, workshops.
        
        Optionally stores result in database.
        
        Returns:
            List of extracted education degrees (cleaned and deduplicated)
        """
        # Step 1: Try Python extraction
        education_list = []
        section_start = self.find_section(self.EDUCATION_HEADINGS)
        
        if section_start is None:
            logger.warning("No education section found in resume")
        else:
            # Find where education section ends (next major section or end of resume)
            section_end = len(self.lines)
            for i in range(section_start + 1, len(self.lines)):
                line_lower = self.lines[i].lower()
                # Check if it's another major section
                if any(re.search(rf'^{re.escape(heading)}', line_lower, re.IGNORECASE) 
                       for heading in self.VALID_EXPERIENCE_HEADINGS + self.SKILLS_HEADINGS):
                    section_end = i
                    break
            
            # Extract education from this section
            education_lines = self.lines[section_start + 1:section_end]
            
            # Advanced degree patterns - handles various formats with/without spaces and periods
            degree_patterns = [
                # Full form degrees with specialization (MUST BE FIRST - more specific patterns)
                r'\bBachelor\s+of\s+[A-Za-z][A-Za-z\s&]+(?:Engineering|Science|Technology|Arts|Commerce|Applications|Administration|Studies)\b',
                r'\bMaster\s+of\s+[A-Za-z][A-Za-z\s&]+(?:Engineering|Science|Technology|Arts|Commerce|Applications|Administration|Studies)\b',
                r'\bBachelor\s+of\s+(?:Computer\s+Science|Electrical|Electronics|Mechanical|Civil|Information\s+Technology)\b',
                r'\bMaster\s+of\s+(?:Computer\s+Science|Electrical|Electronics|Mechanical|Civil|Information\s+Technology)\b',
                # Engineering Variants
                r'\bB\s*\.?\s*Tech\b',
                r'\bB\s*\.?\s*E\b',
                r'\bM\s*\.?\s*Tech\b',
                r'\bM\s*\.?\s*E\b',
                # Full form engineering degrees
                r'\bBachelor\s+of\s+Technology\b',
                r'\bBachelor\s+of\s+Engineering\b',
                r'\bMaster\s+of\s+Technology\b',
                r'\bMaster\s+of\s+Engineering\b',
                # Science degrees
                r'\bB\s*\.?\s*Sc\b',
                r'\bM\s*\.?\s*Sc\b',
                r'\bBachelor\s+of\s+Science\b',
                r'\bMaster\s+of\s+Science\b',
                # Commerce degrees
                r'\bB\s*\.?\s*Com\b',
                r'\bM\s*\.?\s*Com\b',
                r'\bBachelor\s+of\s+Commerce\b',
                r'\bMaster\s+of\s+Commerce\b',
                # Arts degrees
                r'\bB\s*\.?\s*A\b',
                r'\bM\s*\.?\s*A\b',
                r'\bBachelor\s+of\s+Arts\b',
                r'\bMaster\s+of\s+Arts\b',
                # Computer applications
                r'\bB\s*\.?\s*C\s*\.?\s*A\b',
                r'\bM\s*\.?\s*C\s*\.?\s*A\b',
                r'\bBachelor\s+of\s+Computer\s+Applications\b',
                r'\bMaster\s+of\s+Computer\s+Applications\b',
                # Management
                r'\bM\s*\.?\s*B\s*\.?\s*A\b',
                r'\bB\s*\.?\s*B\s*\.?\s*A\b',
                r'\bMaster\s+of\s+Business\s+Administration\b',
                r'\bBachelor\s+of\s+Business\s+Administration\b',
                # Doctorate
                r'\bP\s*\.?\s*h\s*\.?\s*D\b',
                r'\bDoctor\s+of\s+Philosophy\b',
                # Diploma Patterns
                r'\bDiploma\s+in\s+[A-Za-z &]+\b',
                r'\bPost\s+Graduate\s+Diploma\b',
                r'\bPG\s*\.?\s*Diploma\b',
                r'\bP\s*\.?\s*G\s*\.?\s*D\s*\.?\s*C\s*\.?\s*A\b',
                r'\bPolytechnic\b',
                # Schooling
                r'\b10\+2\b',
                r'\b12th\b',
                r'\bHSC\b',
                r'\bSSC\b',
                r'\bSSLC\b',
                r'\bCBSE\b',
                r'\bICSE\b',
                r'\bIntermediate\b',
                # Abroad Boards
                r'\bIGCSE\b',
                r'\bA-Levels\b',
                r'\bO-Levels\b',
                # Generic catch-all
                r'\bBachelor\b',
                r'\bMaster\b',
                r'\bDiploma\b',
            ]
            
            # Specialization patterns for branch/major detection
            specialization_patterns = [
                r'\bCSE\b|Computer\s*Science',
                r'\bECE\b|Electronics\s*and\s*Communication',
                r'\bEEE\b|Electrical\s*and\s*Electronics',
                r'\bME\b|Mechanical',
                r'\bCE\b|Civil',
                r'\bIT\b|Information\s*Technology',
                r'\bAI\b|Artificial\s*Intelligence',
                r'\bDS\b|Data\s*Science',
                r'\bML\b|Machine\s*Learning',
                r'\bCyber\b|Cyber\s*Security',
                r'\bCloud\b|Cloud\s*Computing',
            ]
            
            current_degree = []
            for line in education_lines:
                # Skip empty lines
                if not line:
                    if current_degree:
                        degree_text = ' '.join(current_degree)
                        if any(re.search(pattern, degree_text, re.IGNORECASE) for pattern in degree_patterns):
                            education_list.append(degree_text.strip())
                        current_degree = []
                    continue
                
                # Skip lines that are clearly section headers or non-degree content
                line_lower = line.lower()
                if any(word in line_lower for word in ['certification', 'training', 'workshop', 'course']) and not any(re.search(pattern, line, re.IGNORECASE) for pattern in degree_patterns):
                    if current_degree:
                        degree_text = ' '.join(current_degree)
                        if any(re.search(pattern, degree_text, re.IGNORECASE) for pattern in degree_patterns):
                            education_list.append(degree_text.strip())
                        current_degree = []
                    continue
                
                # Check if line contains a degree indicator
                line_lower = line.lower()
                if any(re.search(pattern, line, re.IGNORECASE) for pattern in degree_patterns):
                    if current_degree:
                        degree_text = ' '.join(current_degree)
                        if any(re.search(pattern, degree_text, re.IGNORECASE) for pattern in degree_patterns):
                            education_list.append(degree_text.strip())
                    # Start new degree entry - include the full line to capture specialization
                    # e.g., "BTech, Civil Engineering" should be captured as one entry
                    current_degree = [line]
                elif current_degree:
                    # Continue building current degree entry (but stop at "Career objectives" or similar)
                    # Check if this line looks like it's part of the degree (specialization, etc.)
                    # or if it's a new section (CGPA, university name, etc.)
                    line_lower = line.lower()
                    
                    # Stop if it's clearly a new section or metadata
                    if any(word in line_lower for word in ['career objectives', 'objective', 'certification', 'work history', 'experience']):
                        # Save current degree and stop
                        degree_text = ' '.join(current_degree)
                        if any(re.search(pattern, degree_text, re.IGNORECASE) for pattern in degree_patterns):
                            education_list.append(degree_text.strip())
                        current_degree = []
                    # Stop if line contains CGPA/GPA/percentage (metadata, not part of degree name)
                    elif re.search(r'\b(CGPA|GPA|percentage|%)\b', line_lower):
                        # Save current degree before this metadata line
                        degree_text = ' '.join(current_degree)
                        if any(re.search(pattern, degree_text, re.IGNORECASE) for pattern in degree_patterns):
                            education_list.append(degree_text.strip())
                        current_degree = []
                    # Stop if line looks like a university name (contains "University", "College", etc.)
                    elif re.search(r'\b(University|College|Institute|School)\b', line, re.IGNORECASE):
                        # Save current degree before university name
                        degree_text = ' '.join(current_degree)
                        if any(re.search(pattern, degree_text, re.IGNORECASE) for pattern in degree_patterns):
                            education_list.append(degree_text.strip())
                        current_degree = []
                    # Otherwise, continue building (might be specialization on next line)
                    else:
                        # Only add if it looks like it could be part of the degree (short line, no numbers)
                        # This handles cases like "Civil Engineering" on a separate line
                        if len(line.split()) <= 5 and not re.search(r'\d', line):
                            current_degree.append(line)
                        else:
                            # Save current degree and start fresh
                            degree_text = ' '.join(current_degree)
                            if any(re.search(pattern, degree_text, re.IGNORECASE) for pattern in degree_patterns):
                                education_list.append(degree_text.strip())
                            current_degree = []
                elif len(line.split()) <= 15 and 'from' in line_lower:  # Potential degree line with "from"
                    # Check if it contains a degree pattern
                    if any(re.search(pattern, line, re.IGNORECASE) for pattern in degree_patterns):
                        current_degree = [line]
            
            # Add last degree if exists
            if current_degree:
                degree_text = ' '.join(current_degree)
                if any(re.search(pattern, degree_text, re.IGNORECASE) for pattern in degree_patterns):
                    education_list.append(degree_text.strip())
        
        # Clean and deduplicate
        cleaned = []
        for edu in education_list:
            # Remove years, percentages, GPAs, university names (keep only degree info)
            # Remove year ranges
            edu = re.sub(r'\d{4}[-–]\d{4}', '', edu)
            # Remove standalone years
            edu = re.sub(r'\b\d{4}\b', '', edu)
            # Remove percentages
            edu = re.sub(r'\d+\.?\d*%', '', edu)
            # Remove GPA/CGPA with any trailing characters
            edu = re.sub(r'GPA[:\s]*\d+\.?\d*[/]?\d*[^\w\s]*', '', edu, flags=re.IGNORECASE)
            edu = re.sub(r'CGPA[:\s]*\d+\.?\d*[/]?\d*[^\w\s]*', '', edu, flags=re.IGNORECASE)
            # Remove "from" and everything after it (university name, location, etc.)
            # But keep the degree name before "from"
            if ' from ' in edu.lower():
                parts = re.split(r'\s+from\s+', edu, flags=re.IGNORECASE, maxsplit=1)
                if parts:
                    edu = parts[0].strip()
            # Remove common university indicators (but keep if it's part of degree name)
            # Match university names - look for institution names at the end
            # Pattern: word(s) followed by University/College/Institute/School
            # But be careful not to remove specialization fields (e.g., "Civil Engineering")
            # Only remove if it's clearly a university name (contains "University", "College", etc.)
            edu = re.sub(r'[,\s]+[A-Z][A-Za-z\s]*(?:University|College|Institute|School)[,\s]*.*$', '', edu, flags=re.IGNORECASE)
            edu = re.sub(r',\s*[A-Z][A-Za-z\s]*(?:University|College|Institute|School)[,\s]*.*', '', edu, flags=re.IGNORECASE)
            # Remove standalone university abbreviations like "JNTU" at the end
            # But preserve "Bachelor of X" and "Master of X" patterns
            # Only remove short acronyms (2-6 letters) that are NOT part of degree name
            if not re.search(r'\b(?:Bachelor|Master)\s+of\s+', edu, re.IGNORECASE):
                edu = re.sub(r'\s+[A-Z]{2,6}\s*$', '', edu)  # Remove 2-6 letter acronyms at end
            # Remove everything after "Career objectives" or similar sections
            edu = re.sub(r'\s+Career objectives.*$', '', edu, flags=re.IGNORECASE)
            edu = re.sub(r'\s+Objective.*$', '', edu, flags=re.IGNORECASE)
            # Clean up extra spaces
            edu = ' '.join(edu.split()).strip()
            # Fix missing spaces after commas (e.g., "BTech,CivilEngineering" -> "BTech, Civil Engineering")
            # Apply multiple times to handle all commas
            while re.search(r',([A-Za-z])', edu):
                edu = re.sub(r',([A-Za-z])', r', \1', edu)  # Add space after comma if missing
            # Remove trailing commas, dashes, and other punctuation (but keep periods that are part of abbreviations)
            # IMPORTANT: Don't remove commas in the middle (e.g., "BTech, Civil Engineering" should keep the comma)
            edu = re.sub(r'[,–\-]+$', '', edu).strip()  # Only remove trailing punctuation
            # Remove standalone single letters at the end (often artifacts from GPA removal)
            # But be careful not to remove letters that are part of abbreviations like "B.Tech"
            edu = re.sub(r'\s+[A-Z]\s*$', '', edu)  # Only remove if it's a standalone word at the end

            # Strip any lingering leading/trailing punctuation characters (., *, -, etc.)
            edu = re.sub(r'^[\W_]+', '', edu)
            edu = re.sub(r'[\W_]+$', '', edu)

            # Skip entries that are effectively empty or purely punctuation after cleaning
            alnum_payload = re.sub(r'[^A-Za-z0-9]+', '', edu)

            if edu and len(edu) > 3 and alnum_payload and edu not in cleaned:  # Filter out very short/punctuation artifacts
                cleaned.append(edu)
        
        # Step 2: Sort by priority (highest education first)
        cleaned = self._sort_by_priority(cleaned)
        
        # Step 3: Store in database if requested
        if self.store_in_db and cleaned:
            highest_degree = cleaned[0] if cleaned else None
            education_details = '\n'.join(cleaned) if cleaned else ''
            self._store_in_database(highest_degree, education_details)
        
        return cleaned


def extract_education(
    resume_text: str, 
    store_in_db: bool = False, 
    candidate_id: Optional[int] = None
) -> List[str]:
    """
    Convenience function to extract education from resume text.
    
    Args:
        resume_text: The resume text to parse
        store_in_db: If True, store extracted education in database
        candidate_id: Candidate ID for database storage (required if store_in_db=True)
        
    Returns:
        List of extracted education degrees
        
    Example:
        >>> resume = \"\"\"
        ... EDUCATION
        ... B.Tech in Computer Science
        ... XYZ University, 2018-2022
        ... \"\"\"
        >>> degrees = extract_education(resume)
        >>> print(degrees)
        ['B.Tech in Computer Science']
    """
    extractor = EducationExtractor(
        resume_text, 
        store_in_db=store_in_db,
        candidate_id=candidate_id
    )
    return extractor.extract()


if __name__ == "__main__":
    # Example usage
    sample_resume = """
    John Doe
    john.doe@email.com | +1-234-567-8900
    
    SUMMARY
    Experienced software developer with 5 years of experience...
    
    EDUCATION
    B.Tech 
    XYZ University, 2018-2022
    CGPA: 8.5/10
    
    M.Tech in Data Science
    ABC University, 2022-2024
    
    WORK EXPERIENCE
    Senior Python Developer
    ABC Company
    Jan 2020 - Present
    
    SKILLS
    Python, Django, React, SQL, AWS, Docker
    """
    
    # Using the class
    extractor = EducationExtractor(sample_resume)
    education = extractor.extract()
    print("Using class:")
    print(education)
    
    # Using the convenience function
    education2 = extract_education(sample_resume)
    print("\nUsing function:")
    print(education2)
    
    # Example with database storage (requires candidate_id)
    # education3 = extract_education(sample_resume, store_in_db=True, candidate_id=1)

