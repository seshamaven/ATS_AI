"""
Education Extractor for Resumes
Keyword-Based Section Extraction approach
Optimized for 180k+ resumes with various formats (PDF, DOCX, DOC)
"""

import os
import re
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

# Database imports
try:
    from ats_database import ATSDatabase
except ImportError:
    ATSDatabase = None

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    filename: str
    education: Optional[str] = None  # Simple string like "BSC computers"
    error: Optional[str] = None


class EducationExtractor:
    """Extract education information from resumes using keyword-based section extraction."""
    
    # Section start keywords (education section headers)
    EDUCATION_HEADERS = [
        r'\beducation\b',
        r'\beducational\s*(background|details|qualification|qualifications)?\b',
        r'\bacademic\s*(background|credentials|qualifications?)?\b',
        r'\bscholastic\s*record\b',
        r'\beducation\s*(&|and)\s*(training|certifications?)\b',
    ]
    
    # Section end keywords (next section headers)
    SECTION_ENDS = [
        r'\b(professional\s+)?experience\b',
        r'\bwork\s*(history|experience)\b',
        r'\bemployment\s*(history)?\b',
        r'\bcareer\s*(experience|history)?\b',
        r'\bskills?\b',
        r'\btechnical\s+skills?\b',
        r'\bcore\s+competenc(y|ies)\b',
        r'\bprojects?\b',
        r'\bcertification[s]?\b',
        r'\bprofessional\s+summary\b',
        r'\bsummary\b',
        r'\bobjective\b',
        r'\bachievements?\b',
        r'\bawards?\b',
        r'\bpublications?\b',
        r'\breferences?\b',
        r'\bpersonal\s*(details|information)?\b',
        r'\bdeclaration\b',
        r'\bvolunteer\b',
        r'\bactivities\b',
        r'\binterests?\b',
        r'\blanguages?\b',
        r'\baffiliations?\b',
        r'\bwork\s+preferences\b',
        r'\bprofile\s+sources\b',
        r'\bcontact\s+information\b',
    ]
    
    # Degree patterns with hierarchy level (higher number = higher education)
    # Level: PhD=7, Master=6, Bachelor=5, Associate=4, Diploma=3, Certificate=2, HighSchool=1
    DEGREE_PATTERNS = [
        # Doctorates (Level 7)
        (r'\b(Ph\.?D\.?|Doctorate|Doctor\s+of\s+Philosophy)\b', 'PhD', 7),
        # Masters (Level 6)
        (r'\b(M\.?S\.?|Master\s*(?:\'?s)?\s*(?:of\s+)?Science)\b', 'M.S.', 6),
        (r'\b(M\.?A\.?|Master\s*(?:\'?s)?\s*(?:of\s+)?Arts)\b', 'M.A.', 6),
        (r'\b(M\.?B\.?A\.?|Master\s*(?:\'?s)?\s*(?:of\s+)?Business\s+Administration)\b', 'MBA', 6),
        (r'\b(M\.?Tech\.?|Master\s*(?:\'?s)?\s*(?:of\s+)?Technolog?y?)\b', 'M.Tech', 6),
        (r'\b(M\.?E\.?|Master\s*(?:\'?s)?\s*(?:of\s+)?Engineering)\b', 'M.E.', 6),
        (r'\b(M\.?C\.?A\.?|Master\s*(?:\'?s)?\s*(?:of\s+)?Computer\s+Applications?)\b', 'MCA', 6),
        (r'\bMaster(?:\'?s)?\s+(?:of\s+|in\s+)?(\w+(?:\s+\w+)*)', 'Master', 6),
        # Bachelors (Level 5)
        (r'\b(B\.?S\.?|Bachelor\s*(?:\'?s)?\s*(?:of\s+)?Science)\b', 'B.S.', 5),
        (r'\b(B\.?A\.?|Bachelor\s*(?:\'?s)?\s*(?:of\s+)?Arts)\b', 'B.A.', 5),
        (r'\b(B\.?Tech\.?|Bachelor\s*(?:\'?s)?\s*(?:of\s+)?Technolog?y?)\b', 'B.Tech', 5),
        (r'\b(B\.?E\.?|Bachelor\s*(?:\'?s)?\s*(?:of\s+)?Engineering)\b', 'B.E.', 5),
        (r'\b(B\.?C\.?A\.?|Bachelor\s*(?:\'?s)?\s*(?:of\s+)?Computer\s+Applications?)\b', 'BCA', 5),
        (r'\b(B\.?B\.?A\.?|Bachelor\s*(?:\'?s)?\s*(?:of\s+)?Business\s+Administration)\b', 'BBA', 5),
        (r'\b(B\.?Com\.?|Bachelor\s*(?:\'?s)?\s*(?:of\s+)?Commerce)\b', 'B.Com', 5),
        (r'\b(B\.?Sc\.?|Bachelor\s*(?:\'?s)?\s*(?:of\s+)?Science)\b', 'B.Sc', 5),
        (r'\bBachelor(?:\'?s)?\s+(?:of\s+|in\s+)?(\w+(?:\s+\w+)*)', 'Bachelor', 5),
        # Associates (Level 4)
        (r'\b(A\.?S\.?|Associate\s*(?:\'?s)?\s*(?:of\s+)?Science)\b', 'A.S.', 4),
        (r'\b(A\.?A\.?|Associate\s*(?:\'?s)?\s*(?:of\s+)?Arts)\b', 'A.A.', 4),
        (r'\bAssociate(?:\'?s)?\s+(?:of\s+|in\s+)?(\w+(?:\s+\w+)*)', 'Associate', 4),
        # Diploma (Level 3)
        (r'\b(Diploma)\s+(?:in\s+)?(\w+(?:\s+\w+)*)', 'Diploma', 3),
        (r'\b(Post\s+Graduate\s+Diploma|PG\s*Diploma|PGDCA)\b', 'PG Diploma', 3),
        # Certificate (Level 2) - from colleges/institutions
        (r'\b([\w\s]+Certification)\b', 'Certification', 2),
        (r'\b(Certificate\s+in\s+[\w\s]+)\b', 'Certificate', 2),
        # High School (Level 1)
        (r'\b(High\s+School\s+Diploma|HSC|SSC|Intermediate|12th|10th)\b', 'High School', 1),
    ]
    
    # Degree level mapping for quick lookup
    DEGREE_LEVELS = {
        'PhD': 7, 'Doctorate': 7,
        'M.S.': 6, 'M.A.': 6, 'MBA': 6, 'M.Tech': 6, 'M.E.': 6, 'MCA': 6, 'Master': 6,
        'B.S.': 5, 'B.A.': 5, 'B.Tech': 5, 'B.E.': 5, 'BCA': 5, 'BBA': 5, 'B.Com': 5, 'B.Sc': 5, 'Bachelor': 5,
        'A.S.': 4, 'A.A.': 4, 'Associate': 4,
        'Diploma': 3, 'PG Diploma': 3,
        'Certificate': 2, 'Certification': 2,
        'High School': 1,
    }
    
    # Institution indicators
    INSTITUTION_INDICATORS = [
        r'\buniversity\b',
        r'\bcollege\b',
        r'\binstitute\b',
        r'\bschool\b',
        r'\bacademy\b',
        r'\bpolytechnic\b',
    ]
    
    # Year pattern
    YEAR_PATTERN = r'\b(19|20)\d{2}\b'
    
    # GPA/CGPA patterns
    GPA_PATTERNS = [
        r'\b(GPA|CGPA|Grade)\s*[:\-]?\s*(\d+\.?\d*)\s*(?:/\s*\d+\.?\d*)?\b',
        r'\b(\d+\.?\d*)\s*(?:/\s*\d+\.?\d*)?\s*(GPA|CGPA)\b',
        r'\b(\d{1,2}(?:\.\d+)?)\s*%\b',
    ]

    def __init__(self, resume_text: str = None, store_in_db: bool = False, candidate_id: Optional[int] = None):
        """
        Initialize the education extractor.
        
        Args:
            resume_text: The resume text to parse (optional, can be set later)
            store_in_db: If True, store extracted education in database
            candidate_id: Candidate ID for database storage (required if store_in_db=True)
        """
        self.resume_text = resume_text
        self.store_in_db = store_in_db
        self.candidate_id = candidate_id
        
        # Compile regex patterns for performance
        self.edu_header_pattern = re.compile(
            '|'.join(self.EDUCATION_HEADERS), 
            re.IGNORECASE | re.MULTILINE
        )
        self.section_end_pattern = re.compile(
            '|'.join(self.SECTION_ENDS), 
            re.IGNORECASE | re.MULTILINE
        )
        self.year_pattern = re.compile(self.YEAR_PATTERN)
        self.gpa_patterns = [re.compile(p, re.IGNORECASE) for p in self.GPA_PATTERNS]
        self.degree_patterns = [(re.compile(p, re.IGNORECASE), name, level) for p, name, level in self.DEGREE_PATTERNS]
        self.institution_pattern = re.compile(
            '|'.join(self.INSTITUTION_INDICATORS), 
            re.IGNORECASE
        )

    def find_education_section(self, text: str) -> Optional[str]:
        """Find and extract the education section from resume text."""
        if not text:
            return None
        
        # Find education section start - look for standalone header
        # Pattern for section header (usually on its own line or followed by newline)
        # Also handles markdown headers (## Education)
        header_patterns = [
            r'(?:^|\n)\s*(?:#{1,3}\s*)?(EDUCATION|Education)\s*(?:\n|:|\s*$)',
            r'(?:^|\n)\s*(?:#{1,3}\s*)?(EDUCATIONAL\s+(?:BACKGROUND|DETAILS|QUALIFICATIONS?))\s*(?:\n|:|\s*$)',
            r'(?:^|\n)\s*(?:#{1,3}\s*)?(ACADEMIC\s+(?:BACKGROUND|CREDENTIALS|QUALIFICATIONS?))\s*(?:\n|:|\s*$)',
            r'(?:^|\n)\s*(?:#{1,3}\s*)?(EDUCATION\s*(?:&|AND)\s*(?:TRAINING|CERTIFICATIONS?))\s*(?:\n|:|\s*$)',
        ]
        
        match = None
        for pattern in header_patterns:
            m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if m:
                match = m
                break
        
        if not match:
            # Fallback to simple search
            match = self.edu_header_pattern.search(text)
            if not match:
                return None
        
        start_pos = match.start()
        
        # Find the next section header after education
        # Look for uppercase headers or common section patterns
        remaining_text = text[match.end():]
        
        # Patterns for next section (must be on new line, typically uppercase or title case)
        # Also handles markdown headers (## Section)
        next_section_patterns = [
            r'\n\s*(?:#{1,3}\s*)?(EXPERIENCE|PROFESSIONAL\s+EXPERIENCE|WORK\s+EXPERIENCE|WORK\s+HISTORY|EMPLOYMENT)',
            r'\n\s*(?:#{1,3}\s*)?(SKILLS|TECHNICAL\s+SKILLS|CORE\s+COMPETENCIES)',
            r'\n\s*(?:#{1,3}\s*)?(PROJECTS?|CERTIFICATIONS?|CERTIFICATES?)',
            r'\n\s*(?:#{1,3}\s*)?(PROFESSIONAL\s+SUMMARY|SUMMARY|OBJECTIVE|CAREER\s+OBJECTIVES?)',
            r'\n\s*(?:#{1,3}\s*)?(ACHIEVEMENTS?|AWARDS?|PUBLICATIONS?)',
            r'\n\s*(?:#{1,3}\s*)?(PERSONAL\s+(?:DETAILS|INFORMATION)|DECLARATION)',
            r'\n\s*(?:#{1,3}\s*)?(REFERENCES?|VOLUNTEER|ACTIVITIES|INTERESTS?|LANGUAGES?)',
            r'\n\s*(?:#{1,3}\s*)?(AFFILIATIONS?|WORK\s+PREFERENCES|PROFILE\s+SOURCES|CONTACT\s+INFORMATION)',
            r'\n[A-Z][A-Z\s]{3,}(?:\n|:)',  # Any uppercase header
        ]
        
        end_pos = None
        for pattern in next_section_patterns:
            end_match = re.search(pattern, remaining_text, re.IGNORECASE)
            if end_match:
                if end_pos is None or end_match.start() < end_pos:
                    end_pos = end_match.start()
        
        if end_pos is not None:
            end_pos = match.end() + end_pos
        else:
            # If no next section found, take next 1500 characters
            end_pos = min(start_pos + 1500, len(text))
        
        education_section = text[start_pos:end_pos].strip()
        
        # Ensure we got meaningful content
        if len(education_section) < 20:
            return None
            
        return education_section

    def extract_degrees(self, text: str) -> List[Tuple[str, int, str]]:
        """Extract degree names from text with their levels and original text."""
        degrees = []
        
        # Comprehensive degree patterns - covers US, UK, Indian formats
        # Level: PhD=7, Master=6, Bachelor=5, Associate=4, Diploma=3, Certificate=2, HighSchool=1
        full_degree_patterns = [
            # ==================== PhD / Doctorate (Level 7) ====================
            (r'\b(Ph\.?D\.?(?:\s+(?:in|of)\s+[A-Za-z][A-Za-z\s&/\-\.]+)?)', 7),
            (r'\b(Doctorate\s+(?:of|in)\s+[A-Za-z][A-Za-z\s&/\-\.]+)', 7),
            (r'\b(Doctor\s+of\s+(?:Philosophy|Science|Engineering)[A-Za-z\s&/\-\.]*)', 7),
            
            # ==================== Master's Degrees (Level 6) ====================
            # Full form: Master's of/in X
            (r"\b(Master(?:'?s)?\s+(?:of|in)\s+[A-Za-z][A-Za-z\s&/\-\.]+)", 6),
            (r'\b(Masters?\s+(?:of|in|from)\s+[A-Za-z][A-Za-z\s&/\-\.]+)', 6),
            (r"\b(Master(?:'?s)?\s+Degree(?:\s+in)?[A-Za-z\s&/\-\.,]*)", 6),
            # Abbreviations: M.S., M.A., MBA, MCA, M.Tech, M.E., M.Sc, M.Com
            (r'\b(M\.?S\.?(?:c\.?)?(?:\s+(?:in|from)\s+[A-Za-z][A-Za-z\s&/\-\.]+)?)', 6),
            (r'\b(M\.?A\.?(?:\s+(?:in|from)\s+[A-Za-z][A-Za-z\s&/\-\.]+)?)', 6),
            (r'\b(M\.?B\.?A\.?|MBA)(?:\s+(?:in|from)\s+[A-Za-z][A-Za-z\s&/\-\.]+)?', 6),
            (r'\b(M\.?C\.?A\.?|MCA)(?:\s+(?:in|from)\s+[A-Za-z][A-Za-z\s&/\-\.]+)?', 6),
            (r'\b(M\.?\s*Tech\.?(?:nology)?(?:[\s,]+(?:in\s+)?[A-Za-z][A-Za-z\s&/\-\.]+)?)', 6),
            (r'\b(M\.?\s*E\.?(?:ng)?(?:[\s,]+(?:in\s+)?[A-Za-z][A-Za-z\s&/\-\.]+)?)', 6),
            (r'\b(M\.?Sc\.?(?:\s+(?:in|from)\s+[A-Za-z][A-Za-z\s&/\-\.]+)?)', 6),
            (r'\b(M\.?Com\.?(?:\s+(?:in|from)\s+[A-Za-z][A-Za-z\s&/\-\.]+)?)', 6),
            
            # ==================== Bachelor's Degrees (Level 5) ====================
            # Full form: Bachelor's of/in X
            (r"\b(Bachelor(?:'?s)?\s+(?:of|in)\s+[A-Za-z][A-Za-z\s&/\-\.]+)", 5),
            (r'\b(Bachelors?\s+(?:of|in)\s+[A-Za-z][A-Za-z\s&/\-\.]+)', 5),
            (r"\b(Bachelor(?:'?s)?\s+Degree(?:\s+in)?[A-Za-z\s&/\-\.,]*)", 5),
            # B.Tech / B. Tech / BTech - with specialization (comma, in, from)
            (r'\b(B\.?\s*[-]?\s*Tech\.?(?:nology)?(?:[\s,]+(?:in\s+)?[A-Za-z][A-Za-z\s&/\-\.]+)?)', 5),
            (r'\b(B\.?\s*[-]?\s*E\.?(?:ng)?(?:[\s,]+(?:in\s+)?[A-Za-z][A-Za-z\s&/\-\.]+)?)', 5),
            # B.S., B.A., BCA, BBA, B.Com, B.Sc
            (r'\b(B\.?S\.?(?:c\.?)?(?:\s+(?:in|from)\s+[A-Za-z][A-Za-z\s&/\-\.]+)?)', 5),
            (r'\b(B\.?A\.?(?:\s+(?:in|from)\s+[A-Za-z][A-Za-z\s&/\-\.]+)?)', 5),
            (r'\b(B\.?C\.?A\.?|BCA)(?:\s+(?:in|from)\s+[A-Za-z][A-Za-z\s&/\-\.]+)?', 5),
            (r'\b(B\.?B\.?A\.?|BBA)(?:\s+(?:in|from)\s+[A-Za-z][A-Za-z\s&/\-\.]+)?', 5),
            (r'\b(B\.?Com\.?)(?:\s+(?:in|from)\s+[A-Za-z][A-Za-z\s&/\-\.]+)?', 5),
            (r'\b(B\.?Sc\.?)(?:\s+(?:in|from)\s+[A-Za-z][A-Za-z\s&/\-\.]+)?', 5),
            # BSC/BSc with specialization in parentheses - e.g., "BSC(MPCS)", "B.Sc(Computer Science)"
            (r'\b(B\.?S\.?[Cc]\.?\s*\([A-Za-z\s,&]+\)(?:\s+from\s+[A-Z][A-Za-z\s]+)?)', 5),
            (r'\b(BSC\s*\([A-Za-z\s,&]+\)(?:\s+from\s+[A-Z][A-Za-z\s]+)?)', 5),
            # General Studies
            (r'\b(General\s+Studies(?:\s+Degree)?)', 5),
            
            # ==================== Associate's Degrees (Level 4) ====================
            (r"\b((?:Pursuing\s+)?Associate(?:['\u2019]?s)?\s+(?:of|in|Degree\s+in)\s+[A-Za-z][A-Za-z\s&/\-\.]+)", 4),
            (r"\b((?:Pursuing\s+)?Associate(?:['\u2019]?s)?\s+Degree[A-Za-z\s&/\-\.,]*)", 4),
            (r'\b((?:Pursuing\s+)?Associates?\s+(?:of|in|Degree)\s+[A-Za-z][A-Za-z\s&/\-\.]+)', 4),
            (r'\b(A\.?A\.?S\.?(?:\s+(?:in)\s+[A-Za-z][A-Za-z\s&/\-\.]+)?)', 4),
            (r'\b(A\.?S\.?(?:\s+(?:in)\s+[A-Za-z][A-Za-z\s&/\-\.]+)?)', 4),
            (r'\b(A\.?A\.?(?:\s+(?:in)\s+[A-Za-z][A-Za-z\s&/\-\.]+)?)', 4),
            
            # ==================== Diploma (Level 3) ====================
            (r'\b((?:Post[-\s]?Graduate\s+)?Diploma\s+(?:in\s+)?[A-Za-z][A-Za-z\s&/\-\.]+)', 3),
            (r'\b(PG[-\s]?Diploma\s+(?:in\s+)?[A-Za-z][A-Za-z\s&/\-\.]+)', 3),
            (r'\b(PGDCA|P\.G\.D\.C\.A\.?)', 3),
            (r'\b(Polytechnic(?:\s+(?:in\s+)?[A-Za-z][A-Za-z\s&/\-\.]+)?)', 3),
            
            # ==================== Certificate (Level 2) ====================
            # ONLY educational certificates, not professional certifications
            (r'\b((?:Computer|Information)\s+(?:Systems?|Technology)\s+Certification)', 2),
            (r'\b(Certificate\s+(?:in\s+)?(?:Computer|Information|Technology)[A-Za-z\s&/\-\.]+)', 2),
            # ITIL and similar
            (r'\b(ITIL(?:\s*V?[0-9]+)?(?:\s+Foundation|\s+Practitioner)?)', 2),
            
            # ==================== High School (Level 1) ====================
            (r'\b(High\s+School\s+Diploma)', 1),
            (r'\b(Intermediate(?:\s+(?:in\s+)?[A-Za-z][A-Za-z\s&/\-\.]+)?)', 1),
            (r'\b(HSC|SSC|SSLC)\b', 1),
            (r'\b(12th|10th)\s*(?:Standard|Class|Grade)?', 1),
            (r'\b(Senior\s+Secondary|Higher\s+Secondary)', 1),
            (r'\b(CBSE|ICSE|State\s+Board)\b', 1),
        ]
        
        for pattern, level in full_degree_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match and len(match.strip()) > 2:
                    # Clean the extracted degree text
                    clean_degree = match.strip()
                    clean_degree = re.sub(r'\s+', ' ', clean_degree)  # normalize spaces
                    
                    # Truncate at common stop points
                    # BUT keep "from" if it's part of the degree pattern (e.g., "B. Tech from JNTU")
                    for stop in [' at ', ' - ', '\n', '|', '–', '—', ',  ', '  ']:
                        if stop in clean_degree:
                            clean_degree = clean_degree.split(stop)[0]
                    
                    # Only truncate at "from" if it's followed by common non-university words
                    # Keep "from University/College/Institute" patterns
                    if ' from ' in clean_degree.lower():
                        parts = clean_degree.split(' from ', 1)
                        if len(parts) == 2:
                            after_from = parts[1].lower()
                            # Keep if followed by university/college/institute names
                            if not any(inst in after_from for inst in ['university', 'college', 'institute', 'school', 'academy', 'technological', 'jntu', 'iit', 'nit']):
                                clean_degree = parts[0]
                    
                    clean_degree = clean_degree.rstrip('|–-,. ')
                    
                    # Skip if too short
                    if len(clean_degree) < 3:
                        continue
                    
                    # Limit length to avoid capturing too much text
                    if len(clean_degree) > 80:
                        clean_degree = clean_degree[:80].rsplit(' ', 1)[0]
                    
                    # Skip obvious false positives
                    lower = clean_degree.lower()
                    
                    # Skip if starts with "as " or "be " (common false positives)
                    if lower.startswith('as ') or lower.startswith('be '):
                        # But allow legitimate degrees like "Associate" or "Bachelor"
                        if not any(deg in lower for deg in ['associate', 'bachelor', 'master', 'b.s', 'b.a', 'm.s', 'm.a']):
                            continue
                    
                    false_positives = [
                        'systems and', 'teams and', 'programs and', 'items', 'problems', 
                        'forms', 'terms', 'platforms', 'ms office', 'ms word', 'ms excel',
                        'ms project', 'ms teams', 'ms dynamics', 'ms access', 'ms visio',
                        'ms sql', 'ms azure', 'ms build', 'ms partner', 'ms ssis', 'ms vb',
                        'ms. word', 'ms. excel', 'ms. powerpoint',
                        'me to ', 'me learn', 'me know', 'me scope',
                        'ba team', 'ba and scrum', 'ba developer', 'ma annual', 'ma 04', 
                        'annual sales', 'scrum team', 'impediments', 'wordpress', 'html,',
                        'national retail', 'blue print', 'contract',
                        'qualifications profile', 'qualification profile',
                        'international information systems security certification consortium',
                    ]
                    if any(fp in lower for fp in false_positives):
                        continue
                    
                    # Must contain education-related words for short matches
                    edu_words = ['science', 'arts', 'engineering', 'technology', 'technological', 'business', 
                                 'computer', 'management', 'administration', 'commerce', 'education',
                                 'psychology', 'nursing', 'economics', 'finance', 'accounting',
                                 'mathematics', 'physics', 'chemistry', 'biology', 'history',
                                 'literature', 'communication', 'information', 'electrical',
                                 'mechanical', 'civil', 'degree', 'diploma', 'mba', 'bca', 'mca',
                                 'b.tech', 'm.tech', 'b.e.', 'm.e.', 'b.sc', 'm.sc', 'b.com',
                                 'health', 'human', 'performance', 'international', 'pursuing',
                                 'university', 'college', 'institute', 'jawaharlal', 'nehru', 'jntu',
                                 'svu', 'mpcs', 'mpc', 'bipc', 'cec', 'hec', 'intermediate', 'board',
                                 'hsc', 'ssc', 'sslc', 'cbse', 'icse', '12th', '10th', 'secondary', 'standard']
                    
                    # For short matches or M.A./M.S./B.A./B.S. style, require edu words
                    if len(clean_degree) < 25 or lower.startswith(('m.a', 'ma ', 'm.s', 'ms ', 'b.a', 'ba ', 'b.s', 'bs ', 'aas')):
                        if not any(ew in lower for ew in edu_words):
                            # Check if it's a clear degree pattern or school board
                            if not re.match(r'^(master|bachelor|associate|diploma|phd|doctorate|pursuing|hsc|ssc|sslc|cbse|icse|12th|10th|intermediate)', lower):
                                continue
                    
                    degrees.append((clean_degree, level, match))
        
        # Remove duplicates, keeping highest level
        seen = {}
        for name, level, original in degrees:
            key = name[:30].lower()
            if key not in seen or level > seen[key][1]:
                seen[key] = (name, level, original)
        
        return [(name, level, orig) for name, level, orig in seen.values()]
    
    def get_highest_degree(self, degrees: List[Tuple[str, int, str]]) -> Tuple[Optional[str], int]:
        """Get the highest degree from a list of (degree_name, level, original) tuples."""
        if not degrees:
            return None, 0
        highest = max(degrees, key=lambda x: x[1])
        return highest[0], highest[1]

    def extract_institutions(self, text: str) -> List[str]:
        """Extract institution names from text."""
        institutions = []
        lines = text.split('\n')
        
        for line in lines:
            if self.institution_pattern.search(line):
                # Clean up the line
                clean_line = re.sub(r'[•\-–—|]', '', line).strip()
                if clean_line and len(clean_line) > 5:
                    institutions.append(clean_line[:100])  # Limit length
        
        return institutions[:5]  # Return max 5 institutions

    def extract_years(self, text: str) -> List[str]:
        """Extract years from text."""
        # Find all 4-digit years
        full_years = re.findall(r'\b((?:19|20)\d{2})\b', text)
        valid_years = [y for y in full_years if 1950 <= int(y) <= 2030]
        return list(set(valid_years))

    def extract_gpa(self, text: str) -> Optional[str]:
        """Extract GPA/CGPA from text."""
        for pattern in self.gpa_patterns:
            match = pattern.search(text)
            if match:
                return match.group(0)
        return None

    def get_highest_education_string(self, text: str) -> Optional[str]:
        """Extract the highest education as a simple string."""
        degrees = self.extract_degrees(text)
        if not degrees:
            return None
        
        # Get highest degree
        highest = max(degrees, key=lambda x: x[1])
        return highest[0]  # Return just the degree string

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

    def extract(self) -> List[str]:
        """
        Main extraction method - extracts education from resume text.
        
        Returns:
            List of extracted education strings (highest first)
        """
        if not self.resume_text:
            logger.warning("No resume text provided")
            if self.store_in_db:
                self._store_in_database('Unknown', '')
            return []
        
        # Find education section first
        education_section = self.find_education_section(self.resume_text)
        
        # Try to extract from education section first
        education = None
        if education_section:
            education = self.get_highest_education_string(education_section)
        
        # Fallback to full text if no degree found in education section
        if not education:
            education = self.get_highest_education_string(self.resume_text)
        
        result = [education] if education else []
        
        # Store in database if requested
        if self.store_in_db:
            highest = result[0] if result else 'Unknown'
            self._store_in_database(highest, '\n'.join(result))
        
        return result

    def get_highest_education(self) -> Optional[str]:
        """
        Extract and return only the highest education degree.
        
        Returns:
            Highest education string, or None if not found
        """
        result = self.extract()
        return result[0] if result else None


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
    B.Tech in Computer Science
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
    
    # Get highest education only
    highest = extractor.get_highest_education()
    print(f"\nHighest education: {highest}")
