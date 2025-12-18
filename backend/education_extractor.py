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
        r'\beducation\s*(&|and)\s*certifications\b',  # cover "Education & Certifications"
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
            # M.A. / MA directly followed by subject: e.g., "MA English Education"
            (r'\b(M\.?A\.?\s+[A-Za-z][A-Za-z\s&/\-\.]+)', 6),
            (r'\b(M\.?A\.?(?:\s+(?:in|from)\s+[A-Za-z][A-Za-z\s&/\-\.]+)?)', 6),
            (r'\b(M\.?B\.?A\.?|MBA)(?:\s+(?:in|from)\s+[A-Za-z][A-Za-z\s&/\-\.]+)?', 6),
            (r'\b(M\.?C\.?A\.?|MCA)(?:\s+(?:in|from)\s+[A-Za-z][A-Za-z\s&/\-\.]+)?', 6),
            (r'\b(M\.?\s*Tech\.?(?:nology)?(?:[\s,]+(?:in\s+)?[A-Za-z][A-Za-z\s&/\-\.]+)?)', 6),
            (r'\b(M\.?\s*E\.?(?:ng)?(?:[\s,]+(?:in\s+)?[A-Za-z][A-Za-z\s&/\-\.]+)?)', 6),
            (r'\b(M\.?Sc\.?(?:\s+(?:in|from)\s+[A-Za-z][A-Za-z\s&/\-\.]+)?)', 6),
            (r'\b(M\.?Com\.?(?:\s+(?:in|from)\s+[A-Za-z][A-Za-z\s&/\-\.]+)?)', 6),
            
            # ==================== Bachelor's Degrees (Level 5) ====================
            # CRITICAL: Order matters! More specific patterns MUST come before generic ones
            
            # Pattern 1: "Bachelor's Degree in X" - MOST SPECIFIC, MUST BE FIRST
            (r"\b(Bachelor(?:'?s)?\s+Degree\s+in\s+[A-Za-z][A-Za-z\s&/\-\.]+)", 5),
            # Pattern 2: "Bachelor's Degree" (without "in") - still specific
            (r"\b(Bachelor(?:'?s)?\s+Degree(?:\s+[A-Za-z][A-Za-z\s&/\-\.,]+)?)", 5),
            # Pattern 3: "Bachelor's of X" or "Bachelor's in X" - specific with specialization
            (r"\b(Bachelor(?:'?s)?\s+(?:of|in)\s+[A-Za-z][A-Za-z\s&/\-\.]+)", 5),
            # Pattern 4: "Bachelors of X" or "Bachelors in X" (without apostrophe)
            (r'\b(Bachelors?\s+(?:of|in)\s+[A-Za-z][A-Za-z\s&/\-\.]+)', 5),
            # Pattern 5: B.A. / BA directly followed by subject: e.g., "BA English Literature"
            (r'\b(B\.?A\.?\s+[A-Za-z][A-Za-z\s&/\-\.]+)', 5),
            # Pattern 6: Specific pattern for "Bachelor's in Computer Science"
            (r"\b(Bachelor'?s?\s+in\s+Computer\s+Science)\b", 5),
            # Pattern 7: "Computer Science, Bachelors (BSCS)" format
            (r'\b(Computer\s+Science,?\s+Bachelors?\s*\([A-Z]+\))', 5),
            # Pattern 8: "Field, Bachelors (Abbreviation)" format - more general
            (r'\b([A-Za-z][A-Za-z\s&/\-\.]+,?\s+Bachelors?\s*\([A-Z]+\))', 5),
            # Pattern 9: Standalone "Bachelors" with parentheses abbreviation
            (r'\b(Bachelors?\s*\([A-Z]+\))', 5),
            # Pattern 10: CRITICAL: Only match standalone "Bachelors" if NOT part of a longer degree phrase
            # This MUST be LAST to avoid matching when a more specific pattern exists
            # Use negative lookahead to ensure it's not followed by "'s", " Degree", " of", or " in"
            (r'\b(Bachelors?)(?!\s*[\'s]|\s+Degree|\s+of|\s+in)\b', 5),  # Standalone "Bachelors" or "Bachelor" (not part of longer phrase)
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
            # BSC/BSc followed directly by specialization (without "in" or "from") - e.g., "BSC computers", "BSC Computer Science"
            # This pattern matches "BSC" followed by one or more words (specialization)
            (r'\b((?:B\.?S\.?C\.?|BSC|BSc)\s+[A-Za-z][A-Za-z\s&/\-\.]+)', 5),
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
            # Intermediate pattern - stop before "Skills" keyword
            (r'\b(Intermediate(?:\s+(?:in\s+)?[A-Za-z][A-Za-z\s&/\-\.]*?)(?=\s+Skills|\s+skill|\s+SKILLS|$))', 1),
            (r'\b(Intermediate(?:\s+(?:in\s+)?[A-Za-z]{1,10})(?=\s+[A-Z][a-z]+\s+Skills|$))', 1),
            (r'\b(HSC|SSC|SSLC)\b', 1),
            (r'\b(12th|10th)\s*(?:Standard|Class|Grade)?', 1),
            (r'\b(Senior\s+Secondary|Higher\s+Secondary)', 1),
            (r'\b(CBSE|ICSE|State\s+Board)\b', 1),
            
            # ==================== Additional Bachelor's Patterns ====================
            # BTech variations
            (r'\b(BTech|B\.?\s*Tech|B[-]?Tech)\b', 5),
            (r'\b(Bachelor\s+of\s+Technology(?:\s+in\s+[A-Za-z][A-Za-z\s&/\-\.]+)?)', 5),
            (r'\b(Bachelor\s+of\s+Technology\s+in\s+(?:IT|CSE|ECE|Mechanical\s+Engineering|Civil\s+Engineering|Electrical\s+Engineering))', 5),
            # BE variations
            (r'\b(B\.?\s*E\.?|B\s+E|BE|BEng)\b', 5),
            (r'\b(Bachelor\s+of\s+Engineering(?:\s+in\s+[A-Za-z][A-Za-z\s&/\-\.]+)?)', 5),
            (r'\b(Bachelor\s+of\s+Engineering\s+in\s+(?:Computer\s+Science|Electronics|Mechanical\s+Engineering|Electrical\s+Engineering|Civil\s+Engineering))', 5),
            # BS/BSc variations
            (r'\b(B\.?\s*S\.?|B\s+S|BS|BSc|B\.?\s*Sc\.?)\b', 5),
            (r'\b(Bachelor\s+of\s+Science(?:\s+\(BS\))?(?:\s+in\s+[A-Za-z][A-Za-z\s&/\-\.]+)?)', 5),
            (r'\b(Bachelor\s+of\s+Science\s+in\s+(?:Computer\s+Science|Information\s+Technology|Business|Physics|Chemistry|Mathematics))', 5),
            # BA variations
            (r'\b(B\.?\s*A\.?|B\s+A|BA)\b', 5),
            (r'\b(Bachelor\s+of\s+Arts(?:\s+in\s+[A-Za-z][A-Za-z\s&/\-\.]+)?)', 5),
            (r'\b(Bachelor\s+of\s+Arts\s+in\s+(?:English|Economics|Journalism|Sociology))', 5),
            # BCom variations
            (r'\b(B\.?\s*Com\.?|BCom|Bachelors?\s+of\s+Commerce)\b', 5),
            (r'\b(Bachelor\s+of\s+Commerce(?:\s+in\s+(?:Accounting|Finance))?)', 5),
            # BCA, BBA
            (r'\b(BCA|Bachelor\s+of\s+Computer\s+Applications?)\b', 5),
            (r'\b(BBA|Bachelor\s+of\s+Business\s+Administration|Bachelors?\s+in\s+Business\s+Administration)\b', 5),
            # Additional Bachelor's degrees
            (r'\b(Bachelor\s+of\s+Social\s+Work|Bachelor\s+of\s+Fine\s+Arts|Bachelor\s+of\s+Education|Bachelor\s+of\s+Information\s+Systems|Bachelor\s+of\s+Public\s+Administration|Bachelor\s+of\s+Health\s+Sciences|Bachelor\s+of\s+Human\s+Resources|Bachelor\s+of\s+International\s+Business)\b', 5),
            # Engineering specializations
            (r'\b(Bachelor\s+of\s+(?:Mechanical|Civil|Electrical|Electronics|Software|Chemical|Industrial|Petroleum|Aeronautical|Aerospace|Agricultural|Biomedical|Genetic|Information\s+Technology|Information\s+Systems|Information\s+Science|Applied\s+Sciences|Environmental\s+Science|Forestry|Marine|Mining|Metallurgical|Textile)\s+Engineering)\b', 5),
            (r'\b(Bachelor\s+of\s+(?:Architecture|Planning|Design|Fine\s+Arts|Visual\s+Communication|Mass\s+Communication|Journalism|Social\s+Work|Education|Special\s+Education|Early\s+Childhood\s+Education|Tourism|Hotel\s+Management|Hospitality\s+Management|Travel\s+&\s+Tourism\s+Management|Business\s+Management|Business\s+Studies|Strategic\s+Management|Public\s+Relations|Communications\s+Engineering|Biochemistry|Biotechnology|Microbiology|Molecular\s+Biology|Physics|Chemistry|Mathematics|Statistics|Artificial\s+Intelligence|Machine\s+Learning|Data\s+Science|Robotics|Cyber\s+Security|Cloud\s+Computing|Embedded\s+Systems|VLSI\s+Design|Network\s+Engineering|Software\s+Systems|Health\s+Sciences|Nursing|Pharmacy|Physiotherapy|Occupational\s+Therapy|Nutrition|Dietetics|Medical\s+Laboratory\s+Technology|Radiology|Ophthalmology|Dental\s+Surgery|Public\s+Health|Agriculture|Horticulture|Animal\s+Science|Fisheries|Veterinary\s+Science|Accounting|Finance|Actuarial\s+Science|Criminology|Criminal\s+Justice|Legal\s+Studies|Forensic\s+Science|Homeland\s+Security|Political\s+Science|Public\s+Policy|International\s+Relations|Anthropology|Sociology|Psychology|Philosophy|History|Geography|Humanities|Liberal\s+Arts|Theology))\b', 5),
            # Additional specialized Bachelor's - Engineering
            (r'\b(Bachelor\s+of\s+(?:Renewable\s+Energy|Automation|Instrumentation|Mechatronics|Power\s+Systems|Communication\s+Systems|Nano\s+Technology|Optoelectronics|Maritime|Structural|Transportation|Water\s+Resources|Environmental|Mining\s+Safety)\s+Engineering)\b', 5),
            # Additional specialized Bachelor's - Non-Engineering
            (r'\b(Bachelor\s+of\s+(?:Military\s+Science|Defense\s+Studies|Strategic\s+Studies|Intelligence\s+Studies|Global\s+Studies|International\s+Business|Supply\s+Chain|Logistics|Operations\s+Research|Big\s+Data\s+Analytics|FinTech|Blockchain|UX\s+Design|Product\s+Design|Game\s+Design|Animation|Film\s+Studies|Media\s+Studies|Fashion\s+Technology|Apparel\s+Design|Textile\s+Engineering|Ceramic\s+Technology|Polymer\s+Science|Leather\s+Technology|Marine\s+Biology|Oceanography|Fisheries\s+Science|Food\s+Technology|Dairy\s+Technology|Rural\s+Development|Social\s+Sciences|Gender\s+Studies|Peace\s+Studies|Disaster\s+Management|Human\s+Resource\s+Management|Organizational\s+Behavior|Entrepreneurship|Marketing\s+Management|Finance\s+Management|Banking\s+and\s+Insurance|Investment\s+Management|Risk\s+Management|Auditing|Taxation|International\s+Economics|Business\s+Economics|Development\s+Studies|Public\s+Health\s+Administration|Clinical\s+Psychology|Cognitive\s+Science|Neuroscience|Biomedical\s+Science|Pharmacy\s+Practice|Medical\s+Technology|Nutrition\s+Science|Paramedical\s+Science|Optometry|Prosthetics|Audiology|Speech\s+Pathology|Clinical\s+Research|Immunology|Virology|Genetics|Marine\s+Transportation|Nautical\s+Science|Naval\s+Architecture|Aviation|Pilot\s+Studies|Air\s+Traffic\s+Management|Aeronautics|Space\s+Science|Astrophysics|Astronomy|Surveying|Urban\s+Planning|Geoinformatics|Geographic\s+Information\s+Systems|Cartography|Construction\s+Management|Ecology|Climate\s+Science|Earth\s+Science|Geology|Geophysics|Petroleum\s+Geoscience|Commerce\s+Honors|Accounting|Finance|Actuarial\s+Science|Criminology|Criminal\s+Justice|Legal\s+Studies|Forensic\s+Science|Homeland\s+Security|Political\s+Science|Public\s+Policy|International\s+Relations|Anthropology|Sociology|Psychology|Philosophy|History|Geography|Humanities|Liberal\s+Arts|Theology))\b', 5),
            
            # ==================== Additional Master's Patterns ====================
            # MTech variations
            (r'\b(MTech|M\.?\s*Tech|M[-]?Tech|Masters?\s+of\s+Technology)\b', 6),
            (r'\b(Master\s+of\s+Technology(?:\s+in\s+[A-Za-z][A-Za-z\s&/\-\.]+)?)', 6),
            (r'\b(Master\s+of\s+Technology\s+in\s+(?:CSE|ECE))', 6),
            # ME variations
            (r'\b(M\.?\s*E\.?|M\s+E|ME|Master\s+of\s+Engineering)\b', 6),
            (r'\b(Master\s+of\s+Engineering(?:\s+in\s+[A-Za-z][A-Za-z\s&/\-\.]+)?)', 6),
            # MS variations
            (r'\b(M\.?\s*S\.?|M\s+S|MS|MSc|M\.?\s*Sc\.?)\b', 6),
            (r'\b(Master\s+of\s+Science(?:\s+in\s+[A-Za-z][A-Za-z\s&/\-\.]+)?)', 6),
            (r'\b(MS\s+in\s+(?:Computer\s+Science|Information\s+Technology|Business\s+Analytics|Data\s+Science|Marketing|Finance|HR|Electrical\s+Engineering|Mechanical\s+Engineering|Civil\s+Engineering|Computer\s+Engineering|Software\s+Engineering|Data\s+Engineering|Artificial\s+Intelligence|Electrical\s+Systems|Computer\s+Networks|Information\s+Security|Enterprise\s+Computing|Business\s+Information\s+Systems|Digital\s+Forensics|Robotics\s+Engineering|Civil\s+Structures|Mechanical\s+Design|Thermal\s+Engineering|Chemical\s+Process\s+Engineering|Industrial\s+Automation|Material\s+Science|Metallurgy|Microelectronics|Artificial\s+Intelligence\s+Systems|Cloud\s+Infrastructure|Software\s+Systems|Data\s+Analytics\s+Engineering|Systems\s+Engineering|Applied\s+Mathematics|Applied\s+Physics|Applied\s+Chemistry|Electrical\s+Power|Signal\s+Processing|Wireless\s+Communication|Biomedical\s+Engineering|Transportation\s+Systems|Construction\s+Engineering|Renewable\s+Energy\s+Systems|Aerospace\s+Structures|Athletic\s+Training|Nutrition|Biostatistics|Agricultural\s+Sciences|Applied\s+Economics|Computational\s+Chemistry|Environmental\s+Policy|Climate\s+Modeling|Mathematical\s+Finance|Quantitative\s+Economics|Applied\s+Statistics|Advanced\s+Manufacturing|Industrial\s+Automation|Chemical\s+Processes|Geological\s+Engineering|Mining\s+Engineering|Metallurgical\s+Engineering|Polymer\s+Engineering|Textile\s+Engineering|Material\s+Science\s+Engineering|Bioengineering|Behavioral\s+Science|Development\s+Economics|Rural\s+Development|Peace\s+Studies|Human\s+Rights|Public\s+Relations|Organizational\s+Psychology|UX\s+Design|Artificial\s+Intelligence\s+and\s+Robotics|Big\s+Data\s+Systems|Digital\s+Marketing\s+Analytics))', 6),
            # MA variations
            (r'\b(M\.?\s*A\.?|M\s+A|MA|Master\s+of\s+Arts)\b', 6),
            (r'\b(Master\s+of\s+Arts\s+in\s+(?:English|Sociology|Journalism))', 6),
            # MCom variations
            (r'\b(M\.?\s*Com\.?|MCom|Master\s+of\s+Commerce)\b', 6),
            # MBA variations
            (r'\b(MBA|Master\s+of\s+Business\s+Administration|Masters?\s+in\s+Business\s+Administration)\b', 6),
            (r'\b(MBA\s+in\s+(?:Marketing|Finance|HR|IT|Analytics|Operations|Project\s+Management))', 6),
            # MCA
            (r'\b(MCA|Master\s+of\s+Computer\s+Applications?)\b', 6),
            # Additional Master's degrees - Engineering
            (r'\b(Master\s+of\s+(?:Mechanical|Civil|Electrical|Electronics|Software|Chemical|Industrial|Petroleum|Aeronautical|Aerospace|Agricultural|Biomedical|Genetic|Marine|Mining|Metallurgical|Textile|Renewable\s+Energy|Automation|Instrumentation|Mechatronics|Power\s+Systems|Communication\s+Systems|Structural|Transportation|Water\s+Resources|Environmental)\s+Engineering)\b', 6),
            # Additional Master's degrees - Non-Engineering
            (r'\b(Master\s+of\s+(?:Information\s+Technology|Information\s+Systems|Applied\s+Sciences|Data\s+Science|Analytics|Artificial\s+Intelligence|Cyber\s+Security|Machine\s+Learning|Robotics|Cloud\s+Computing|Embedded\s+Systems|VLSI\s+Design|Network\s+Engineering|Project\s+Management|Operations\s+Management|Supply\s+Chain\s+Management|Logistics|Retail\s+Management|Tourism\s+Management|Hotel\s+Management|Architecture|Design|Fine\s+Arts|Social\s+Work|Criminology|Criminal\s+Justice|Legal\s+Studies|Forensic\s+Science|Homeland\s+Security|Political\s+Science|Public\s+Administration|Public\s+Policy|Anthropology|Sociology|Psychology|Philosophy|History|Geography|Humanities|Liberal\s+Arts|Theology|Health\s+Sciences|Nursing|Pharmacy|Physiotherapy|Molecular\s+Biology|Biotechnology|Biochemistry|Mathematics|Statistics|Physics|Chemistry|Environmental\s+Science|Data\s+Engineering|Bioinformatics|Medical\s+Science|Radiology|Ophthalmology|Occupational\s+Therapy|Public\s+Health|Nano\s+Technology|Optoelectronics|Marine\s+Biology|Oceanography|Fisheries\s+Science|Food\s+Technology|Dairy\s+Technology|Public\s+Health\s+Administration|Occupational\s+Health|Epidemiology|Clinical\s+Psychology|Cognitive\s+Science|Neuroscience|Biomedical\s+Science|Pharmacy\s+Practice|Clinical\s+Research|Nutrition\s+Science|Paramedical\s+Science|Optometry|Immunology|Virology|Genetics|Marine\s+Transportation|Nautical\s+Science|Naval\s+Architecture|Space\s+Science|Astrophysics|Astronomy|Aviation|Air\s+Traffic\s+Management|Aeronautics|Urban\s+Planning|Geoinformatics|GIS|Surveying|Construction\s+Management|Ecology|Climate\s+Science|Earth\s+Science|Geology|Geophysics|Petroleum\s+Geoscience))\b', 6),
            
            # ==================== Additional Associate's Patterns ====================
            (r'\b(Associate(?:\'?s)?\s+Degree|Associates?\s+Degree)\b', 4),
            (r'\b(Associate\s+of\s+Arts(?:\s+in\s+(?:Psychology|Sociology|Political\s+Science|Business\s+Analytics))?)\b', 4),
            (r'\b(Associate\s+of\s+Science(?:\s+in\s+(?:Information\s+Technology|Mechanical\s+Engineering|Civil\s+Engineering|Chemical\s+Engineering|Physics|Mathematics|Computer\s+Systems))?)\b', 4),
            (r'\b(Associate\s+of\s+Computer\s+Science|Associate\s+in\s+Information\s+Technology|Associate\s+of\s+Applied\s+Science(?:\s+in\s+Computer\s+Systems)?)\b', 4),
            (r'\b(Associate\s+of\s+(?:Marketing|Accounting|Finance|Management|Business\s+Studies))\b', 4),
            (r'\b(A\.?A\.?S\.?|A\.?S\.?|A\.?A\.?)\b', 4),

            
            # ==================== Additional Diploma Patterns ====================
            (r'\b(PG[-]?Diploma|Post\s+Graduate\s+Diploma|Postgraduate\s+Diploma|PGDM)\b', 3),
            (r'\b(Post\s+Graduate\s+Diploma\s+in\s+(?:Computer\s+Science|HR|Marketing|Finance|Management))', 3),
            (r'\b(Graduate\s+Diploma|Diploma\s+in\s+(?:Computer\s+Science|Electronics|Engineering|IT|Accounting|Business|HR|Marketing|Finance))\b', 3),
            
            # ==================== Additional Doctorate Patterns ====================
            (r'\b(PHD|PhD|Ph\.?D\.?|Doctorate|Doctor\s+of\s+Philosophy|Doctor\s+of\s+Engineering|Doctor\s+of\s+Science|Doctor\s+of\s+Business\s+Administration|DBA|Doctor\s+of\s+Management|Doctor\s+of\s+Education)\b', 7),
            (r'\b(Ph\.?\s*D\.?\s+(?:in|of)\s+[A-Za-z][A-Za-z\s&/\-\.]+)', 7),
            (r'\b(D\.?\s*Phil\.?|DPhil)\b', 7),
            (r'\b(D\.?\s*Sc\.?|DSc|Doctor\s+of\s+Science)\b', 7),
            (r'\b(D\.?\s*Eng\.?|DEng|Doctor\s+of\s+Engineering)\b', 7),
            (r'\b(D\.?\s*Ed\.?|DEd|Doctor\s+of\s+Education)\b', 7),
            (r'\b(D\.?\s*M\.?D\.?|DM|Doctor\s+of\s+Medicine)\b', 7),
            (r'\b(D\.?\s*V\.?M\.?|DVM|Doctor\s+of\s+Veterinary\s+Medicine)\b', 7),
            
            # ==================== Additional Master's Variations ====================
            # More MS/MSc variations
            (r'\b(M\.?\s*S\.?\s*[Cc]\.?\s+(?:in|of)\s+[A-Za-z][A-Za-z\s&/\-\.]+)', 6),
            (r'\b(M\.?\s*Sc\.?\s+(?:in|of)\s+[A-Za-z][A-Za-z\s&/\-\.]+)', 6),
            (r'\b(M\.?\s*Phil\.?|MPhil)\b', 6),
            (r'\b(M\.?\s*Res\.?|MRes|Master\s+of\s+Research)\b', 6),
            (r'\b(M\.?\s*Eng\.?|MEng|Master\s+of\s+Engineering)\b', 6),
            (r'\b(M\.?\s*Arch\.?|MArch|Master\s+of\s+Architecture)\b', 6),
            (r'\b(M\.?\s*Des\.?|MDes|Master\s+of\s+Design)\b', 6),
            (r'\b(M\.?\s*FA|MFA|Master\s+of\s+Fine\s+Arts)\b', 6),
            (r'\b(M\.?\s*Ed\.?|MEd|Master\s+of\s+Education)\b', 6),
            (r'\b(M\.?\s*PA|MPA|Master\s+of\s+Public\s+Administration)\b', 6),
            (r'\b(M\.?\s*SW|MSW|Master\s+of\s+Social\s+Work)\b', 6),
            (r'\b(M\.?\s*Lib\.?\s*St\.?|MLS|Master\s+of\s+Library\s+Science)\b', 6),
            (r'\b(M\.?\s*Div\.?|MDiv|Master\s+of\s+Divinity)\b', 6),
            (r'\b(M\.?\s*Th\.?|MTh|Master\s+of\s+Theology)\b', 6),
            (r'\b(M\.?\s*D\.?S\.?|MDS|Master\s+of\s+Dental\s+Surgery)\b', 6),
            (r'\b(M\.?\s*Ch\.?|MCh|Master\s+of\s+Surgery)\b', 6),
            (r'\b(M\.?\s*Pharm\.?|MPharm|Master\s+of\s+Pharmacy)\b', 6),
            (r'\b(M\.?\s*V\.?Sc\.?|MVSc|Master\s+of\s+Veterinary\s+Science)\b', 6),
            (r'\b(M\.?\s*F\.?A\.?|MFA|Master\s+of\s+Fine\s+Arts)\b', 6),
            (r'\b(M\.?\s*Mus\.?|MMus|Master\s+of\s+Music)\b', 6),
            (r'\b(M\.?\s*Plan\.?|MPlan|Master\s+of\s+Planning)\b', 6),
            (r'\b(M\.?\s*URP|MURP|Master\s+of\s+Urban\s+and\s+Regional\s+Planning)\b', 6),
            (r'\b(M\.?\s*LL\.?\s*M\.?|LLM|Master\s+of\s+Laws)\b', 6),
            (r'\b(M\.?\s*Jur\.?|MJur|Master\s+of\s+Jurisprudence)\b', 6),
            (r'\b(M\.?\s*Fin\.?|MFin|Master\s+of\s+Finance)\b', 6),
            (r'\b(M\.?\s*Acc\.?|MAcc|Master\s+of\s+Accountancy)\b', 6),
            (r'\b(M\.?\s*Tax\.?|MTax|Master\s+of\s+Taxation)\b', 6),
            (r'\b(M\.?\s*HRM|MHRM|Master\s+of\s+Human\s+Resource\s+Management)\b', 6),
            (r'\b(M\.?\s*IM|MIM|Master\s+in\s+Management)\b', 6),
            (r'\b(M\.?\s*EM|MEM|Master\s+of\s+Engineering\s+Management)\b', 6),
            (r'\b(M\.?\s*ISM|MISM|Master\s+of\s+Information\s+Systems\s+Management)\b', 6),
            (r'\b(M\.?\s*IT|MIT|Master\s+of\s+Information\s+Technology)\b', 6),
            (r'\b(M\.?\s*CS|MCS|Master\s+of\s+Computer\s+Science)\b', 6),
            (r'\b(M\.?\s*Comp\.?\s*Sci\.?|MCompSci|Master\s+of\s+Computer\s+Science)\b', 6),
            (r'\b(M\.?\s*IS|MIS|Master\s+of\s+Information\s+Systems)\b', 6),
            (r'\b(M\.?\s*DS|MDS|Master\s+of\s+Data\s+Science)\b', 6),
            (r'\b(M\.?\s*AI|MAI|Master\s+of\s+Artificial\s+Intelligence)\b', 6),
            (r'\b(M\.?\s*Cyb\.?\s*Sec\.?|MCybSec|Master\s+of\s+Cyber\s+Security)\b', 6),
            (r'\b(M\.?\s*SE|MSE|Master\s+of\s+Software\s+Engineering)\b', 6),
            (r'\b(M\.?\s*EE|MEE|Master\s+of\s+Electrical\s+Engineering)\b', 6),
            (r'\b(M\.?\s*ME|MME|Master\s+of\s+Mechanical\s+Engineering)\b', 6),
            (r'\b(M\.?\s*CE|MCE|Master\s+of\s+Civil\s+Engineering)\b', 6),
            (r'\b(M\.?\s*ChE|MChE|Master\s+of\s+Chemical\s+Engineering)\b', 6),
            (r'\b(M\.?\s*IE|MIE|Master\s+of\s+Industrial\s+Engineering)\b', 6),
            (r'\b(M\.?\s*Env\.?\s*E\.?|MEnvE|Master\s+of\s+Environmental\s+Engineering)\b', 6),
            (r'\b(M\.?\s*BME|MBME|Master\s+of\s+Biomedical\s+Engineering)\b', 6),
            (r'\b(M\.?\s*Aero\.?\s*E\.?|MAeroE|Master\s+of\s+Aerospace\s+Engineering)\b', 6),
            (r'\b(M\.?\s*Nuc\.?\s*E\.?|MNucE|Master\s+of\s+Nuclear\s+Engineering)\b', 6),
            (r'\b(M\.?\s*Mat\.?\s*E\.?|MMatE|Master\s+of\s+Materials\s+Engineering)\b', 6),
            (r'\b(M\.?\s*Pet\.?\s*E\.?|MPetE|Master\s+of\s+Petroleum\s+Engineering)\b', 6),
            (r'\b(M\.?\s*Mar\.?\s*E\.?|MMarE|Master\s+of\s+Marine\s+Engineering)\b', 6),
            (r'\b(M\.?\s*Min\.?\s*E\.?|MMinE|Master\s+of\s+Mining\s+Engineering)\b', 6),
            (r'\b(M\.?\s*Met\.?\s*E\.?|MMetE|Master\s+of\s+Metallurgical\s+Engineering)\b', 6),
            (r'\b(M\.?\s*Text\.?\s*E\.?|MTextE|Master\s+of\s+Textile\s+Engineering)\b', 6),
            (r'\b(M\.?\s*Ag\.?\s*E\.?|MAgE|Master\s+of\s+Agricultural\s+Engineering)\b', 6),
            (r'\b(M\.?\s*Food\s+Tech\.?|MFoodTech|Master\s+of\s+Food\s+Technology)\b', 6),
            (r'\b(M\.?\s*Biotech\.?|MBiotech|Master\s+of\s+Biotechnology)\b', 6),
            (r'\b(M\.?\s*Pharm\.?\s*Tech\.?|MPharmTech|Master\s+of\s+Pharmaceutical\s+Technology)\b', 6),
            (r'\b(M\.?\s*Nano\.?\s*Tech\.?|MNanoTech|Master\s+of\s+Nano\s+Technology)\b', 6),
            (r'\b(M\.?\s*Ren\.?\s*E\.?|MRenE|Master\s+of\s+Renewable\s+Energy\s+Engineering)\b', 6),
            (r'\b(M\.?\s*Auto\.?\s*E\.?|MAutoE|Master\s+of\s+Automation\s+Engineering)\b', 6),
            (r'\b(M\.?\s*Inst\.?\s*E\.?|MInstE|Master\s+of\s+Instrumentation\s+Engineering)\b', 6),
            (r'\b(M\.?\s*Mechatronics|MMechatronics|Master\s+of\s+Mechatronics)\b', 6),
            (r'\b(M\.?\s*Power\s+Sys\.?|MPowerSys|Master\s+of\s+Power\s+Systems\s+Engineering)\b', 6),
            (r'\b(M\.?\s*Comm\.?\s*E\.?|MCommE|Master\s+of\s+Communication\s+Engineering)\b', 6),
            (r'\b(M\.?\s*Opto\.?|MOpto|Master\s+of\s+Optoelectronics)\b', 6),
            (r'\b(M\.?\s*VLSI|MVLSI|Master\s+of\s+VLSI\s+Design)\b', 6),
            (r'\b(M\.?\s*Embedded|MEmbedded|Master\s+of\s+Embedded\s+Systems)\b', 6),
            (r'\b(M\.?\s*Cloud|MCloud|Master\s+of\s+Cloud\s+Computing)\b', 6),
            (r'\b(M\.?\s*Robotics|MRobotics|Master\s+of\s+Robotics)\b', 6),
            (r'\b(M\.?\s*AI\s+&\s+ML|MAIML|Master\s+of\s+Artificial\s+Intelligence\s+and\s+Machine\s+Learning)\b', 6),
            (r'\b(M\.?\s*DS\s+&\s+Analytics|MDSAnalytics|Master\s+of\s+Data\s+Science\s+and\s+Analytics)\b', 6),
            (r'\b(M\.?\s*PM|MPM|Master\s+of\s+Project\s+Management)\b', 6),
            (r'\b(M\.?\s*OM|MOM|Master\s+of\s+Operations\s+Management)\b', 6),
            (r'\b(M\.?\s*SCM|MSCM|Master\s+of\s+Supply\s+Chain\s+Management)\b', 6),
            (r'\b(M\.?\s*Log\.?|MLog|Master\s+of\s+Logistics)\b', 6),
            (r'\b(M\.?\s*Retail|MRetail|Master\s+of\s+Retail\s+Management)\b', 6),
            (r'\b(M\.?\s*Tour\.?|MTour|Master\s+of\s+Tourism\s+Management)\b', 6),
            (r'\b(M\.?\s*Hotel|MHotel|Master\s+of\s+Hotel\s+Management)\b', 6),
            (r'\b(M\.?\s*HM|MHM|Master\s+of\s+Hospitality\s+Management)\b', 6),
            (r'\b(M\.?\s*Arch\.?|MArch|Master\s+of\s+Architecture)\b', 6),
            (r'\b(M\.?\s*Des\.?|MDes|Master\s+of\s+Design)\b', 6),
            (r'\b(M\.?\s*ID|MID|Master\s+of\s+Industrial\s+Design)\b', 6),
            (r'\b(M\.?\s*UX|MUX|Master\s+of\s+UX\s+Design)\b', 6),
            (r'\b(M\.?\s*Prod\.?\s*Des\.?|MProdDes|Master\s+of\s+Product\s+Design)\b', 6),
            (r'\b(M\.?\s*Game\s+Des\.?|MGameDes|Master\s+of\s+Game\s+Design)\b', 6),
            (r'\b(M\.?\s*Animation|MAnimation|Master\s+of\s+Animation)\b', 6),
            (r'\b(M\.?\s*Film|MFilm|Master\s+of\s+Film\s+Studies)\b', 6),
            (r'\b(M\.?\s*Media|MMedia|Master\s+of\s+Media\s+Studies)\b', 6),
            (r'\b(M\.?\s*Fashion|MFashion|Master\s+of\s+Fashion\s+Technology)\b', 6),
            (r'\b(M\.?\s*Apparel|MApparel|Master\s+of\s+Apparel\s+Design)\b', 6),
            (r'\b(M\.?\s*Textile|MTextile|Master\s+of\s+Textile\s+Engineering)\b', 6),
            (r'\b(M\.?\s*Ceramic|MCeramic|Master\s+of\s+Ceramic\s+Technology)\b', 6),
            (r'\b(M\.?\s*Polymer|MPolymer|Master\s+of\s+Polymer\s+Science)\b', 6),
            (r'\b(M\.?\s*Leather|MLeather|Master\s+of\s+Leather\s+Technology)\b', 6),
            (r'\b(M\.?\s*Mar\.?\s*Bio\.?|MMarBio|Master\s+of\s+Marine\s+Biology)\b', 6),
            (r'\b(M\.?\s*Ocean\.?|MOcean|Master\s+of\s+Oceanography)\b', 6),
            (r'\b(M\.?\s*Fish\.?|MFish|Master\s+of\s+Fisheries\s+Science)\b', 6),
            (r'\b(M\.?\s*Food\s+Tech\.?|MFoodTech|Master\s+of\s+Food\s+Technology)\b', 6),
            (r'\b(M\.?\s*Dairy|MDairy|Master\s+of\s+Dairy\s+Technology)\b', 6),
            (r'\b(M\.?\s*Rural\s+Dev\.?|MRuralDev|Master\s+of\s+Rural\s+Development)\b', 6),
            (r'\b(M\.?\s*Soc\.?\s*Sci\.?|MSocSci|Master\s+of\s+Social\s+Sciences)\b', 6),
            (r'\b(M\.?\s*Gender|MGender|Master\s+of\s+Gender\s+Studies)\b', 6),
            (r'\b(M\.?\s*Peace|MPeace|Master\s+of\s+Peace\s+Studies)\b', 6),
            (r'\b(M\.?\s*Disaster|MDisaster|Master\s+of\s+Disaster\s+Management)\b', 6),
            (r'\b(M\.?\s*HRM|MHRM|Master\s+of\s+Human\s+Resource\s+Management)\b', 6),
            (r'\b(M\.?\s*OB|MOB|Master\s+of\s+Organizational\s+Behavior)\b', 6),
            (r'\b(M\.?\s*Ent\.?|MEnt|Master\s+of\s+Entrepreneurship)\b', 6),
            (r'\b(M\.?\s*MM|MMM|Master\s+of\s+Marketing\s+Management)\b', 6),
            (r'\b(M\.?\s*FM|MFM|Master\s+of\s+Finance\s+Management)\b', 6),
            (r'\b(M\.?\s*Banking|MBanking|Master\s+of\s+Banking\s+and\s+Insurance)\b', 6),
            (r'\b(M\.?\s*IM|MIM|Master\s+of\s+Investment\s+Management)\b', 6),
            (r'\b(M\.?\s*RM|MRM|Master\s+of\s+Risk\s+Management)\b', 6),
            (r'\b(M\.?\s*Audit|MAudit|Master\s+of\s+Auditing)\b', 6),
            (r'\b(M\.?\s*Tax|MTax|Master\s+of\s+Taxation)\b', 6),
            (r'\b(M\.?\s*IEcon|MIEcon|Master\s+of\s+International\s+Economics)\b', 6),
            (r'\b(M\.?\s*BEcon|MBEcon|Master\s+of\s+Business\s+Economics)\b', 6),
            (r'\b(M\.?\s*Dev\.?\s*Stu\.?|MDevStu|Master\s+of\s+Development\s+Studies)\b', 6),
            (r'\b(M\.?\s*PHA|MPHA|Master\s+of\s+Public\s+Health\s+Administration)\b', 6),
            (r'\b(M\.?\s*OH|MOH|Master\s+of\s+Occupational\s+Health)\b', 6),
            (r'\b(M\.?\s*Epi\.?|MEpi|Master\s+of\s+Epidemiology)\b', 6),
            (r'\b(M\.?\s*Clin\.?\s*Psych\.?|MClinPsych|Master\s+of\s+Clinical\s+Psychology)\b', 6),
            (r'\b(M\.?\s*Cog\.?\s*Sci\.?|MCogSci|Master\s+of\s+Cognitive\s+Science)\b', 6),
            (r'\b(M\.?\s*Neuro|MNeuro|Master\s+of\s+Neuroscience)\b', 6),
            (r'\b(M\.?\s*BMS|MBMS|Master\s+of\s+Biomedical\s+Science)\b', 6),
            (r'\b(M\.?\s*Pharm\.?\s*Prac\.?|MPharmPrac|Master\s+of\s+Pharmacy\s+Practice)\b', 6),
            (r'\b(M\.?\s*Clin\.?\s*Res\.?|MClinRes|Master\s+of\s+Clinical\s+Research)\b', 6),
            (r'\b(M\.?\s*Nutr\.?|MNutr|Master\s+of\s+Nutrition\s+Science)\b', 6),
            (r'\b(M\.?\s*Para\.?|MPara|Master\s+of\s+Paramedical\s+Science)\b', 6),
            (r'\b(M\.?\s*Optom\.?|MOptom|Master\s+of\s+Optometry)\b', 6),
            (r'\b(M\.?\s*Immuno|MImmuno|Master\s+of\s+Immunology)\b', 6),
            (r'\b(M\.?\s*Virol\.?|MVirol|Master\s+of\s+Virology)\b', 6),
            (r'\b(M\.?\s*Genet\.?|MGenet|Master\s+of\s+Genetics)\b', 6),
            (r'\b(M\.?\s*Mar\.?\s*Trans\.?|MMarTrans|Master\s+of\s+Marine\s+Transportation)\b', 6),
            (r'\b(M\.?\s*Naut\.?|MNaut|Master\s+of\s+Nautical\s+Science)\b', 6),
            (r'\b(M\.?\s*Nav\.?\s*Arch\.?|MNavArch|Master\s+of\s+Naval\s+Architecture)\b', 6),
            (r'\b(M\.?\s*Space|MSpace|Master\s+of\s+Space\s+Science)\b', 6),
            (r'\b(M\.?\s*Astro\.?\s*Phys\.?|MAstroPhys|Master\s+of\s+Astrophysics)\b', 6),
            (r'\b(M\.?\s*Astron\.?|MAstron|Master\s+of\s+Astronomy)\b', 6),
            (r'\b(M\.?\s*Aviation|MAviation|Master\s+of\s+Aviation)\b', 6),
            (r'\b(M\.?\s*ATM|MATM|Master\s+of\s+Air\s+Traffic\s+Management)\b', 6),
            (r'\b(M\.?\s*Aero\.?|MAero|Master\s+of\s+Aeronautics)\b', 6),
            (r'\b(M\.?\s*UP|MUP|Master\s+of\s+Urban\s+Planning)\b', 6),
            (r'\b(M\.?\s*Geo\.?\s*Info\.?|MGeoInfo|Master\s+of\s+Geoinformatics)\b', 6),
            (r'\b(M\.?\s*GIS|MGIS|Master\s+of\s+GIS)\b', 6),
            (r'\b(M\.?\s*Surv\.?|MSurv|Master\s+of\s+Surveying)\b', 6),
            (r'\b(M\.?\s*CM|MCM|Master\s+of\s+Construction\s+Management)\b', 6),
            (r'\b(M\.?\s*SE|MSE|Master\s+of\s+Structural\s+Engineering)\b', 6),
            (r'\b(M\.?\s*TE|MTE|Master\s+of\s+Transportation\s+Engineering)\b', 6),
            (r'\b(M\.?\s*WRE|MWRE|Master\s+of\s+Water\s+Resources\s+Engineering)\b', 6),
            (r'\b(M\.?\s*Env\.?\s*E\.?|MEnvE|Master\s+of\s+Environmental\s+Engineering)\b', 6),
            (r'\b(M\.?\s*Ecol\.?|MEcol|Master\s+of\s+Ecology)\b', 6),
            (r'\b(M\.?\s*Climate|MClimate|Master\s+of\s+Climate\s+Science)\b', 6),
            (r'\b(M\.?\s*Earth|MEarth|Master\s+of\s+Earth\s+Science)\b', 6),
            (r'\b(M\.?\s*Geol\.?|MGeol|Master\s+of\s+Geology)\b', 6),
            (r'\b(M\.?\s*Geophys\.?|MGeophys|Master\s+of\s+Geophysics)\b', 6),
            (r'\b(M\.?\s*Pet\.?\s*Geo\.?|MPetGeo|Master\s+of\s+Petroleum\s+Geoscience)\b', 6),
            
            # ==================== Additional Bachelor's Variations ====================
            # More BE/BTech variations
            (r'\b(B\.?\s*Eng\.?|BEng|Bachelor\s+of\s+Engineering)\b', 5),
            (r'\b(B\.?\s*Tech\.?\s+(?:in|of)\s+[A-Za-z][A-Za-z\s&/\-\.]+)', 5),
            (r'\b(B\.?\s*E\.?\s+(?:in|of)\s+[A-Za-z][A-Za-z\s&/\-\.]+)', 5),
            (r'\b(B\.?\s*Arch\.?|BArch|Bachelor\s+of\s+Architecture)\b', 5),
            (r'\b(B\.?\s*Plan\.?|BPlan|Bachelor\s+of\s+Planning)\b', 5),
            (r'\b(B\.?\s*Des\.?|BDes|Bachelor\s+of\s+Design)\b', 5),
            (r'\b(B\.?\s*FA|BFA|Bachelor\s+of\s+Fine\s+Arts)\b', 5),
            (r'\b(B\.?\s*VC|BVC|Bachelor\s+of\s+Visual\s+Communication)\b', 5),
            (r'\b(B\.?\s*MC|BMC|Bachelor\s+of\s+Mass\s+Communication)\b', 5),
            (r'\b(B\.?\s*Jour\.?|BJour|Bachelor\s+of\s+Journalism)\b', 5),
            (r'\b(B\.?\s*SW|BSW|Bachelor\s+of\s+Social\s+Work)\b', 5),
            (r'\b(B\.?\s*Ed\.?|BEd|Bachelor\s+of\s+Education)\b', 5),
            (r'\b(B\.?\s*Sp\.?\s*Ed\.?|BSpEd|Bachelor\s+of\s+Special\s+Education)\b', 5),
            (r'\b(B\.?\s*ECE|BECE|Bachelor\s+of\s+Early\s+Childhood\s+Education)\b', 5),
            (r'\b(B\.?\s*Tour\.?|BTour|Bachelor\s+of\s+Tourism)\b', 5),
            (r'\b(B\.?\s*HM|BHM|Bachelor\s+of\s+Hotel\s+Management)\b', 5),
            (r'\b(B\.?\s*Hosp\.?|BHosp|Bachelor\s+of\s+Hospitality\s+Management)\b', 5),
            (r'\b(B\.?\s*TTM|BTTM|Bachelor\s+of\s+Travel\s+&\s+Tourism\s+Management)\b', 5),
            (r'\b(B\.?\s*BM|BBM|Bachelor\s+of\s+Business\s+Management)\b', 5),
            (r'\b(B\.?\s*BS|BBS|Bachelor\s+of\s+Business\s+Studies)\b', 5),
            (r'\b(B\.?\s*SM|BSM|Bachelor\s+of\s+Strategic\s+Management)\b', 5),
            (r'\b(B\.?\s*PR|BPR|Bachelor\s+of\s+Public\s+Relations)\b', 5),
            (r'\b(B\.?\s*Comm\.?\s*E\.?|BCommE|Bachelor\s+of\s+Communications\s+Engineering)\b', 5),
            (r'\b(B\.?\s*Biochem\.?|BBiochem|Bachelor\s+of\s+Biochemistry)\b', 5),
            (r'\b(B\.?\s*Biotech\.?|BBiotech|Bachelor\s+of\s+Biotechnology)\b', 5),
            (r'\b(B\.?\s*Micro\.?|BMicro|Bachelor\s+of\s+Microbiology)\b', 5),
            (r'\b(B\.?\s*MB|BMB|Bachelor\s+of\s+Molecular\s+Biology)\b', 5),
            (r'\b(B\.?\s*Phys\.?|BPhys|Bachelor\s+of\s+Physics)\b', 5),
            (r'\b(B\.?\s*Chem\.?|BChem|Bachelor\s+of\s+Chemistry)\b', 5),
            (r'\b(B\.?\s*Math\.?|BMath|Bachelor\s+of\s+Mathematics)\b', 5),
            (r'\b(B\.?\s*Stat\.?|BStat|Bachelor\s+of\s+Statistics)\b', 5),
            (r'\b(B\.?\s*AI|BAI|Bachelor\s+of\s+Artificial\s+Intelligence)\b', 5),
            (r'\b(B\.?\s*ML|BML|Bachelor\s+of\s+Machine\s+Learning)\b', 5),
            (r'\b(B\.?\s*DS|BDS|Bachelor\s+of\s+Data\s+Science)\b', 5),
            (r'\b(B\.?\s*Robotics|BRobotics|Bachelor\s+of\s+Robotics)\b', 5),
            (r'\b(B\.?\s*Cyb\.?\s*Sec\.?|BCybSec|Bachelor\s+of\s+Cyber\s+Security)\b', 5),
            (r'\b(B\.?\s*Cloud|BCloud|Bachelor\s+of\s+Cloud\s+Computing)\b', 5),
            (r'\b(B\.?\s*Embedded|BEmbedded|Bachelor\s+of\s+Embedded\s+Systems)\b', 5),
            (r'\b(B\.?\s*VLSI|BVLSI|Bachelor\s+of\s+VLSI\s+Design)\b', 5),
            (r'\b(B\.?\s*NE|BNE|Bachelor\s+of\s+Network\s+Engineering)\b', 5),
            (r'\b(B\.?\s*SS|BSS|Bachelor\s+of\s+Software\s+Systems)\b', 5),
            (r'\b(B\.?\s*HS|BHS|Bachelor\s+of\s+Health\s+Sciences)\b', 5),
            (r'\b(B\.?\s*Nurs\.?|BNurs|Bachelor\s+of\s+Nursing)\b', 5),
            (r'\b(B\.?\s*Pharm\.?|BPharm|Bachelor\s+of\s+Pharmacy)\b', 5),
            (r'\b(B\.?\s*PT|BPT|Bachelor\s+of\s+Physiotherapy)\b', 5),
            (r'\b(B\.?\s*OT|BOT|Bachelor\s+of\s+Occupational\s+Therapy)\b', 5),
            (r'\b(B\.?\s*Nutr\.?|BNutr|Bachelor\s+of\s+Nutrition)\b', 5),
            (r'\b(B\.?\s*Diet\.?|BDiet|Bachelor\s+of\s+Dietetics)\b', 5),
            (r'\b(B\.?\s*MLT|BMLT|Bachelor\s+of\s+Medical\s+Laboratory\s+Technology)\b', 5),
            (r'\b(B\.?\s*Rad\.?|BRad|Bachelor\s+of\s+Radiology)\b', 5),
            (r'\b(B\.?\s*Ophth\.?|BOphth|Bachelor\s+of\s+Ophthalmology)\b', 5),
            (r'\b(B\.?\s*DS|BDS|Bachelor\s+of\s+Dental\s+Surgery)\b', 5),
            (r'\b(B\.?\s*PH|BPH|Bachelor\s+of\s+Public\s+Health)\b', 5),
            (r'\b(B\.?\s*Agri\.?|BAgri|Bachelor\s+of\s+Agriculture)\b', 5),
            (r'\b(B\.?\s*Hort\.?|BHort|Bachelor\s+of\s+Horticulture)\b', 5),
            (r'\b(B\.?\s*AS|BAS|Bachelor\s+of\s+Animal\s+Science)\b', 5),
            (r'\b(B\.?\s*Fish\.?|BFish|Bachelor\s+of\s+Fisheries)\b', 5),
            (r'\b(B\.?\s*VS|BVS|Bachelor\s+of\s+Veterinary\s+Science)\b', 5),
            (r'\b(B\.?\s*Com\.?\s*Hons\.?|BComHons|Bachelor\s+of\s+Commerce\s+Honors)\b', 5),
            (r'\b(B\.?\s*Acc\.?|BAcc|Bachelor\s+of\s+Accounting)\b', 5),
            (r'\b(B\.?\s*Fin\.?|BFin|Bachelor\s+of\s+Finance)\b', 5),
            (r'\b(B\.?\s*Act\.?\s*Sci\.?|BActSci|Bachelor\s+of\s+Actuarial\s+Science)\b', 5),
            (r'\b(B\.?\s*Crim\.?|BCrim|Bachelor\s+of\s+Criminology)\b', 5),
            (r'\b(B\.?\s*CJ|BCJ|Bachelor\s+of\s+Criminal\s+Justice)\b', 5),
            (r'\b(B\.?\s*LS|BLS|Bachelor\s+of\s+Legal\s+Studies)\b', 5),
            (r'\b(B\.?\s*FS|BFS|Bachelor\s+of\s+Forensic\s+Science)\b', 5),
            (r'\b(B\.?\s*HS|BHS|Bachelor\s+of\s+Homeland\s+Security)\b', 5),
            (r'\b(B\.?\s*PS|BPS|Bachelor\s+of\s+Political\s+Science)\b', 5),
            (r'\b(B\.?\s*PA|BPA|Bachelor\s+of\s+Public\s+Administration)\b', 5),
            (r'\b(B\.?\s*PP|BPP|Bachelor\s+of\s+Public\s+Policy)\b', 5),
            (r'\b(B\.?\s*IR|BIR|Bachelor\s+of\s+International\s+Relations)\b', 5),
            (r'\b(B\.?\s*Anthro\.?|BAnthro|Bachelor\s+of\s+Anthropology)\b', 5),
            (r'\b(B\.?\s*Soc\.?|BSoc|Bachelor\s+of\s+Sociology)\b', 5),
            (r'\b(B\.?\s*Psych\.?|BPsych|Bachelor\s+of\s+Psychology)\b', 5),
            (r'\b(B\.?\s*Phil\.?|BPhil|Bachelor\s+of\s+Philosophy)\b', 5),
            (r'\b(B\.?\s*Hist\.?|BHist|Bachelor\s+of\s+History)\b', 5),
            (r'\b(B\.?\s*Geo\.?|BGeo|Bachelor\s+of\s+Geography)\b', 5),
            (r'\b(B\.?\s*Hum\.?|BHum|Bachelor\s+of\s+Humanities)\b', 5),
            (r'\b(B\.?\s*LA|BLA|Bachelor\s+of\s+Liberal\s+Arts)\b', 5),
            (r'\b(B\.?\s*Theol\.?|BTheol|Bachelor\s+of\s+Theology)\b', 5),
            (r'\b(B\.?\s*Ren\.?\s*E\.?|BRenE|Bachelor\s+of\s+Renewable\s+Energy\s+Engineering)\b', 5),
            (r'\b(B\.?\s*Auto\.?\s*E\.?|BAutoE|Bachelor\s+of\s+Automation\s+Engineering)\b', 5),
            (r'\b(B\.?\s*Inst\.?\s*E\.?|BInstE|Bachelor\s+of\s+Instrumentation\s+Engineering)\b', 5),
            (r'\b(B\.?\s*Mechatronics|BMechatronics|Bachelor\s+of\s+Mechatronics)\b', 5),
            (r'\b(B\.?\s*Power\s+Sys\.?|BPowerSys|Bachelor\s+of\s+Power\s+Systems\s+Engineering)\b', 5),
            (r'\b(B\.?\s*Comm\.?\s*Sys\.?|BCommSys|Bachelor\s+of\s+Communication\s+Systems)\b', 5),
            (r'\b(B\.?\s*Nano\.?\s*Tech\.?|BNanoTech|Bachelor\s+of\s+Nano\s+Technology)\b', 5),
            (r'\b(B\.?\s*Opto\.?|BOpto|Bachelor\s+of\s+Optoelectronics)\b', 5),
            (r'\b(B\.?\s*Mil\.?\s*Sci\.?|BMilSci|Bachelor\s+of\s+Military\s+Science)\b', 5),
            (r'\b(B\.?\s*Def\.?\s*Stu\.?|BDefStu|Bachelor\s+of\s+Defense\s+Studies)\b', 5),
            (r'\b(B\.?\s*Strat\.?\s*Stu\.?|BStratStu|Bachelor\s+of\s+Strategic\s+Studies)\b', 5),
            (r'\b(B\.?\s*Int\.?\s*Stu\.?|BIntStu|Bachelor\s+of\s+Intelligence\s+Studies)\b', 5),
            (r'\b(B\.?\s*GS|BGS|Bachelor\s+of\s+Global\s+Studies)\b', 5),
            (r'\b(B\.?\s*IB|BIB|Bachelor\s+of\s+International\s+Business)\b', 5),
            (r'\b(B\.?\s*SC|BSC|Bachelor\s+of\s+Supply\s+Chain)\b', 5),
            (r'\b(B\.?\s*Log\.?|BLog|Bachelor\s+of\s+Logistics)\b', 5),
            (r'\b(B\.?\s*OR|BOR|Bachelor\s+of\s+Operations\s+Research)\b', 5),
            (r'\b(B\.?\s*BDA|BBDA|Bachelor\s+of\s+Big\s+Data\s+Analytics)\b', 5),
            (r'\b(B\.?\s*FinTech|BFinTech|Bachelor\s+of\s+FinTech)\b', 5),
            (r'\b(B\.?\s*Blockchain|BBlockchain|Bachelor\s+of\s+Blockchain)\b', 5),
            (r'\b(B\.?\s*UX|BUX|Bachelor\s+of\s+UX\s+Design)\b', 5),
            (r'\b(B\.?\s*Prod\.?\s*Des\.?|BProdDes|Bachelor\s+of\s+Product\s+Design)\b', 5),
            (r'\b(B\.?\s*Game\s+Des\.?|BGameDes|Bachelor\s+of\s+Game\s+Design)\b', 5),
            (r'\b(B\.?\s*Animation|BAnimation|Bachelor\s+of\s+Animation)\b', 5),
            (r'\b(B\.?\s*Film|BFilm|Bachelor\s+of\s+Film\s+Studies)\b', 5),
            (r'\b(B\.?\s*Media|BMedia|Bachelor\s+of\s+Media\s+Studies)\b', 5),
            (r'\b(B\.?\s*Fashion|BFashion|Bachelor\s+of\s+Fashion\s+Technology)\b', 5),
            (r'\b(B\.?\s*Apparel|BApparel|Bachelor\s+of\s+Apparel\s+Design)\b', 5),
            (r'\b(B\.?\s*Textile\s+Eng\.?|BTextileEng|Bachelor\s+of\s+Textile\s+Engineering)\b', 5),
            (r'\b(B\.?\s*Ceramic|BCeramic|Bachelor\s+of\s+Ceramic\s+Technology)\b', 5),
            (r'\b(B\.?\s*Polymer|BPolymer|Bachelor\s+of\s+Polymer\s+Science)\b', 5),
            (r'\b(B\.?\s*Leather|BLeather|Bachelor\s+of\s+Leather\s+Technology)\b', 5),
            (r'\b(B\.?\s*Mar\.?\s*Bio\.?|BMarBio|Bachelor\s+of\s+Marine\s+Biology)\b', 5),
            (r'\b(B\.?\s*Ocean\.?|BOcean|Bachelor\s+of\s+Oceanography)\b', 5),
            (r'\b(B\.?\s*Fish\.?\s*Sci\.?|BFishSci|Bachelor\s+of\s+Fisheries\s+Science)\b', 5),
            (r'\b(B\.?\s*Food\s+Tech\.?|BFoodTech|Bachelor\s+of\s+Food\s+Technology)\b', 5),
            (r'\b(B\.?\s*Dairy\s+Tech\.?|BDairyTech|Bachelor\s+of\s+Dairy\s+Technology)\b', 5),
            (r'\b(B\.?\s*Rural\s+Dev\.?|BRuralDev|Bachelor\s+of\s+Rural\s+Development)\b', 5),
            (r'\b(B\.?\s*Soc\.?\s*Sci\.?|BSocSci|Bachelor\s+of\s+Social\s+Sciences)\b', 5),
            (r'\b(B\.?\s*Gender|BGender|Bachelor\s+of\s+Gender\s+Studies)\b', 5),
            (r'\b(B\.?\s*Peace|BPeace|Bachelor\s+of\s+Peace\s+Studies)\b', 5),
            (r'\b(B\.?\s*Disaster|BDisaster|Bachelor\s+of\s+Disaster\s+Management)\b', 5),
            (r'\b(B\.?\s*HRM|BHRM|Bachelor\s+of\s+Human\s+Resource\s+Management)\b', 5),
            (r'\b(B\.?\s*OB|BOB|Bachelor\s+of\s+Organizational\s+Behavior)\b', 5),
            (r'\b(B\.?\s*Ent\.?|BEnt|Bachelor\s+of\s+Entrepreneurship)\b', 5),
            (r'\b(B\.?\s*MM|BMM|Bachelor\s+of\s+Marketing\s+Management)\b', 5),
            (r'\b(B\.?\s*FM|BFM|Bachelor\s+of\s+Finance\s+Management)\b', 5),
            (r'\b(B\.?\s*Banking|BBanking|Bachelor\s+of\s+Banking\s+and\s+Insurance)\b', 5),
            (r'\b(B\.?\s*IM|BIM|Bachelor\s+of\s+Investment\s+Management)\b', 5),
            (r'\b(B\.?\s*RM|BRM|Bachelor\s+of\s+Risk\s+Management)\b', 5),
            (r'\b(B\.?\s*Audit|BAudit|Bachelor\s+of\s+Auditing)\b', 5),
            (r'\b(B\.?\s*Tax|BTax|Bachelor\s+of\s+Taxation)\b', 5),
            (r'\b(B\.?\s*IEcon|BIEcon|Bachelor\s+of\s+International\s+Economics)\b', 5),
            (r'\b(B\.?\s*BEcon|BBEcon|Bachelor\s+of\s+Business\s+Economics)\b', 5),
            (r'\b(B\.?\s*Dev\.?\s*Stu\.?|BDevStu|Bachelor\s+of\s+Development\s+Studies)\b', 5),
            (r'\b(B\.?\s*PHA|BPHA|Bachelor\s+of\s+Public\s+Health\s+Administration)\b', 5),
            (r'\b(B\.?\s*Clin\.?\s*Psych\.?|BClinPsych|Bachelor\s+of\s+Clinical\s+Psychology)\b', 5),
            (r'\b(B\.?\s*Cog\.?\s*Sci\.?|BCogSci|Bachelor\s+of\s+Cognitive\s+Science)\b', 5),
            (r'\b(B\.?\s*Neuro|BNeuro|Bachelor\s+of\s+Neuroscience)\b', 5),
            (r'\b(B\.?\s*BMS|BBMS|Bachelor\s+of\s+Biomedical\s+Science)\b', 5),
            (r'\b(B\.?\s*Pharm\.?\s*Prac\.?|BPharmPrac|Bachelor\s+of\s+Pharmacy\s+Practice)\b', 5),
            (r'\b(B\.?\s*Med\.?\s*Tech\.?|BMedTech|Bachelor\s+of\s+Medical\s+Technology)\b', 5),
            (r'\b(B\.?\s*Nutr\.?\s*Sci\.?|BNutrSci|Bachelor\s+of\s+Nutrition\s+Science)\b', 5),
            (r'\b(B\.?\s*Para\.?|BPara|Bachelor\s+of\s+Paramedical\s+Science)\b', 5),
            (r'\b(B\.?\s*Optom\.?|BOptom|Bachelor\s+of\s+Optometry)\b', 5),
            (r'\b(B\.?\s*Prosth\.?|BProsth|Bachelor\s+of\s+Prosthetics)\b', 5),
            (r'\b(B\.?\s*Audio\.?|BAudio|Bachelor\s+of\s+Audiology)\b', 5),
            (r'\b(B\.?\s*SP|BSP|Bachelor\s+of\s+Speech\s+Pathology)\b', 5),
            (r'\b(B\.?\s*Clin\.?\s*Res\.?|BClinRes|Bachelor\s+of\s+Clinical\s+Research)\b', 5),
            (r'\b(B\.?\s*Immuno|BImmuno|Bachelor\s+of\s+Immunology)\b', 5),
            (r'\b(B\.?\s*Virol\.?|BVirol|Bachelor\s+of\s+Virology)\b', 5),
            (r'\b(B\.?\s*Genet\.?|BGenet|Bachelor\s+of\s+Genetics)\b', 5),
            (r'\b(B\.?\s*Mar\.?\s*Trans\.?|BMarTrans|Bachelor\s+of\s+Marine\s+Transportation)\b', 5),
            (r'\b(B\.?\s*Naut\.?|BNaut|Bachelor\s+of\s+Nautical\s+Science)\b', 5),
            (r'\b(B\.?\s*Nav\.?\s*Arch\.?|BNavArch|Bachelor\s+of\s+Naval\s+Architecture)\b', 5),
            (r'\b(B\.?\s*Mar\.?\s*E\.?|BMarE|Bachelor\s+of\s+Maritime\s+Engineering)\b', 5),
            (r'\b(B\.?\s*Aviation|BAviation|Bachelor\s+of\s+Aviation)\b', 5),
            (r'\b(B\.?\s*Pilot|BPilot|Bachelor\s+of\s+Pilot\s+Studies)\b', 5),
            (r'\b(B\.?\s*ATM|BATM|Bachelor\s+of\s+Air\s+Traffic\s+Management)\b', 5),
            (r'\b(B\.?\s*Aero\.?|BAero|Bachelor\s+of\s+Aeronautics)\b', 5),
            (r'\b(B\.?\s*Space|BSpace|Bachelor\s+of\s+Space\s+Science)\b', 5),
            (r'\b(B\.?\s*Astro\.?\s*Phys\.?|BAstroPhys|Bachelor\s+of\s+Astrophysics)\b', 5),
            (r'\b(B\.?\s*Astron\.?|BAstron|Bachelor\s+of\s+Astronomy)\b', 5),
            (r'\b(B\.?\s*Surv\.?|BSurv|Bachelor\s+of\s+Surveying)\b', 5),
            (r'\b(B\.?\s*UP|BUP|Bachelor\s+of\s+Urban\s+Planning)\b', 5),
            (r'\b(B\.?\s*Geo\.?\s*Info\.?|BGeoInfo|Bachelor\s+of\s+Geoinformatics)\b', 5),
            (r'\b(B\.?\s*GIS|BGIS|Bachelor\s+of\s+Geographic\s+Information\s+Systems)\b', 5),
            (r'\b(B\.?\s*Cart\.?|BCart|Bachelor\s+of\s+Cartography)\b', 5),
            (r'\b(B\.?\s*CM|BCM|Bachelor\s+of\s+Construction\s+Management)\b', 5),
            (r'\b(B\.?\s*SE|BSE|Bachelor\s+of\s+Structural\s+Engineering)\b', 5),
            (r'\b(B\.?\s*TE|BTE|Bachelor\s+of\s+Transportation\s+Engineering)\b', 5),
            (r'\b(B\.?\s*WRE|BWRE|Bachelor\s+of\s+Water\s+Resources\s+Engineering)\b', 5),
            (r'\b(B\.?\s*Env\.?\s*E\.?|BEnvE|Bachelor\s+of\s+Environmental\s+Engineering)\b', 5),
            (r'\b(B\.?\s*Ecol\.?|BEcol|Bachelor\s+of\s+Ecology)\b', 5),
            (r'\b(B\.?\s*Climate|BClimate|Bachelor\s+of\s+Climate\s+Science)\b', 5),
            (r'\b(B\.?\s*Earth|BEarth|Bachelor\s+of\s+Earth\s+Science)\b', 5),
            (r'\b(B\.?\s*Geol\.?|BGeol|Bachelor\s+of\s+Geology)\b', 5),
            (r'\b(B\.?\s*Geophys\.?|BGeophys|Bachelor\s+of\s+Geophysics)\b', 5),
            (r'\b(B\.?\s*Pet\.?\s*Geo\.?|BPetGeo|Bachelor\s+of\s+Petroleum\s+Geoscience)\b', 5),
            (r'\b(B\.?\s*Min\.?\s*Safe\.?|BMinSafe|Bachelor\s+of\s+Mining\s+Safety)\b', 5),
            
            # ==================== Additional Associate's Variations ====================
            (r'\b(A\.?\s*AS|AAS|Associate\s+of\s+Applied\s+Science)\b', 4),
            (r'\b(A\.?\s*AA|AAA|Associate\s+of\s+Arts)\b', 4),
            (r'\b(A\.?\s*S\.?\s+(?:in|of)\s+[A-Za-z][A-Za-z\s&/\-\.]+)', 4),
            (r'\b(A\.?\s*A\.?\s+(?:in|of)\s+[A-Za-z][A-Za-z\s&/\-\.]+)', 4),
            
            # ==================== Additional Diploma Variations ====================
            (r'\b(Dip\.?\s+(?:in|of)\s+[A-Za-z][A-Za-z\s&/\-\.]+)', 3),
            (r'\b(Diploma\s+(?:in|of)\s+[A-Za-z][A-Za-z\s&/\-\.]+)', 3),
            (r'\b(Adv\.?\s*Dip\.?|AdvDip|Advanced\s+Diploma)\b', 3),
            (r'\b(Grad\.?\s*Dip\.?|GradDip|Graduate\s+Diploma)\b', 3),
            (r'\b(Post\s*Grad\.?\s*Dip\.?|PostGradDip|Post\s+Graduate\s+Diploma)\b', 3),
            (r'\b(PG\s*Dip\.?|PGDip|PG\s+Diploma)\b', 3),
            (r'\b(PGDCA|P\.?\s*G\.?\s*D\.?\s*C\.?\s*A\.?)\b', 3),
            (r'\b(PGDM|P\.?\s*G\.?\s*D\.?\s*M\.?)\b', 3),
            (r'\b(Poly\.?|Polytechnic)\b', 3),
            
            # ==================== Additional Certificate Variations ====================
            (r'\b(Cert\.?\s+(?:in|of)\s+[A-Za-z][A-Za-z\s&/\-\.]+)', 2),
            (r'\b(Certificate\s+(?:in|of)\s+[A-Za-z][A-Za-z\s&/\-\.]+)', 2),
            (r'\b(Prof\.?\s*Cert\.?|ProfCert|Professional\s+Certificate)\b', 2),
            (r'\b(Adv\.?\s*Cert\.?|AdvCert|Advanced\s+Certificate)\b', 2),
            (r'\b(Grad\.?\s*Cert\.?|GradCert|Graduate\s+Certificate)\b', 2),
            (r'\b(Post\s*Grad\.?\s*Cert\.?|PostGradCert|Post\s+Graduate\s+Certificate)\b', 2),
            (r'\b(PG\s*Cert\.?|PGCert|PG\s+Certificate)\b', 2),
            
            # ==================== Additional High School Variations ====================
            (r'\b(HS\s*Dip\.?|HSDip|High\s+School\s+Diploma)\b', 1),
            (r'\b(GED|General\s+Educational\s+Development)\b', 1),
            (r'\b(GCSE|General\s+Certificate\s+of\s+Secondary\s+Education)\b', 1),
            (r'\b(A\s*Level|A\s*Levels|Advanced\s+Level)\b', 1),
            (r'\b(O\s*Level|O\s*Levels|Ordinary\s+Level)\b', 1),
            (r'\b(IB\s*Diploma|International\s+Baccalaureate)\b', 1),
            (r'\b(AP\s*Program|Advanced\s+Placement)\b', 1),
            (r'\b(Matric|Matriculation)\b', 1),
            (r'\b(Plus\s*Two|10\+2|12th\s+Standard)\b', 1),
            (r'\b(PUC|Pre\s*University\s+Course)\b', 1),
            (r'\b(Pre\s*Degree|Pre\s*Degree\s+Course)\b', 1),
            (r'\b(Foundation\s+Course|Foundation\s+Program)\b', 1),
        ]
        
        for pattern, level in full_degree_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match and len(match.strip()) > 2:
                    # Clean the extracted degree text
                    clean_degree = match.strip()
                    clean_degree = re.sub(r'\s+', ' ', clean_degree)  # normalize spaces
                    
                    # Truncate at common stop points and skill-related keywords
                    # Stop words that indicate end of education and start of skills/other sections
                    skill_stop_words = [
                        ' skills', ' skill', ' technical skills', ' core competencies',
                        ' programming', ' languages', ' frameworks', ' tools', ' technologies',
                        ' python', ' java', ' javascript', ' html', ' css', ' sql', ' mysql',
                        ' numpy', ' pandas', ' django', ' flask', ' react', ' angular', ' vue',
                        ' experience', ' work experience', ' projects', ' certifications',
                        ' achievements', ' awards', ' publications', ' references'
                    ]
                    
                    # Check for skill stop words (case-insensitive)
                    # Also check for standalone "Skills" keyword (most common issue)
                    lower_degree = clean_degree.lower()
                    
                    # First, check for standalone "Skills" keyword (most aggressive)
                    skills_pattern = r'\s+skills?\s+'
                    if re.search(skills_pattern, lower_degree):
                        match = re.search(skills_pattern, lower_degree)
                        if match:
                            clean_degree = clean_degree[:match.start()].strip()
                            lower_degree = clean_degree.lower()
                    
                    # Then check other skill stop words
                    for stop_word in skill_stop_words:
                        if stop_word in lower_degree:
                            idx = lower_degree.find(stop_word)
                            clean_degree = clean_degree[:idx].strip()
                            break
                    
                    # Also truncate at common stop points
                    # BUT preserve specializations - don't truncate if it's part of the degree name
                    # Only truncate at these if they're followed by dates/years or other non-degree text
                    for stop in [' at ', ' - ', '\n', '|', ',  ', '  ']:
                        if stop in clean_degree:
                            # Check if what comes after the stop is a year/date (e.g., "– 2016")
                            parts = clean_degree.split(stop, 1)
                            if len(parts) == 2:
                                after_stop = parts[1].strip()
                                # If it's a year (4 digits) or date pattern, truncate there
                                if re.match(r'^\d{4}', after_stop) or re.match(r'^\d{1,2}[/-]\d{4}', after_stop):
                                    clean_degree = parts[0]
                                    break
                                # If it's clearly not part of degree (e.g., university name, location), truncate
                                if any(word in after_stop.lower() for word in ['university', 'college', 'institute', 'school', 'minor in', 'major in']):
                                    # Don't truncate - this might be part of the degree context
                                    pass
                                else:
                                    # Might be part of degree name, be conservative and don't truncate
                                    pass
                    
                    # Handle em dashes and en dashes separately (often used before years)
                    # CRITICAL: Only truncate if the dash is followed by a year/date
                    # DO NOT truncate if it's part of the degree name (e.g., "Bachelor's Degree in Civil Engineering – 2016")
                    for dash in ['–', '—', '-']:
                        if dash in clean_degree:
                            parts = clean_degree.split(dash, 1)
                            if len(parts) == 2:
                                before_dash = parts[0].strip()
                                after_dash = parts[1].strip()
                                
                                # CRITICAL: Only truncate if:
                                # 1. Followed by year/date (e.g., "– 2016")
                                # 2. AND the part before dash contains "Degree" or "in" (indicating it's a complete degree phrase)
                                # This prevents truncating "Bachelor's Degree in Civil Engineering" to just "Bachelor"
                                is_complete_degree = 'degree' in before_dash.lower() or ' in ' in before_dash.lower()
                                
                                if is_complete_degree and (re.match(r'^\d{4}', after_dash) or re.match(r'^\d{1,2}[/-]\d{4}', after_dash)):
                                    # Safe to truncate - we have a complete degree phrase before the dash
                                    clean_degree = before_dash
                                    break
                                # If followed by "(minor in" or "(major in", truncate at dash but keep degree
                                elif is_complete_degree and re.match(r'^\s*\(', after_dash):
                                    clean_degree = before_dash
                                    break
                                # Otherwise, don't truncate - might be part of degree name
                    
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
        # CRITICAL: If we have both a generic match (e.g., "Bachelor") and a specific match 
        # (e.g., "Bachelor's Degree in Civil Engineering"), prefer the specific one
        seen = {}
        for name, level, original in degrees:
            # Use first word as key to group related degrees (e.g., "bachelor", "bachelor's", "bachelors")
            name_lower = name.lower().strip()
            first_word = name_lower.split()[0] if name_lower.split() else name_lower[:10]
            key = first_word
            
            # If we already have a match for this key, prefer the longer/more specific one
            if key in seen:
                existing_name, existing_level, existing_orig = seen[key]
                existing_lower = existing_name.lower().strip()
                
                # CRITICAL: Always prefer longer, more specific names at the same level
                # e.g., "Bachelor's Degree in Civil Engineering" > "Bachelor"
                if level == existing_level:
                    if len(name) > len(existing_name):
                        seen[key] = (name, level, original)
                    # Also prefer if it contains "degree" or "in" (more specific)
                    elif 'degree' in name_lower or ' in ' in name_lower:
                        if 'degree' not in existing_lower and ' in ' not in existing_lower:
                            seen[key] = (name, level, original)
                # If different level, prefer higher level
                elif level > existing_level:
                    seen[key] = (name, level, original)
            else:
                seen[key] = (name, level, original)
        
        # Additional pass: Remove generic matches if we have specific matches
        # e.g., remove "Bachelor" if we have "Bachelor's Degree in Civil Engineering"
        filtered_degrees = []
        generic_keys = set()
        specific_keys = set()
        
        for name, level, orig in seen.values():
            name_lower = name.lower().strip()
            # Check if it's a generic match (exact match only)
            if name_lower in ['bachelor', 'bachelors', 'master', 'masters', 'phd', 'doctorate', 'diploma']:
                generic_keys.add((name_lower, level))
            else:
                # For specific matches, use a broader key to catch variations
                # e.g., "bachelor's degree in civil engineering" -> "bachelor"
                base_key = name_lower.split()[0] if name_lower.split() else name_lower[:10]
                specific_keys.add((base_key, level))
        
        # Only add generic matches if no specific match exists for that level
        # CRITICAL: Always prefer specific matches over generic ones
        for name, level, orig in seen.values():
            name_lower = name.lower().strip()
            is_generic = name_lower in ['bachelor', 'bachelors', 'master', 'masters', 'phd', 'doctorate', 'diploma']
            
            if is_generic:
                # Check if there's a specific match at the same level that starts with the same base word
                base_key = name_lower  # For "bachelor", base_key is "bachelor"
                has_specific = any(
                    level == spec_level and (spec_key == base_key or spec_key.startswith(base_key))
                    for spec_key, spec_level in specific_keys
                )
                if not has_specific:
                    filtered_degrees.append((name, level, orig))
                else:
                    # Log that we're filtering out a generic match in favor of a specific one
                    logger.debug(f"Filtering out generic '{name}' in favor of specific match at level {level}")
            else:
                filtered_degrees.append((name, level, orig))
        
        # Final check: if we still have both generic and specific, remove generic
        final_filtered = []
        generic_names = {name.lower().strip() for name, _, _ in filtered_degrees 
                        if name.lower().strip() in ['bachelor', 'bachelors', 'master', 'masters', 'phd', 'doctorate', 'diploma']}
        specific_names = {name.lower().strip() for name, _, _ in filtered_degrees 
                         if name.lower().strip() not in ['bachelor', 'bachelors', 'master', 'masters', 'phd', 'doctorate', 'diploma']}
        
        for name, level, orig in filtered_degrees:
            name_lower = name.lower().strip()
            # If it's generic and we have any specific match at the same level, skip it
            if name_lower in generic_names and specific_names:
                # Check if any specific name starts with the same base word
                base_word = name_lower.split()[0] if name_lower.split() else name_lower
                has_matching_specific = any(spec_name.startswith(base_word) for spec_name in specific_names)
                if has_matching_specific:
                    logger.debug(f"Final filter: Removing generic '{name}' because specific match exists")
                    continue
            final_filtered.append((name, level, orig))
        
        return final_filtered if final_filtered else [(name, level, orig) for name, level, orig in seen.values()]
    
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
