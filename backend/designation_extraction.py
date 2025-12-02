"""
Designation Extraction Module (No AI/LLM)
=========================================

Extracts current/most recent job title/designation from resume text using:
- Comprehensive known designation database (180k+ profiles)
- Regex pattern matching
- Experience section parsing
- Job title validation with fuzzy matching

Author: ATS System
"""

import re
import logging
from typing import Optional, List, Set, Dict, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# ============================================================================
# KNOWN DESIGNATIONS DATABASE
# ============================================================================
# Comprehensive list of real-world job designations extracted from 180k+ profiles

KNOWN_DESIGNATIONS = {
    # .NET / C# Developers
    ".NET Developer", ".NET Application Developer", ".NET Web Developer", 
    ".NET Programmer", "C# Developer", "C#.NET Developer", "ASP.NET Developer",
    "Senior .NET Developer", "Sr. .NET Developer", "Lead .NET Developer",
    "Full Stack .NET Developer", ".NET Full Stack Developer",
    
    # Java Developers
    "Java Developer", "Java Web Developer", "Java Application Developer",
    "J2EE Developer", "Java Programmer", "Java Software Engineer",
    "Senior Java Developer", "Sr. Java Developer", "Lead Java Developer",
    "Full Stack Java Developer", "Java Full Stack Developer",
    
    # Python Developers
    "Python Developer", "Python Web Developer", "Python Application Developer",
    "Python Engineer", "Senior Python Developer", "Sr. Python Developer",
    "Full Stack Python Developer", "Python Full Stack Developer",
    
    # Frontend Developers
    "Frontend Developer", "Front End Developer", "UI Developer", "UX Developer",
    "React Developer", "Angular Developer", "Vue Developer", "JavaScript Developer",
    "Node.js Developer", "Node Developer", "ReactJS Developer", "AngularJS Developer",
    
    # Full Stack Developers
    "Full Stack Developer", "Full Stack Engineer", "FullStack Developer",
    "Full Stack Web Developer", "Full Stack Software Engineer",
    "Senior Full Stack Developer", "Sr. Full Stack Developer",
    
    # Backend Developers
    "Backend Developer", "Back End Developer", "Backend Engineer",
    "Backend Software Engineer", "API Developer", "Web Services Developer",
    
    # Mobile Developers
    "Android Developer", "iOS Developer", "Mobile Developer",
    "Android Software Engineer", "iOS Software Engineer", "Mobile Application Developer",
    "React Native Developer", "Flutter Developer",
    
    # DevOps Engineers
    "DevOps Engineer", "DevOps Developer", "DevOps Release Manager",
    "Site Reliability Engineer", "SRE", "Build Engineer", "Release Engineer",
    "Cloud Engineer", "AWS Engineer", "Azure Engineer", "GCP Engineer",
    
    # Data Engineers / Scientists
    "Data Engineer", "Data Scientist", "Big Data Engineer", "ETL Developer",
    "Data Analyst", "Business Intelligence Developer", "BI Developer",
    "Data Warehouse Developer", "Machine Learning Engineer", "ML Engineer",
    "Senior Data Engineer", "Sr. Data Engineer", "Lead Data Engineer",
    
    # Software Engineers (General)
    "Software Engineer", "Software Developer", "Software Programmer",
    "Senior Software Engineer", "Sr. Software Engineer", "Lead Software Engineer",
    "Principal Software Engineer", "Staff Software Engineer",
    
    # QA / Testing
    "QA Engineer", "QA Tester", "QA Analyst", "Test Engineer", "Test Automation Engineer",
    "Quality Assurance Engineer", "QA Automation Engineer", "Performance Test Engineer",
    "Senior QA Engineer", "Sr. QA Engineer", "QA Lead",
    
    # Database
    "Database Administrator", "DBA", "SQL Developer", "Database Developer",
    "Oracle DBA", "SQL Server DBA", "MySQL DBA", "Database Analyst",
    
    # System / Network
    "Systems Engineer", "Systems Administrator", "Network Engineer", "Network Administrator",
    "Linux Administrator", "Windows Administrator", "Cloud Administrator",
    "Infrastructure Engineer", "IT Support Engineer",
    
    # Business / Analyst Roles
    "Business Analyst", "Business Systems Analyst", "Data Analyst",
    "Systems Analyst", "IT Analyst", "Technical Analyst", "Functional Analyst",
    "Senior Business Analyst", "Sr. Business Analyst", "Lead Business Analyst",
    
    # Project / Program Management
    "Project Manager", "Program Manager", "IT Project Manager", "Technical Project Manager",
    "Senior Project Manager", "Sr. Project Manager", "Project Coordinator",
    "Scrum Master", "Product Manager", "Product Owner",
    
    # Architecture
    "Software Architect", "Solutions Architect", "Enterprise Architect",
    "Technical Architect", "System Architect", "Data Architect",
    "Senior Architect", "Sr. Architect", "Lead Architect",
    
    # Management
    "Engineering Manager", "Development Manager", "IT Manager",
    "Technical Manager", "Software Manager", "Project Manager",
    "Senior Manager", "Sr. Manager", "Director",
    
    # Specialized Roles
    "Security Engineer", "Information Security Analyst", "Cybersecurity Engineer",
    "Salesforce Developer", "SharePoint Developer", "SAP Developer",
    "Oracle Developer", "Mainframe Developer", "COBOL Developer",
    "Embedded Software Engineer", "Firmware Engineer", "Hardware Engineer",
    "UI/UX Designer", "Technical Writer", "Business Development Manager",
    
    # Consultants / Specialists
    "Technical Consultant", "Business Consultant", "IT Consultant",
    "Specialist", "Technical Specialist", "Subject Matter Expert", "SME",
    
    # Support Roles
    "Help Desk Support", "IT Support", "Technical Support", "Desktop Support",
    "Application Support", "Production Support",
    
    # Entry Level / Junior
    "Junior Developer", "Junior Software Engineer", "Associate Developer",
    "Entry Level Developer", "Graduate Engineer", "Intern",
}

def normalize_designation_for_lookup(designation: str) -> str:
    """Normalize designation for lookup (lowercase, remove special chars)."""
    # Convert to lowercase
    normalized = designation.lower()
    # Remove common variations
    normalized = re.sub(r'\b(sr\.?|senior)\b', 'senior', normalized)
    normalized = re.sub(r'\b(jr\.?|junior)\b', 'junior', normalized)
    normalized = re.sub(r'\b(\.net|dotnet)\b', 'net', normalized)
    normalized = re.sub(r'\b(asp\.net|aspnet)\b', 'aspnet', normalized)
    # Remove special characters but keep spaces
    normalized = re.sub(r'[^\w\s]', '', normalized)
    # Normalize whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

# Create normalized lookup (lowercase, remove special chars for matching)
DESIGNATION_LOOKUP: Dict[str, str] = {}
DESIGNATION_KEYWORDS: Set[str] = set()

# Initialize lookup
for designation in KNOWN_DESIGNATIONS:
    # Store original -> normalized mapping
    normalized = normalize_designation_for_lookup(designation)
    DESIGNATION_LOOKUP[normalized] = designation
    
    # Extract keywords for partial matching
    words = designation.lower().split()
    for word in words:
        if len(word) > 2:  # Ignore very short words
            DESIGNATION_KEYWORDS.add(word)

# ============================================================================
# COMMON JOB TITLE KEYWORDS
# ============================================================================

# Seniority levels
SENIORITY_KEYWORDS = [
    'senior', 'junior', 'lead', 'principal', 'chief', 'head', 'director',
    'manager', 'associate', 'assistant', 'executive', 'vp', 'vice president',
    'coordinator', 'specialist', 'analyst', 'consultant', 'architect',
    'engineer', 'developer', 'programmer', 'administrator', 'officer'
]

# Job title suffixes (common endings for job titles)
JOB_TITLE_SUFFIXES = [
    'engineer', 'developer', 'programmer', 'architect', 'analyst', 'consultant',
    'manager', 'director', 'lead', 'specialist', 'coordinator', 'administrator',
    'executive', 'officer', 'associate', 'assistant', 'representative',
    'designer', 'tester', 'qa', 'scientist', 'researcher', 'trainer',
    'supervisor', 'superintendent', 'technician', 'technologist'
]

# Invalid designations (section headers, common words that aren't titles)
INVALID_DESIGNATIONS = {
    'experience', 'work history', 'employment', 'career', 'objective',
    'summary', 'education', 'skills', 'certifications', 'projects',
    'references', 'contact', 'personal', 'details', 'qualifications',
    'achievements', 'awards', 'publications', 'presentations'
}

# Company name indicators (words that suggest it's a company, not a title)
COMPANY_INDICATORS = [
    'inc', 'llc', 'ltd', 'corp', 'corporation', 'pvt', 'limited', 'company',
    'solutions', 'technologies', 'systems', 'services', 'group', 'enterprises',
    'consulting', 'consultants', 'associates', 'partners'
]

# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for pattern matching."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text


def fuzzy_match_designation(candidate: str, threshold: float = 0.75) -> Optional[str]:
    """
    Try to match candidate designation against known designations using fuzzy matching.
    
    Args:
        candidate: The designation candidate to match
        threshold: Similarity threshold (0.0 to 1.0)
        
    Returns:
        Matched known designation or None
    """
    if not candidate:
        return None
    
    candidate_normalized = normalize_designation_for_lookup(candidate)
    
    # Exact match first
    if candidate_normalized in DESIGNATION_LOOKUP:
        return DESIGNATION_LOOKUP[candidate_normalized]
    
    # Fuzzy match
    best_match = None
    best_score = 0.0
    
    for known_normalized, known_original in DESIGNATION_LOOKUP.items():
        # Use SequenceMatcher for similarity
        similarity = SequenceMatcher(None, candidate_normalized, known_normalized).ratio()
        
        # Also check if candidate contains key words from known designation
        candidate_words = set(candidate_normalized.split())
        known_words = set(known_normalized.split())
        word_overlap = len(candidate_words & known_words) / max(len(candidate_words), 1)
        
        # Combined score
        combined_score = (similarity * 0.7) + (word_overlap * 0.3)
        
        if combined_score > best_score and combined_score >= threshold:
            best_score = combined_score
            best_match = known_original
    
    if best_match:
        logger.info(f"Fuzzy matched '{candidate}' -> '{best_match}' (score: {best_score:.2f})")
    
    return best_match


def is_valid_designation(designation: str) -> bool:
    """
    Validate if extracted text is a valid job designation.
    
    Args:
        designation: The extracted designation candidate
        
    Returns:
        True if valid, False if invalid
    """
    if not designation or not isinstance(designation, str):
        return False
    
    designation = designation.strip()
    if not designation or len(designation) < 2:
        return False
    
    designation_lower = designation.lower()
    
    # Reject invalid designations
    if designation_lower in INVALID_DESIGNATIONS:
        return False
    
    # Reject if it contains company indicators
    for indicator in COMPANY_INDICATORS:
        if indicator in designation_lower:
            return False
    
    # Reject if it's too long (likely a sentence, not a title)
    if len(designation) > 100:
        return False
    
    # Reject if it contains email or phone
    if '@' in designation or re.search(r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', designation):
        return False
    
    # Reject if it's a location pattern (City, State)
    if ',' in designation:
        parts = [p.strip() for p in designation.split(',')]
        if len(parts) == 2:
            # Check if it looks like a location
            if all(len(p.split()) <= 3 for p in parts):
                return False
    
    # Reject if it contains degree patterns
    degree_patterns = [
        r'\b([BM]\.?[AS]\.?|MBA|PhD|MD|JD|B\.?Tech|M\.?Tech)\b',
        r'\bin\s+[A-Z][a-z]+',
        r'degree|diploma|certificate'
    ]
    for pattern in degree_patterns:
        if re.search(pattern, designation, re.IGNORECASE):
            return False
    
    # Check against known designations (fuzzy match)
    matched = fuzzy_match_designation(designation, threshold=0.6)
    if matched:
        return True
    
    # Fallback: Should contain at least one job title keyword or suffix
    has_title_keyword = any(
        keyword in designation_lower for keyword in JOB_TITLE_SUFFIXES
    ) or any(
        keyword in designation_lower for keyword in SENIORITY_KEYWORDS
    )
    
    # If it's a short phrase (2-5 words) and looks like Title Case, accept it
    words = designation.split()
    if 2 <= len(words) <= 5:
        # Check if it's mostly Title Case (first letter of each word capitalized)
        title_case_count = sum(1 for word in words if word and word[0].isupper())
        if title_case_count >= len(words) * 0.7:  # At least 70% Title Case
            return True
    
    # If it has a job title keyword, accept it
    if has_title_keyword:
        return True
    
    return False


def extract_designation(text: str) -> Optional[str]:
    """
    Extract current/most recent job designation from resume text using pattern matching.
    Prioritizes known designations from the database.
    
    Args:
        text: Resume text to extract designation from
        
    Returns:
        Extracted designation string or None if not found
    """
    if not text or not isinstance(text, str):
        logger.warning("Invalid text input for designation extraction")
        return None
    
    text_lower = text.lower()
    
    # ========================================================================
    # Strategy 1: Look for explicit designation patterns in experience section
    # ========================================================================
    
    # Pattern 1: "Position:", "Role:", "Title:", "Designation:"
    explicit_patterns = [
        r'(?:position|role|title|designation|job\s+title)[:\s]+([A-Z][A-Za-z\s&.,-]+?)(?:\n|$|,|;|at\s+[A-Z])',
        r'(?:currently|presently|working\s+as|serving\s+as)[:\s]+([A-Z][A-Za-z\s&.,-]+?)(?:\n|$|,|;|at\s+[A-Z])',
    ]
    
    for pattern in explicit_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            candidate = match.strip()
            # Try to match against known designations first
            matched = fuzzy_match_designation(candidate, threshold=0.7)
            if matched:
                logger.info(f"Found designation via explicit pattern (matched): {matched}")
                return matched
            if is_valid_designation(candidate):
                logger.info(f"Found designation via explicit pattern: {candidate}")
                return candidate
    
    # ========================================================================
    # Strategy 2: Extract from Experience/Work History section
    # ========================================================================
    
    # Find experience section
    exp_section_patterns = [
        r'(?:experience|work\s+history|employment\s+history|professional\s+experience)(.*?)(?=\n\n[A-Z]|education|skills|certifications|projects|$)',
        r'(?:work\s+experience|employment|career)(.*?)(?=\n\n[A-Z]|education|skills|certifications|projects|$)',
    ]
    
    exp_text = None
    for pattern in exp_section_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            exp_text = match.group(1)
            break
    
    if not exp_text:
        # If no explicit experience section, use first 2000 chars (likely contains experience)
        exp_text = text[:2000]
    
    # ========================================================================
    # Strategy 3: Look for job title patterns in experience section
    # ========================================================================
    
    # Pattern: Company Name - Job Title (common format)
    # Example: "Infosys - Software Engineer"
    company_title_pattern = r'([A-Z][A-Za-z0-9\s&.,-]+?)\s*[-–—]\s*([A-Z][A-Za-z\s&.,-]+?)(?:\s+\d{4}|\s+[A-Z][a-z]+\s+\d{4}|$)'
    matches = re.findall(company_title_pattern, exp_text[:1000])  # Check first 1000 chars
    if matches:
        # Take the last match (most recent)
        for company, title in reversed(matches):
            title_candidate = title.strip()
            # Try known designations first
            matched = fuzzy_match_designation(title_candidate, threshold=0.7)
            if matched:
                logger.info(f"Found designation via company-title pattern (matched): {matched}")
                return matched
            if is_valid_designation(title_candidate):
                logger.info(f"Found designation via company-title pattern: {title_candidate}")
                return title_candidate
    
    # Pattern: Job Title at Company
    # Example: "Software Engineer at Microsoft"
    title_at_company_pattern = r'([A-Z][A-Za-z\s&.,-]+?)\s+at\s+([A-Z][A-Za-z0-9\s&.,-]+?)(?:\s+\d{4}|\s+[A-Z][a-z]+\s+\d{4}|$)'
    matches = re.findall(title_at_company_pattern, exp_text[:1000], re.IGNORECASE)
    if matches:
        for title, company in reversed(matches):
            title_candidate = title.strip()
            matched = fuzzy_match_designation(title_candidate, threshold=0.7)
            if matched:
                logger.info(f"Found designation via title-at-company pattern (matched): {matched}")
                return matched
            if is_valid_designation(title_candidate):
                logger.info(f"Found designation via title-at-company pattern: {title_candidate}")
                return title_candidate
    
    # Pattern: Search for known designations directly in text
    # Look in first 1000 chars of experience section (most recent job)
    recent_exp = exp_text[:1000] if exp_text else text[:1000]
    
    # Try to find known designations in the text
    for known_designation in sorted(KNOWN_DESIGNATIONS, key=len, reverse=True):  # Longest first
        # Create pattern that handles variations
        pattern = re.escape(known_designation)
        # Make it case-insensitive and allow word boundaries
        pattern = r'\b' + pattern.replace(r'\.', r'\.?') + r'\b'
        
        if re.search(pattern, recent_exp, re.IGNORECASE):
            logger.info(f"Found known designation in text: {known_designation}")
            return known_designation
    
    # Pattern: Standalone job titles (Title Case, 2-5 words, contains job keywords)
    # Example: "Senior Software Engineer", "Project Manager"
    standalone_title_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b'
    
    # Find all Title Case phrases
    title_candidates = re.findall(standalone_title_pattern, recent_exp)
    
    # Score candidates based on job title keywords and known designations
    scored_titles = []
    for candidate in title_candidates:
        candidate_lower = candidate.lower()
        score = 0
        
        # Check against known designations (highest priority)
        matched = fuzzy_match_designation(candidate, threshold=0.6)
        if matched:
            score += 100  # Very high score for known designations
            candidate = matched  # Use the matched version
        
        # Check for job title suffixes
        for suffix in JOB_TITLE_SUFFIXES:
            if suffix in candidate_lower:
                score += 10
                break
        
        # Check for seniority keywords
        for seniority in SENIORITY_KEYWORDS:
            if seniority in candidate_lower:
                score += 5
                break
        
        # Prefer 2-4 word titles
        word_count = len(candidate.split())
        if 2 <= word_count <= 4:
            score += 3
        elif word_count == 1:
            score -= 5  # Single words are less likely to be full titles
        
        if score > 0 and is_valid_designation(candidate):
            scored_titles.append((candidate, score))
    
    # Return highest scored title
    if scored_titles:
        scored_titles.sort(key=lambda x: x[1], reverse=True)
        best_title = scored_titles[0][0]
        logger.info(f"Found designation via standalone pattern: {best_title}")
        return best_title
    
    # ========================================================================
    # Strategy 4: Look in objective/summary section for current role
    # ========================================================================
    
    # Check first 500 chars (objective/summary area)
    early_text = text[:500]
    
    # Pattern: "Currently working as X", "Present role: X"
    current_role_patterns = [
        r'(?:currently|presently).*?(?:working\s+as|serving\s+as|role\s+of|position\s+of)[:\s]+([A-Z][A-Za-z\s&.,-]+?)(?:\n|$|,|;|at\s+[A-Z])',
        r'(?:current|present)\s+(?:role|position|title|designation)[:\s]+([A-Z][A-Za-z\s&.,-]+?)(?:\n|$|,|;|at\s+[A-Z])',
    ]
    
    for pattern in current_role_patterns:
        matches = re.findall(pattern, early_text, re.IGNORECASE)
        for match in matches:
            candidate = match.strip()
            matched = fuzzy_match_designation(candidate, threshold=0.7)
            if matched:
                logger.info(f"Found designation in objective/summary (matched): {matched}")
                return matched
            if is_valid_designation(candidate):
                logger.info(f"Found designation in objective/summary: {candidate}")
                return candidate
    
    logger.warning("No valid designation found in resume text")
    return None


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    # Example usage
    sample_resume = """
    John Doe
    Software Engineer
    
    EXPERIENCE:
    Infosys - Senior Software Engineer
    Jan 2020 - Present
    
    Microsoft - Software Developer
    Jan 2018 - Dec 2019
    """
    
    designation = extract_designation(sample_resume)
    print(f"Extracted designation: {designation}")
    # Expected: "Senior Software Engineer"
