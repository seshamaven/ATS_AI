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
    


".NET Developer", ".NET Application Developer", ".NET Web Developer",
    ".NET Programmer", "C# Developer", "C#.NET Developer", "ASP.NET Developer",
    "Senior .NET Developer", "Sr. .NET Developer", "Lead .NET Developer",
    "Full Stack .NET Developer", ".NET Full Stack Developer", ".NET Core Developer",
    "WPF Developer", "WinForms Developer", ".NET API Developer",
    ".NET MVC Developer", "ASP.NET Core Developer", ".NET Backend Developer",
    ".NET Cloud Developer", ".NET Microservices Developer", ".NET Integration Developer",
    ".NET Support Engineer", ".NET Consultant", ".NET Automation Developer",
    ".NET SQL Developer", ".NET Azure Developer", ".NET Architect",
    "C# Software Engineer", "ASP.NET Webforms Developer",
	 "Java Developer", "Java Web Developer", "Java Application Developer",
    "J2EE Developer", "Java Programmer", "Java Software Engineer",
    "Senior Java Developer", "Sr. Java Developer", "Lead Java Developer",
    "Full Stack Java Developer", "Java Full Stack Developer", "Core Java Developer",
    "Java Spring Boot Developer", "Java Microservices Developer", "Java API Developer",
    "Java Cloud Developer", "Java Backend Developer", "Java Integration Developer",
    "Java Architect", "JSP Developer", "Java Hibernate Developer", "Java Kafka Developer",
    "Java AWS Developer",
	 "Python Developer", "Python Web Developer", "Python Application Developer",
    "Python Engineer", "Senior Python Developer", "Sr. Python Developer",
    "Full Stack Python Developer", "Python Full Stack Developer", "Django Developer",
    "Flask Developer", "FastAPI Developer", "Python Backend Developer",
    "Python Automation Engineer", "Python Data Engineer", "Python AI Engineer",
    "Python ML Engineer", "Python API Developer", "Python Scripting Engineer",
    "Python Cloud Developer",
	"Frontend Developer", "Front End Developer", "UI Developer", "UX Developer",
    "React Developer", "Angular Developer", "Vue Developer", "JavaScript Developer",
    "Node.js Developer", "Node Developer", "ReactJS Developer", "AngularJS Developer",
    "Next.js Developer", "Nuxt Developer", "Typescript Developer", "HTML Developer",
    "CSS Developer", "UI Engineer", "UI Architect", "Frontend Architect",
    "Frontend Lead", "Web UI Developer", "Web Designer", "UI/UX Developer",
    "UI/UX Engineer", "Mobile UI/UX Designer", "Interaction Designer", "Web Accessibility Engineer",
	"Full Stack Developer", "Full Stack Engineer", "FullStack Developer",
    "Full Stack Web Developer", "Full Stack Software Engineer",
    "Senior Full Stack Developer", "Sr. Full Stack Developer",
    "Java Full Stack Engineer", ".NET Full Stack Engineer", "Python Full Stack Engineer",
    "MERN Stack Developer", "MEAN Stack Developer", "LAMP Stack Developer",
    "PERL Stack Developer",
	"Backend Developer", "Back End Developer", "Backend Engineer",
    "Backend Software Engineer", "API Developer", "Web Services Developer",
    "Integration Developer", "Server Side Developer", "API Integration Engineer",
    "Backend Architect", "Backend Lead", "Platform Engineer", "Microservices Architect",
    "Serverless Backend Developer", "GraphQL Backend Developer",
	"Android Developer", "iOS Developer", "Mobile Developer",
    "Android Software Engineer", "iOS Software Engineer", "Mobile Application Developer",
    "React Native Developer", "Flutter Developer", "Kotlin Developer",
    "Swift Developer", "Mobile Automation Engineer", "Mobile Architect",
	"DevOps Engineer", "DevOps Developer", "DevOps Release Manager",
    "Site Reliability Engineer", "SRE", "Build Engineer", "Release Engineer",
    "Cloud Engineer", "AWS Engineer", "Azure Engineer", "GCP Engineer",
    "Cloud DevOps Engineer", "Cloud Architect", "AWS Architect",
    "Kubernetes Engineer", "Docker Engineer", "Terraform Engineer",
    "CI/CD Engineer", "Platform Reliability Engineer", "Cloud Security Engineer",
	"Data Engineer", "Data Scientist", "Big Data Engineer", "ETL Developer",
    "Data Analyst", "Business Intelligence Developer", "BI Developer",
    "Data Warehouse Developer", "Machine Learning Engineer", "ML Engineer",
    "AI Engineer", "Deep Learning Engineer", "NLP Engineer",
    "MLOps Engineer", "Data Visualization Engineer", "Tableau Developer",
    "Power BI Developer", "Hadoop Developer", "Spark Developer",
    "Snowflake Developer", "Data Governance Analyst", "Predictive Analytics Engineer",
    "Computer Vision Engineer", "AI Research Engineer",
	"QA Engineer", "QA Tester", "QA Analyst", "Test Engineer",
    "Test Automation Engineer", "Quality Assurance Engineer",
    "QA Automation Engineer", "Performance Test Engineer",
    "Mobile Test Engineer", "Security Testing Engineer",
    "Manual Tester", "Functional Tester", "SDET", "SDET Lead",
    "Automation QA Engineer", "Performance Test Analyst",
	"Database Administrator", "DBA", "SQL Developer", "Database Developer",
    "Oracle DBA", "SQL Server DBA", "MySQL DBA", "PostgreSQL Developer",
    "MongoDB Developer", "NoSQL Developer", "Database Analyst", "Data Modeler",
    "ETL Tester", "Informatica Developer", "Data Migration Specialist",
    "Database Engineer", "Data Warehouse Engineer", "Database Architect",
    "Big Data Analyst", "Teradata Developer", "Redshift Developer",
	"Systems Engineer", "Systems Administrator", "Network Engineer",
    "Network Administrator", "Linux Administrator", "Windows Administrator",
    "Cloud Administrator", "Infrastructure Engineer", "IT Support Engineer",
    "Systems Analyst", "IT Operations Engineer", "Network Security Engineer",
    "Firewall Engineer", "Virtualization Engineer", "Active Directory Engineer",
    "Server Engineer", "Storage Engineer", "VMware Engineer", "IT Infrastructure Architect",
    "Platform Engineer", "Site Reliability Engineer (SRE)",
	"Project Manager", "Program Manager", "IT Project Manager",
    "Technical Project Manager", "Senior Project Manager", "Project Coordinator",
    "Scrum Master", "Agile Coach", "Product Manager", "Product Owner",
    "Portfolio Manager", "PMO Analyst", "Program Coordinator", "Delivery Manager",
    "Project Lead", "Technical Program Manager", "Project Scheduler",
    "Resource Manager", "Release Manager", "Change Manager",
	"Security Engineer", "Information Security Analyst", "Cybersecurity Engineer",
    "SOC Analyst", "SIEM Engineer", "Penetration Tester", "Ethical Hacker",
    "Security Consultant", "Identity Access Engineer", "Application Security Engineer",
    "Network Security Analyst", "Threat Analyst", "Vulnerability Analyst",
    "Cloud Security Engineer", "Security Architect", "Red Team Engineer",
    "Blue Team Engineer", "Data Privacy Officer", "Compliance Analyst", "IAM Specialist",
	"Salesforce Developer", "SharePoint Developer", "SAP Developer",
    "Oracle Developer", "Mainframe Developer", "COBOL Developer",
    "Embedded Engineer", "UI/UX Designer", "Technical Writer",
    "Robotics Engineer", "RPA Developer", "BluePrism Developer",
    "Automation Anywhere Developer", "Chatbot Developer",
    "Blockchain Developer", "IoT Engineer", "AR/VR Developer",
    "Game Developer", "Unity Developer", "Unreal Developer",
    "Graphics Programmer", "Video Processing Engineer", "Digital Twin Engineer",
    "Quantum Computing Researcher", "AI/ML Research Scientist", "Augmented Reality Developer",
	"Help Desk Support", "IT Support", "Technical Support", "Desktop Support",
    "Application Support", "Production Support", "L2 Support Engineer",
    "L3 Support Engineer", "Customer Support Engineer", "IT Service Desk Analyst",
    "Incident Manager", "Problem Manager", "Operations Support Engineer",
    "End User Support", "Technical Account Manager", "Field Support Engineer",
	"Junior Developer", "Junior Software Engineer", "Associate Developer",
    "Entry Level Developer", "Graduate Engineer", "Intern",
    "Trainee Developer", "Apprentice Software Engineer", "Junior QA Engineer",
    "Junior Data Analyst", "Junior System Administrator", "Junior Network Engineer",
    "Internship Developer", "Associate Software Engineer", "Junior Mobile Developer",
	 "Software Architect", "Solutions Architect", "Enterprise Architect",
    "Technical Architect", "System Architect", "Data Architect",
    "Integration Architect", "Cloud Architect", "Security Architect",
    "Application Architect", "Infrastructure Architect", "Platform Architect",
    "DevOps Architect", "AI Architect", "Machine Learning Architect",
    "Solution Design Lead", "Enterprise Solution Consultant", "Technical Solution Lead",
	"RPA Developer", "BluePrism Developer", "Automation Anywhere Developer",
    "UiPath Developer", "AI Engineer", "Machine Learning Engineer",
    "Deep Learning Engineer", "Computer Vision Engineer", "NLP Engineer",
    "Chatbot Developer", "Robotic Process Automation Consultant",
    "AI Research Scientist", "Data Scientist", "MLOps Engineer", "Python Automation Engineer",
    "Intelligent Automation Engineer", "AI Solutions Architect",
	"Technical Consultant", "Business Consultant", "IT Consultant",
    "Specialist", "Technical Specialist", "Subject Matter Expert", "SME",
    "Business Analyst", "Functional Analyst", "Process Analyst",
    "Data Governance Analyst", "Product Analyst", "Technical Trainer",
    "IT Trainer", "Documentation Specialist", "Technical Writer",
    "Innovation Engineer", "Cloud Evangelist", "DevSecOps Engineer",
    "Full Stack Consultant", "Technical Program Manager",
    "AI Research Engineer", "AI Solutions Architect", "ML Research Scientist",
    "Deep Learning Researcher", "NLP Solutions Engineer", "Computer Vision Specialist",
    "Data Analytics Engineer", "Big Data Architect", "Hadoop Solutions Architect",
    "Spark Solutions Engineer", "Data Pipeline Engineer", "Data Quality Analyst",
    "Data Ops Engineer", "Business Data Analyst", "BI Solutions Architect",
    "Power BI Specialist", "Tableau Specialist", "ETL Architect", "SQL Solutions Engineer",
    "NoSQL Solutions Engineer", "MongoDB Administrator", "PostgreSQL Administrator",
    "Database Performance Engineer", "Database Migration Specialist", "Database Security Engineer",
    "Cybersecurity Analyst", "SOC Manager", "SIEM Consultant", "Penetration Testing Specialist",
    "Ethical Hacking Consultant", "Identity Management Engineer", "IAM Consultant",
    "Network Security Specialist", "Cloud Security Consultant", "DevSecOps Consultant",
    "Cloud Security Architect", "Platform Security Engineer", "Site Reliability Lead",
    "Systems Reliability Engineer", "Infrastructure Reliability Specialist",
    "Linux Systems Engineer", "Windows Systems Engineer", "VMware Engineer",
    "Network Administrator", "Network Operations Engineer", "Network Architect",
    "IT Operations Engineer", "IT Support Specialist", "Desktop Support Analyst",
    "Application Support Specialist", "Production Support Analyst", "Release Manager",
    "Build & Release Engineer", "Automation QA Engineer", "Performance Testing Analyst",
    "Functional QA Engineer", "Mobile QA Engineer", "Security QA Engineer",
    "Project Delivery Manager", "Program Delivery Manager", "IT Program Manager",
    "Scrum Product Owner", "Agile Product Manager", "Technical Project Lead",
    "Project Coordinator", "Portfolio Manager", "Change Manager", "Business Project Manager",
    "Technical Architect Lead", "Enterprise Solutions Architect", "Software Solutions Architect",
    "Integration Solutions Architect", "Cloud Solutions Engineer", "Security Solutions Architect",
    "DevOps Lead", "CI/CD Architect", "Kubernetes Lead Engineer", "Docker Lead Engineer",
    "Terraform Consultant", "AWS Solutions Architect", "Azure Solutions Architect",
    "GCP Solutions Architect", "Cloud Platform Engineer", "Platform Automation Engineer",
    "Automation Anywhere Specialist", "Blue Prism Lead Developer", "RPA Solutions Engineer",
    "Robotics Process Automation Engineer", "Chatbot Solutions Engineer", "AI Chatbot Developer",
    "Blockchain Solutions Engineer", "IoT Solutions Architect", "IoT Embedded Developer",
    "AR Developer", "VR Developer", "Mixed Reality Developer", "Game Engine Developer",
    "Unity 3D Developer", "Unreal Engine Developer", "Graphics Engine Programmer",
    "Video Processing Developer", "Embedded Firmware Engineer", "Hardware Design Engineer",
    "FPGA Developer", "DSP Engineer", "IoT Embedded Firmware Engineer",
    "C++ Software Developer", "C++ Application Engineer", "C++ Systems Programmer",
    "C# Software Engineer", "C# Application Developer", ".NET Core Engineer",
    "ASP.NET Developer", "WPF Software Developer", "WinForms Developer",
    "Full Stack .NET Engineer", "Java Spring Developer", "Java Microservices Engineer",
    "J2EE Application Developer", "Java API Engineer", "Java Cloud Engineer",
    "Java Backend Developer", "Java Integration Engineer", "Java Architect",
    "Python Backend Engineer", "Python Data Scientist", "Python AI Developer",
    "Python ML Engineer", "Django Backend Developer", "Flask Web Developer",
    "FastAPI Developer", "Python API Engineer", "Python Cloud Engineer",
    "Python Automation Developer", "Frontend Web Developer", "UI/UX Frontend Engineer",
    "ReactJS Frontend Developer", "Angular Frontend Developer", "Vue.js Frontend Developer",
    "Next.js Developer", "Nuxt.js Developer", "Typescript Frontend Developer",
    "HTML5 Developer", "CSS3 Developer", "UI Designer", "UX Researcher",
    "Web UI Engineer", "Frontend Architect", "Frontend Lead", "Full Stack Web Developer",
    "MERN Stack Engineer", "MEAN Stack Engineer", "LAMP Stack Developer",
    "Backend API Developer", "Server-Side Developer", "Platform Backend Engineer",
    "Android Application Developer", "iOS Application Developer", "Mobile App Engineer",
    "React Native Mobile Developer", "Flutter Mobile Developer", "Kotlin Mobile Developer",
    "Swift iOS Developer", "Mobile Solutions Architect", "Mobile Automation Engineer",
    "DevOps Cloud Engineer", "CI/CD Automation Engineer", "Infrastructure DevOps Engineer",
    "Platform Automation Consultant", "Cloud Reliability Engineer", "SRE Lead Engineer",
    "Data Platform Engineer", "Data Warehouse Architect", "ETL Solutions Architect",
    "BI Data Analyst", "Data Integration Engineer", "Data Governance Specialist",
    "AI/ML Solutions Engineer", "Deep Learning Architect", "NLP Solutions Architect",
    "Computer Vision Engineer", "Analytics Solutions Engineer", "Data Science Consultant",
    "Machine Learning Consultant", "Data Science Architect", "Data Visualization Specialist",
    "Power BI Consultant", "Tableau Consultant", "ETL Solutions Developer", "SQL Developer Lead",
    "NoSQL Solutions Consultant", "Database Migration Engineer", "Database Security Consultant",
    "Cloud Security Engineer", "Identity Access Consultant", "IAM Solutions Engineer",
    "SOC Solutions Engineer", "SIEM Solutions Architect", "Penetration Test Lead",
    "Ethical Hacker Consultant", "Network Security Consultant", "DevSecOps Lead Engineer",
    "Cybersecurity Solutions Architect", "Information Security Consultant",
    "Help Desk Analyst", "IT Support Analyst", "Desktop Support Lead",
    "Application Support Lead", "Production Support Lead", "Technical Support Lead",
    "L2 Technical Support Engineer", "L3 Technical Support Engineer", "QA Automation Lead",
    "Performance Testing Lead", "Functional Testing Lead", "Mobile QA Lead",
    "Security QA Lead", "Manual QA Tester", "Automation QA Consultant",
    "Project Delivery Lead", "Program Delivery Lead", "Scrum Master Lead",
    "Agile Coach Lead", "Technical Project Manager Lead", "Portfolio Manager Lead",
    "Change Management Lead", "Business Project Manager Lead", "Software Architect Lead",
    "Solutions Architect Lead", "Enterprise Architect Lead", "Integration Architect Lead",
    "Cloud Architect Lead", "Security Architect Lead", "Technical Lead",
    "Team Lead Software Development", "Engineering Manager", "Development Manager",
    "IT Manager", "Technical Manager", "Software Manager", "Platform Manager",
    "Product Engineering Manager", "Engineering Director", "Technical Director",
    "IT Director", "Program Director", "Project Director", "Chief Technology Officer",
    "CTO", "VP Engineering", "VP Technology", "Chief Information Officer", "CIO",
    "Technical Consultant", "Business Consultant", "IT Consultant",
    "Specialist", "Technical Specialist", "Subject Matter Expert", "SME",
    "Business Analyst", "Functional Analyst", "Process Analyst",
    "Data Governance Analyst", "Product Analyst", "Technical Trainer",
    "IT Trainer", "Documentation Specialist", "Technical Writer",
    "Innovation Engineer", "Cloud Evangelist", "DevSecOps Engineer",
    "Full Stack Consultant", "Technical Program Manager", "Solution Advisor",
    "Enterprise Consultant", "IT Strategy Consultant", "Automation Consultant",
    "Business Solutions Specialist", "Data Solutions Specialist", "Security Consultant",
    "Network Consultant", "Cloud Solutions Architect", "DevOps Consultant",
    "AI Engineer", "Machine Learning Engineer", "Deep Learning Engineer",
    "NLP Engineer", "Computer Vision Engineer", "Data Visualization Engineer",
    "Analytics Engineer", "Big Data Engineer", "Hadoop Engineer",
    "Spark Developer", "Data Warehouse Developer", "BI Developer",
    "Power BI Developer", "Tableau Developer", "ETL Developer",
    "SQL Developer", "NoSQL Developer", "Database Developer",
    "Database Administrator", "Oracle DBA", "SQL Server DBA",
    "MySQL DBA", "PostgreSQL Developer", "MongoDB Developer",
    "Business Intelligence Developer", "ETL Tester", "Data Modeler",
    "Cybersecurity Engineer", "SOC Analyst", "Penetration Tester",
    "Ethical Hacker", "Security Analyst", "SIEM Engineer",
    "Identity Access Engineer", "IAM Engineer", "Information Security Analyst",
    "Network Security Engineer", "Cloud Security Engineer", "DevSecOps Engineer",
    "Site Reliability Engineer", "SRE", "Platform Reliability Engineer",
    "System Administrator", "Linux Administrator", "Windows Administrator",
    "Cloud Administrator", "Infrastructure Engineer", "IT Support Engineer",
    "Help Desk Support", "Technical Support", "Desktop Support",
    "Application Support", "Production Support", "L2 Support Engineer",
    "L3 Support Engineer", "Release Engineer", "Build Engineer",
    "Automation Engineer", "QA Engineer", "QA Tester", "QA Analyst",
    "Test Automation Engineer", "Performance Test Engineer", "Manual Tester",
    "Functional Tester", "Mobile Test Engineer", "Security Testing Engineer",
    "Project Manager", "Program Manager", "IT Project Manager",
    "Technical Project Manager", "Scrum Master", "Agile Coach",
    "Product Manager", "Product Owner", "Technical Architect",
    "Software Architect", "Solutions Architect", "Enterprise Architect",
    "Integration Architect", "Cloud Architect", "Security Architect",
    "DevOps Engineer", "DevOps Developer", "CI/CD Engineer",
    "Kubernetes Engineer", "Docker Engineer", "Terraform Engineer",
    "AWS Engineer", "Azure Engineer", "GCP Engineer", "Cloud Engineer",
    "Cloud DevOps Engineer", "Platform Engineer", "Automation Anywhere Developer",
    "BluePrism Developer", "RPA Developer", "Robotics Engineer", "Chatbot Developer",
    "Blockchain Developer", "IoT Engineer", "AR/VR Developer", "Game Developer",
    "Unity Developer", "Unreal Developer", "Graphics Programmer",
    "Video Processing Engineer", "Embedded Engineer", "Firmware Engineer",
    "Embedded Software Engineer", "Embedded Systems Developer", "Hardware Engineer",
    "C++ Developer", "C# Developer", "Java Developer", "Python Developer",
    "Frontend Developer", "UI Developer", "UX Designer", "React Developer",
    "Angular Developer", "Node.js Developer", "Vue Developer", "Full Stack Developer",
    "Backend Developer", "Mobile Developer", "iOS Developer", "Android Developer",
    "React Native Developer", "Flutter Developer", "Swift Developer", "Kotlin Developer"


	
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
