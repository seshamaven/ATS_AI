"""
Domain Extraction Module (No AI/LLM)
=====================================

Extracts professional domains/industries from resume text using only deterministic Python logic:
- Keyword pattern matching
- Predefined domain keyword dictionaries
- Rule-based extraction
- Exact matching only (no inference)

Author: ATS System
"""

import re
import logging
from typing import List, Set, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# DOMAIN LIST
# ============================================================================
DOMAINS_LIST = [
    "Information Technology", "Software Development", "Cloud Computing", "Cybersecurity", 
    "Data Science", "Blockchain", "Internet of Things", "Banking", "Finance", "Insurance", 
    "FinTech", "Healthcare", "Pharmaceuticals", "Biotechnology", "Manufacturing", "Automotive", 
    "Energy", "Construction", "Retail", "E-commerce", "Logistics", "Telecommunications",
    "Media & Entertainment", "Advertising & Marketing", "Business Development", "Education Technology", 
    "Public Sector", "Real Estate", "Hospitality", "Travel & Tourism", "Agriculture", 
    "Legal & Compliance", "Human Resources", "Environmental & Sustainability"
]

# ============================================================================
# DOMAIN KEYWORD DICTIONARIES
# ============================================================================

# Information Technology & Software Development
# NOTE:
# - We intentionally treat "Information Technology" and "Software Development" as the same
#   underlying signal by sharing this IT_KEYWORDS dictionary.
# - Later in the pipeline we will de-duplicate them so only one appears in the final result.
IT_KEYWORDS = {
    'information technology', 'it', 'software development', 'software engineer', 'programming',
    'developer', 'coding', 'application development', 'system development', 'web development',
    'mobile development', 'software architecture', 'software design', 'code', 'programming language',
    'python', 'java', 'javascript', 'c++', 'c#', 'csharp', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin',
    'react', 'angular', 'vue', 'nodejs', 'django', 'flask', 'spring', 'dotnet', '.net', 'asp.net',
    'asp.net core', 'asp.net mvc', '.net core', '.net developer', 'full stack', 'full stack developer',
    'software engineering', 'software solutions', 'tech', 'technology', 'it services',
    'software company', 'tech company', 'development team', 'devops', 'sdlc', 'agile', 'scrum',
    'backend developer', 'frontend developer', 'web developer', 'application developer'
}

# Cloud Computing
CLOUD_KEYWORDS = {
    'cloud computing', 'cloud', 'aws', 'amazon web services', 'azure', 'microsoft azure',
    'gcp', 'google cloud', 'cloud platform', 'cloud services', 'cloud infrastructure',
    'saas', 'paas', 'iaas', 'cloud migration', 'cloud architecture', 'multi-cloud',
    'hybrid cloud', 'cloud security', 'cloud deployment', 'serverless', 'lambda',
    'ec2', 's3', 'rds', 'azure functions', 'cloud run', 'kubernetes', 'docker',
    'containerization', 'microservices', 'api gateway', 'cloudformation', 'terraform'
}

# Cybersecurity
CYBERSECURITY_KEYWORDS = {
    'cybersecurity', 'cyber security', 'information security', 'network security',
    'security', 'penetration testing', 'ethical hacking', 'vulnerability assessment',
    'security audit', 'firewall', 'intrusion detection', 'siem', 'soc', 'security operations',
    'data protection', 'encryption', 'ssl', 'tls', 'vpn', 'identity access management',
    'iam', 'oauth', 'jwt', 'security compliance', 'gdpr', 'hipaa', 'pci dss',
    'security analyst', 'security engineer', 'threat intelligence', 'malware analysis'
}

# Data Science
# Note: Removed standalone 'data', 'sql', 'nosql' to avoid false matches with "database"
DATA_SCIENCE_KEYWORDS = {
    'data science', 'data scientist', 'data analysis', 'data analytics', 'data engineer',
    'machine learning', 'ml', 'artificial intelligence', 'ai', 'deep learning', 'neural network',
    'data mining', 'predictive analytics', 'statistical analysis', 'big data', 'hadoop',
    'spark', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
    'data visualization', 'tableau', 'power bi', 'business intelligence', 'bi',
    'data warehouse', 'etl', 'data pipeline', 'data modeling', 'data lake', 
    'data governance', 'data quality', 'feature engineering',
    'machine learning engineer', 'ml engineer', 'data analyst'
}

# Blockchain
BLOCKCHAIN_KEYWORDS = {
    'blockchain', 'bitcoin', 'ethereum', 'cryptocurrency', 'crypto', 'smart contract',
    'solidity', 'defi', 'decentralized finance', 'nft', 'web3', 'dapp', 'distributed ledger',
    'consensus algorithm', 'mining', 'wallet', 'token', 'ico', 'crypto exchange',
    'hyperledger', 'corda', 'blockchain developer', 'blockchain engineer'
}

# Internet of Things
IOT_KEYWORDS = {
    'iot', 'internet of things', 'embedded systems', 'arduino', 'raspberry pi',
    'sensors', 'actuators', 'smart devices', 'connected devices', 'm2m', 'machine to machine',
    'edge computing', 'industrial iot', 'iiot', 'wearables', 'smart home', 'smart city'
}

# Banking
BANKING_KEYWORDS = {
    'banking', 'bank', 'financial institution', 'retail banking', 'corporate banking',
    'investment banking', 'commercial bank', 'private bank', 'central bank',
    'loan', 'mortgage', 'credit', 'debit', 'accounting', 'teller', 'branch manager',
    'banking operations', 'core banking', 'finacle', 'flexcube', 't24', 'banking software',
    'payment processing', 'transaction processing', 'kbc', 'kyc', 'aml', 'compliance'
}

# Finance
FINANCE_KEYWORDS = {
    'finance', 'financial', 'financial services', 'financial analyst', 'financial planning',
    'accounting', 'accountant', 'cpa', 'cfa', 'financial reporting', 'financial modeling',
    'portfolio management', 'asset management', 'wealth management', 'risk management',
    'treasury', 'audit', 'tax', 'taxation', 'financial controller', 'cfo', 'finance manager',
    'budgeting', 'forecasting', 'financial analysis', 'valuation', 'm&a', 'mergers acquisitions'
}

# Insurance
INSURANCE_KEYWORDS = {
    'insurance', 'insurer', 'underwriting', 'actuarial', 'actuary', 'claims', 'claim processing',
    'policy', 'premium', 'coverage', 'life insurance', 'health insurance', 'property insurance',
    'casualty insurance', 'reinsurance', 'insurance agent', 'insurance broker', 'risk assessment'
}

# FinTech
FINTECH_KEYWORDS = {
    'fintech', 'financial technology', 'digital payment', 'payment gateway', 'mobile payment',
    'wallet', 'upi', 'paytm', 'razorpay', 'stripe', 'paypal', 'square', 'venmo',
    'peer to peer lending', 'p2p', 'crowdfunding', 'robo advisor', 'insurtech',
    'regtech', 'wealthtech', 'lending platform', 'trading platform', 'cryptocurrency exchange'
}

# Healthcare
HEALTHCARE_KEYWORDS = {
    'healthcare', 'health care', 'hospital', 'clinic', 'medical', 'physician', 'doctor',
    'nurse', 'nursing', 'patient care', 'healthcare administration', 'medical records',
    'ehr', 'electronic health records', 'emr', 'hipaa', 'healthcare it', 'telemedicine',
    'health informatics', 'clinical', 'diagnosis', 'treatment', 'pharmacy', 'pharmacist'
}

# Pharmaceuticals
PHARMACEUTICALS_KEYWORDS = {
    'pharmaceutical', 'pharma', 'drug', 'medicine', 'medication', 'fda', 'clinical trial',
    'drug development', 'pharmaceutical research', 'biopharma', 'pharmaceutical manufacturing',
    'regulatory affairs', 'pharmacovigilance', 'drug discovery', 'formulation', 'api',
    'active pharmaceutical ingredient', 'gmp', 'good manufacturing practice'
}

# Biotechnology
BIOTECH_KEYWORDS = {
    'biotechnology', 'biotech', 'biomedical', 'genetics', 'genomics', 'molecular biology',
    'biochemistry', 'bioinformatics', 'bioprocessing', 'fermentation', 'cell culture',
    'protein engineering', 'antibody', 'vaccine development', 'bioreactor', 'downstream processing'
}

# Manufacturing
MANUFACTURING_KEYWORDS = {
    'manufacturing', 'production', 'factory', 'plant', 'assembly line', 'quality control',
    'qc', 'qa', 'quality assurance', 'lean manufacturing', 'six sigma', 'process improvement',
    'industrial', 'machinery', 'automation', 'plc', 'scada', 'mrp', 'erp', 'sap',
    'supply chain', 'logistics', 'warehouse', 'inventory management'
}

# Automotive
AUTOMOTIVE_KEYWORDS = {
    'automotive', 'automobile', 'car', 'vehicle', 'auto', 'automotive engineering',
    'automotive design', 'oem', 'original equipment manufacturer', 'tier 1', 'tier 2',
    'automotive parts', 'engine', 'transmission', 'chassis', 'automotive testing',
    'adas', 'advanced driver assistance', 'autonomous vehicle', 'ev', 'electric vehicle'
}

# Energy
ENERGY_KEYWORDS = {
    'energy', 'power', 'electricity', 'renewable energy', 'solar', 'wind', 'hydroelectric',
    'nuclear', 'oil', 'gas', 'petroleum', 'refinery', 'power plant', 'utility',
    'energy management', 'energy efficiency', 'smart grid', 'battery', 'energy storage'
}

# Construction
CONSTRUCTION_KEYWORDS = {
    'construction', 'building', 'civil engineering', 'architecture', 'architect',
    'project management', 'construction management', 'site engineer', 'structural engineer',
    'real estate development', 'infrastructure', 'contractor', 'subcontractor',
    'building materials', 'construction project'
}

# Retail
RETAIL_KEYWORDS = {
    'retail', 'retailer', 'store', 'shop', 'merchandising', 'retail management',
    'point of sale', 'pos', 'inventory', 'retail operations', 'customer service',
    'retail sales', 'brick and mortar', 'retail chain', 'franchise'
}

# E-commerce
ECOMMERCE_KEYWORDS = {
    'e-commerce', 'ecommerce', 'online retail', 'online store', 'online shopping',
    'marketplace', 'amazon', 'flipkart', 'shopify', 'magento', 'woocommerce',
    'online payment', 'order management', 'fulfillment', 'last mile delivery',
    'digital commerce', 'omnichannel', 'online marketplace'
}

# Logistics
LOGISTICS_KEYWORDS = {
    'logistics', 'supply chain', 'warehouse', 'distribution', 'shipping', 'freight',
    'transportation', 'logistics management', 'inventory', 'procurement', 'sourcing',
    '3pl', 'third party logistics', '4pl', 'warehouse management', 'wms',
    'transport management', 'tms', 'cold chain', 'reverse logistics'
}

# Telecommunications
TELECOM_KEYWORDS = {
    'telecommunications', 'telecom', 'telecommunication', 'mobile network', 'wireless',
    'cellular', '5g', '4g', 'lte', 'network operator', 'isp', 'internet service provider',
    'voip', 'telephony', 'telecom infrastructure', 'network engineering', 'rf engineer'
}

# Media & Entertainment
MEDIA_KEYWORDS = {
    'media', 'entertainment', 'broadcasting', 'television', 'tv', 'radio', 'film',
    'movie', 'cinema', 'content creation', 'content production', 'streaming',
    'netflix', 'youtube', 'podcast', 'journalism', 'news', 'publishing', 'editorial'
}

# Advertising & Marketing
MARKETING_KEYWORDS = {
    'advertising', 'marketing', 'digital marketing', 'social media marketing', 'seo',
    'sem', 'ppc', 'pay per click', 'content marketing', 'email marketing', 'e-mail marketing',
    'branding', 'public relations', 'pr', 'marketing campaign', 'ad campaign', 'marketing strategy',
    'market research', 'customer acquisition', 'lead generation', 'crm', 'salesforce',
    'marketing executive', 'marketing professional', 'marketing specialist'
}

# Business Development & Sales (add as separate domain or merge with Marketing)
BUSINESS_DEV_KEYWORDS = {
    'business development', 'bd', 'sales', 'sales executive', 'sales development',
    'sales professional', 'sales specialist', 'account executive', 'account manager',
    'client acquisition', 'business growth', 'revenue generation', 'sales strategy',
    'sales management', 'sales operations', 'territory management', 'key account management',
    'sales representative', 'sales consultant', 'business development executive',
    'business development manager', 'sales manager', 'sales director'
}

# Education Technology
EDTECH_KEYWORDS = {
    'edtech', 'education technology', 'e-learning', 'online learning', 'lms',
    'learning management system', 'educational software', 'educational platform',
    'mooc', 'massive open online course', 'virtual classroom', 'educational app'
}

# Public Sector
PUBLIC_SECTOR_KEYWORDS = {
    'public sector', 'government', 'govt', 'public administration', 'civil service',
    'public policy', 'government agency', 'municipal', 'federal', 'state government',
    'public service', 'regulatory', 'compliance', 'public works'
}

# Real Estate
REAL_ESTATE_KEYWORDS = {
    'real estate', 'property', 'realty', 'realtor', 'broker', 'property management',
    'commercial real estate', 'residential real estate', 'real estate development',
    'property investment', 'real estate agent', 'property sales', 'leasing'
}

# Hospitality
HOSPITALITY_KEYWORDS = {
    'hospitality', 'hotel', 'resort', 'restaurant', 'catering', 'food service',
    'hospitality management', 'hotel management', 'front desk', 'concierge',
    'guest services', 'hospitality industry', 'tourism', 'travel'
}

# Travel & Tourism
TRAVEL_KEYWORDS = {
    'travel', 'tourism', 'tour', 'travel agency', 'travel agent', 'airline',
    'aviation', 'airport', 'cruise', 'travel booking', 'tour operator', 'destination',
    'travel industry', 'hospitality and tourism'
}

# Agriculture
AGRICULTURE_KEYWORDS = {
    'agriculture', 'farming', 'agri', 'agricultural', 'crop', 'livestock', 'farming',
    'agricultural engineering', 'agribusiness', 'agricultural technology', 'agtech',
    'precision agriculture', 'sustainable agriculture', 'organic farming'
}

# Legal & Compliance
LEGAL_KEYWORDS = {
    'legal', 'law', 'lawyer', 'attorney', 'legal counsel', 'compliance', 'regulatory compliance',
    'legal affairs', 'contract', 'litigation', 'legal research', 'paralegal', 'legal assistant',
    'corporate law', 'intellectual property', 'ip', 'patent', 'trademark', 'legal department'
}

# Human Resources
HR_KEYWORDS = {
    'human resources', 'hr', 'recruitment', 'talent acquisition', 'hiring', 'recruiter',
    'hr manager', 'people operations', 'employee relations', 'payroll', 'benefits',
    'compensation', 'performance management', 'training and development', 'organizational development',
    'hr analytics', 'talent management', 'workforce planning'
}

# Environmental & Sustainability
ENVIRONMENTAL_KEYWORDS = {
    'environmental', 'sustainability', 'sustainable', 'green', 'renewable', 'environmental engineering',
    'environmental science', 'climate', 'carbon', 'emissions', 'waste management', 'recycling',
    'environmental compliance', 'esg', 'environmental social governance', 'clean energy',
    'environmental impact', 'conservation'
}

# ============================================================================
# DOMAIN MAPPING
# ============================================================================
DOMAIN_KEYWORD_MAP = {
    "Information Technology": IT_KEYWORDS,
    "Software Development": IT_KEYWORDS,  # Share keywords with IT
    "Cloud Computing": CLOUD_KEYWORDS,
    "Cybersecurity": CYBERSECURITY_KEYWORDS,
    "Data Science": DATA_SCIENCE_KEYWORDS,
    "Blockchain": BLOCKCHAIN_KEYWORDS,
    "Internet of Things": IOT_KEYWORDS,
    "Banking": BANKING_KEYWORDS,
    "Finance": FINANCE_KEYWORDS,
    "Insurance": INSURANCE_KEYWORDS,
    "FinTech": FINTECH_KEYWORDS,
    "Healthcare": HEALTHCARE_KEYWORDS,
    "Pharmaceuticals": PHARMACEUTICALS_KEYWORDS,
    "Biotechnology": BIOTECH_KEYWORDS,
    "Manufacturing": MANUFACTURING_KEYWORDS,
    "Automotive": AUTOMOTIVE_KEYWORDS,
    "Energy": ENERGY_KEYWORDS,
    "Construction": CONSTRUCTION_KEYWORDS,
    "Retail": RETAIL_KEYWORDS,
    "E-commerce": ECOMMERCE_KEYWORDS,
    "Logistics": LOGISTICS_KEYWORDS,
    "Telecommunications": TELECOM_KEYWORDS,
    "Media & Entertainment": MEDIA_KEYWORDS,
    "Advertising & Marketing": MARKETING_KEYWORDS,
    "Business Development": BUSINESS_DEV_KEYWORDS,
    "Education Technology": EDTECH_KEYWORDS,
    "Public Sector": PUBLIC_SECTOR_KEYWORDS,
    "Real Estate": REAL_ESTATE_KEYWORDS,
    "Hospitality": HOSPITALITY_KEYWORDS,
    "Travel & Tourism": TRAVEL_KEYWORDS,
    "Agriculture": AGRICULTURE_KEYWORDS,
    "Legal & Compliance": LEGAL_KEYWORDS,
    "Human Resources": HR_KEYWORDS,
    "Environmental & Sustainability": ENVIRONMENTAL_KEYWORDS
}

# ============================================================================
# EDUCATION KEYWORDS TO IGNORE
# ============================================================================
EDUCATION_KEYWORDS = {
    'b.tech', 'btech', 'm.tech', 'mtech', 'b.e', 'be', 'm.e', 'me',
    'bachelor', 'master', 'phd', 'doctorate', 'degree', 'diploma', 'certificate',
    'education', 'university', 'college', 'school', 'academic', 'course', 'curriculum',
    'gpa', 'cgpa', 'grade', 'semester', 'thesis', 'dissertation', 'graduation',
    'undergraduate', 'graduate', 'postgraduate', 'alumni', 'student', 'studied'
}

# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for keyword matching."""
    # Convert to lowercase and remove extra whitespace
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text


def is_client_industry_mention(text: str, keyword: str) -> bool:
    """
    Check if keyword appears in context of "working with clients" rather than "working in" domain.
    
    Args:
        text: Full resume text
        keyword: The matched keyword
        
    Returns:
        True if keyword is mentioned as a client industry, False otherwise
    """
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    
    # Patterns that indicate client industry mentions (not their domain)
    client_industry_patterns = [
        r'working\s+(across|with|in)\s+.*?' + re.escape(keyword_lower),
        r'clients?\s+(in|from|across)\s+.*?' + re.escape(keyword_lower),
        r'industries?\s+(like|such\s+as|including|:)\s+.*?' + re.escape(keyword_lower),
        r'verticals?\s+(like|such\s+as|including|:)\s+.*?' + re.escape(keyword_lower),
        r'sectors?\s+(like|such\s+as|including|:)\s+.*?' + re.escape(keyword_lower),
        r'domains?\s+(like|such\s+as|including|:)\s+.*?' + re.escape(keyword_lower),
    ]
    
    for pattern in client_industry_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    return False


def is_project_context(text: str, keyword: str) -> bool:
    """
    Check if keyword appears in project context (should be weighted lower or ignored).
    
    Args:
        text: Full resume text
        keyword: The matched keyword
        
    Returns:
        True if keyword is in project context, False otherwise
    """
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    
    # Find keyword position
    keyword_pos = text_lower.find(keyword_lower)
    if keyword_pos == -1:
        return False
    
    # Look for project section headers before the keyword
    project_section_patterns = [
        r'projects?\s*:?\s*',
        r'project\s+experience\s*:?\s*',
        r'personal\s+projects?\s*:?\s*',
        r'academic\s+projects?\s*:?\s*',
        r'side\s+projects?\s*:?\s*',
        r'i\s+have\s+done\s+my\s+project',
        r'project\s+description',
    ]
    
    # Check if any project section header appears before the keyword
    text_before_keyword = text_lower[:keyword_pos]
    for pattern in project_section_patterns:
        if re.search(pattern, text_before_keyword, re.IGNORECASE):
            # Check if keyword is within reasonable distance (1000 chars) after project header
            match = re.search(pattern, text_before_keyword, re.IGNORECASE)
            if match:
                distance = keyword_pos - match.end()
                if distance < 1000:  # Within 1000 characters of project section
                    return True
    
    return False


def is_education_context(text: str, keyword: str) -> bool:
    """
    Check if keyword appears in education context (should be ignored).
    
    Args:
        text: Full resume text
        keyword: The matched keyword
        
    Returns:
        True if keyword is in education context, False otherwise
    """
    # Find keyword position
    keyword_lower = keyword.lower()
    text_lower = text.lower()
    
    # Look for education section headers near the keyword
    education_section_patterns = [
        r'education\s*:?\s*' + re.escape(keyword_lower),
        r'educational\s+background\s*:?\s*' + re.escape(keyword_lower),
        r'academic\s+qualification\s*:?\s*' + re.escape(keyword_lower),
        r'degree\s*:?\s*' + re.escape(keyword_lower),
        r'qualification\s*:?\s*' + re.escape(keyword_lower),
    ]
    
    for pattern in education_section_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    # Check if keyword is near education keywords
    keyword_pos = text_lower.find(keyword_lower)
    if keyword_pos != -1:
        # Check 200 characters before and after
        start = max(0, keyword_pos - 200)
        end = min(len(text_lower), keyword_pos + len(keyword_lower) + 200)
        context = text_lower[start:end]
        
        # If education keywords are nearby, likely education context
        for edu_keyword in EDUCATION_KEYWORDS:
            if edu_keyword in context:
                return True
    
    return False


def has_professional_it_role(text: str) -> bool:
    """
    Check if the resume indicates a professional IT/Software Development role.
    
    Args:
        text: Full resume text
        
    Returns:
        True if professional IT role detected, False otherwise
    """
    text_lower = text.lower()
    
    # Professional IT role patterns (in title, objective, or experience sections)
    it_role_patterns = [
        r'\.net\s+developer',
        r'asp\.net\s+developer',
        r'software\s+developer',
        r'software\s+engineer',
        r'web\s+developer',
        r'application\s+developer',
        r'full\s+stack\s+developer',
        r'backend\s+developer',
        r'frontend\s+developer',
        r'programmer',
        r'developer\s+with',
        r'developer\s+having',
        r'\.net\s+developer\s+with',
    ]
    
    # Check first 2000 characters (title, objective, summary sections)
    early_text = text_lower[:2000]
    for pattern in it_role_patterns:
        if re.search(pattern, early_text, re.IGNORECASE):
            return True
    
    return False


def extract_domains(text: str) -> List[str]:
    """
    Extract professional domains/industries from resume text using keyword matching.
    
    Args:
        text: Resume text to extract domains from
        
    Returns:
        List of domain strings (empty list if none found)
    """
    if not text or not isinstance(text, str):
        logger.warning("Invalid text input for domain extraction")
        return []
    
    # Normalize text
    normalized_text = normalize_text(text)
    
    # Check if professional IT role detected (for priority boost)
    is_it_professional = has_professional_it_role(text)
    
    # Track matched domains with scores
    domain_scores = {}
    
    # Match keywords for each domain
    for domain, keywords in DOMAIN_KEYWORD_MAP.items():
        score = 0
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Special handling for .NET and ASP.NET (handle periods in keywords)
            if '.net' in keyword_lower or 'asp.net' in keyword_lower:
                # For .NET keywords, create pattern that handles periods flexibly
                # Match: .NET, .net, dotnet, ASP.NET, ASP.Net, asp.net, etc.
                if keyword_lower == '.net':
                    pattern = r'(?:\.|dot)\s*net\b'
                elif keyword_lower == 'asp.net':
                    pattern = r'asp\.?\s*\.?\s*net\b'
                elif keyword_lower == 'asp.net core':
                    pattern = r'asp\.?\s*\.?\s*net\s+core\b'
                elif keyword_lower == 'asp.net mvc':
                    pattern = r'asp\.?\s*\.?\s*net\s+mvc\b'
                elif keyword_lower == '.net core':
                    pattern = r'\.?\s*\.?\s*net\s+core\b'
                elif keyword_lower == '.net developer':
                    pattern = r'\.?\s*\.?\s*net\s+developer\b'
                else:
                    # Generic .net pattern - make period optional
                    escaped = re.escape(keyword_lower)
                    pattern = escaped.replace(r'\.', r'\.?\s*\.?')
                    pattern = r'\b' + pattern + r'\b'
            elif len(keyword_lower.split()) == 1:
                # Single word - use word boundaries
                pattern = r'\b' + re.escape(keyword_lower) + r'\b'
            else:
                # Multi-word phrase - use word boundaries at start and end
                pattern = r'\b' + re.escape(keyword_lower) + r'\b'
            
            # Find all matches using regex
            matches = re.findall(pattern, normalized_text, re.IGNORECASE)
            
            if matches:
                # Skip if it's in education context
                if not is_education_context(text, keyword):
                    # Skip if it's mentioned as a client industry (not their domain)
                    if not is_client_industry_mention(text, keyword):
                        # Check if it's in project context (weight lower)
                        is_in_project = is_project_context(text, keyword)
                        
                        # Count occurrences (more occurrences = higher score)
                        occurrences = len(matches)
                        
                        # Weight professional role keywords much higher
                        professional_role_indicators = [
                            'executive', 'manager', 'specialist', 'professional', 'developer',
                            'engineer', 'analyst', 'consultant', 'director', 'lead', 'senior'
                        ]
                        
                        if any(indicator in keyword_lower for indicator in professional_role_indicators):
                            # Professional role keywords get highest weight
                            if is_in_project:
                                score += occurrences * 1  # Lower weight if in project
                            else:
                                score += occurrences * 5  # Very high weight for professional roles
                        elif is_in_project:
                            # Project keywords get very low weight.
                            # For clear IT professionals we will completely ignore project-only
                            # domains (handled below with is_it_professional).
                            score += occurrences * 0.1  # Very low weight for project mentions
                        else:
                            # Regular keywords
                            score += occurrences
        
        # Boost IT/Software Development if professional IT role detected
        if is_it_professional and domain in ["Information Technology", "Software Development"]:
            score += 10  # Significant boost for IT professionals

        # If candidate is clearly an IT professional, ignore domains that are supported
        # **only** by project-context matches (e.g., Data Science, Energy from a single project).
        # We approximate this by dropping non-IT domains whose score came only from
        # low-weight project keywords (score < 1.0 and not IT domains).
        if is_it_professional and domain not in ["Information Technology", "Software Development"]:
            if score < 1.0:
                score = 0
        
        # Only add domain if it has a score > 0
        if score > 0:
            domain_scores[domain] = score
    
    # Sort domains by score (descending)
    sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Extract domain names
    matched_domains = [domain for domain, score in sorted_domains]
    
    # Remove duplicates (e.g., "Information Technology" and "Software Development" might both match)
    # Keep only unique domains. If both IT and Software Development are present, keep only one.
    unique_domains = []
    seen = set()
    has_it = False
    has_software_dev = False
    
    for domain in matched_domains:
        if domain == "Information Technology":
            has_it = True
        if domain == "Software Development":
            has_software_dev = True
        
        if domain not in seen:
            unique_domains.append(domain)
            seen.add(domain)
    
    # Collapse IT + Software Development into a single preferred label.
    # Preference order: "Information Technology" first, otherwise "Software Development".
    if has_it and has_software_dev:
        # Remove "Software Development" if both exist
        unique_domains = [d for d in unique_domains if d != "Software Development"]
    
    # Limit to top 5 domains to avoid too many matches
    result = unique_domains[:5]
    
    if result:
        logger.info(f"Extracted domains: {result}")
    else:
        logger.warning("No domains extracted from resume text")
    
    return result


def extract_domain(text: str) -> Optional[List[str]]:
    """
    Main extraction function (alias for extract_domains for compatibility).
    
    Args:
        text: Resume text to extract domains from
        
    Returns:
        List of domain strings or None if extraction failed
    """
    try:
        domains = extract_domains(text)
        return domains if domains else []
    except Exception as e:
        logger.error(f"Domain extraction failed: {e}")
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
    - Developed web applications using Python and Django
    - Worked on cloud infrastructure with AWS
    - Implemented security measures for banking applications
    
    SKILLS:
    Python, Java, AWS, Docker, Kubernetes
    
    EDUCATION:
    B.Tech in Computer Science
    """
    
    domains = extract_domains(sample_resume)
    print(f"Extracted domains: {domains}")
    # Expected: ["Information Technology", "Software Development", "Cloud Computing", "Banking"]

