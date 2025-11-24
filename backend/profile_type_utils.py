"""
Utility helpers to derive and reuse candidate profile types (Java, .Net, SAP, etc.).

The goal is to keep a single source of truth for profile type detection so that
resume parsing, SQL filtering, and search/ranking code stay in sync.
"""

import logging
import re
from functools import lru_cache
from typing import Iterable, List, Optional, Tuple, Dict, Set, NamedTuple, Any

logger = logging.getLogger(__name__)

# Legacy rules for backward compatibility (simple set-based)
PROFILE_TYPE_RULES = [
    ("Java", {
        "java", "jdk", "jvm", "jre", "spring", "spring boot", "spring mvc", "spring cloud",
        "hibernate", "jpa", "j2ee", "jakarta ee", "servlet", "jsp", "jsf", "struts",
        "microservices", "rest api", "maven", "gradle", "ant", "jboss", "weblogic", "tomcat"
    }),
    (".Net", {
        ".net", "dotnet", "c#", "csharp", "asp.net", "asp.net core", ".net core", ".net framework",
        "entity framework", "ef core", "wpf", "winforms", "web api", "linq", "xamarin", "blazor",
        "ado.net", "wcf", "mvc", "razor", "signalr"
    }),
    ("Python", {
        "python", "django", "flask", "fastapi", "tornado", "pyramid", "bottle",
        "pandas", "numpy", "scipy", "matplotlib", "seaborn", "sqlalchemy", "celery",
        "redis", "asyncio", "pytest", "jupyter", "scikit-learn"
    }),
    ("JavaScript", {
        "javascript", "js", "node", "node.js", "react", "angular", "vue", "typescript",
        "es6", "jquery", "express", "next.js", "nuxt.js", "nuxt", "webpack", "babel",
        "gulp", "grunt", "npm", "yarn", "redux", "mobx"
    }),
    ("Full Stack", {
        "full stack", "fullstack", "mern", "mean", "mevn", "lamp", "lemp",
        "next.js", "nuxt", "react", "node.js", "express", "mongodb", "postgresql", "mysql",
        "frontend", "backend", "full stack developer"
    }),
    ("DevOps", {
        "devops", "ci/cd", "cicd", "jenkins", "gitlab ci", "github actions", "azure devops",
        "docker", "kubernetes", "k8s", "helm", "terraform", "ansible", "chef", "puppet",
        "monitoring", "prometheus", "grafana", "elk", "splunk", "nagios"
    }),
    ("Cloud / Infra", {
        "aws", "azure", "gcp", "google cloud", "cloud architect", "cloud engineer",
        "ec2", "s3", "lambda", "iam", "eks", "aks", "gke", "cloudformation", "vmware",
        "vsphere", "openstack", "cloud infrastructure"
    }),
    ("Data Engineering", {
        "data engineer", "data engineering", "etl", "elt", "airflow", "luigi",
        "snowflake", "spark", "pyspark", "hadoop", "hdfs", "databricks", "kafka",
        "flume", "sqoop", "redshift", "bigquery", "data pipeline", "data warehouse"
    }),
    ("Data Science", {
        "data science", "machine learning", "deep learning", "ml engineer", "ai engineer",
        "artificial intelligence", "llm", "nlp", "computer vision", "tensorflow", "pytorch",
        "scikit-learn", "keras", "neural network", "data scientist"
    }),
    ("Business Intelligence (BI)", {
        "power bi", "tableau", "qlik", "looker", "ssis", "ssrs", "business intelligence",
        "bi", "data visualization", "dashboard", "cognos", "microstrategy", "spotfire"
    }),
    ("Testing / QA", {
        "qa", "quality assurance", "manual testing", "automation testing", "test automation",
        "selenium", "cypress", "playwright", "junit", "testng", "pytest", "mocha", "jest",
        "postman", "api testing", "performance testing", "jmeter", "load testing"
    }),
    ("SAP", {
        "sap", "abap", "hana", "sap hana", "s/4hana", "successfactors", "ariba", "bw",
        "fiori", "sap mm", "sap sd", "sap fico", "sap basis", "sap pi", "sap po"
    }),
    ("ERP", {
        "erp", "oracle apps", "oracle e-business", "oracle fusion", "d365", "dynamics 365",
        "dynamics", "business central", "netsuite", "workday", "odoo", "peopleSoft", "sage"
    }),
    ("Microsoft Power Platform", {
        "power platform", "power apps", "power automate", "power bi", "power virtual agents",
        "dataverse", "power fx", "canvas app", "model-driven app", "power pages"
    }),
    ("RPA", {
        "rpa", "robotic process automation", "ui path", "uipath", "automation anywhere",
        "blue prism", "rpa developer", "process automation", "workfusion"
    }),
    ("Cyber Security", {
        "cyber security", "cybersecurity", "information security", "soc", "security operations center",
        "siem", "penetration testing", "pen testing", "ethical hacking", "vapt", "vulnerability assessment",
        "security analyst", "security engineer", "threat hunting", "incident response"
    }),
    ("Mobile Development", {
        "android", "ios", "kotlin", "swift", "flutter", "react native", "xamarin",
        "mobile app", "ios development", "android development", "ionic", "cordova"
    }),
    ("Salesforce", {
        "salesforce", "sfdc", "apex", "visualforce", "lightning", "lwc", "lightning web components",
        "soql", "sosl", "salesforce admin", "salesforce developer", "sales cloud", "service cloud"
    }),
    ("Low Code / No Code", {
        "low code", "no code", "low-code", "no-code", "appgyver", "outsystems", "mendix",
        "low code platform", "citizen developer", "bubble", "zapier"
    }),
    ("Database", {
        "database", "database administrator", "dba", "mysql", "postgresql", "oracle db",
        "oracle database", "sql server", "mongodb", "redis", "cassandra", "database design", "sql", "nosql"
    }),
    ("Integration / APIs", {
        "integration", "api", "apis", "rest api", "restful api", "soap", "graphql",
        "microservices", "mule", "mulesoft", "boomi", "api integration", "enterprise integration",
        "tibco", "webmethods"
    }),
    ("UI/UX", {
        "ui designer", "ux designer", "ui/ux", "user interface", "user experience",
        "figma", "adobe xd", "sketch", "wireframing", "prototyping", "ui design", "ux design",
        "interaction design", "invision", "zeplin"
    }),
    ("Support", {
        "technical support", "it support", "support engineer", "help desk", "desktop support",
        "application support", "production support", "l1 support", "l2 support", "l3 support",
        "customer support", "support analyst", "troubleshooting", "incident management"
    }),
    ("Business Development", {
        "business development", "bd", "business dev", "bde", "business development executive",
        "business development manager", "b2b sales", "client acquisition", "market expansion",
        "partnership development", "strategic partnerships", "account development"
    }),
]

# Enhanced rules with weights and expanded keywords
PROFILE_TYPE_RULES_ENHANCED: List[Tuple[str, Dict[str, float]]] = [
    (
        "Java",
        {
            "java": 5.0, "jdk": 4.5, "jvm": 4.5, "jre": 4.0,
            "spring boot": 4.5, "spring": 4.0, "spring mvc": 3.5, "spring cloud": 3.5,
            "hibernate": 3.5, "jpa": 3.0, "j2ee": 3.0, "jakarta ee": 3.0,
            "servlet": 2.5, "jsp": 2.5, "jsf": 2.5, "struts": 2.0,
            "microservices": 2.0, "rest api": 2.0, "soap": 1.5,
            "maven": 2.0, "gradle": 2.0,
        }
    ),
    (
        ".Net",
        {
            "c#": 5.0, "csharp": 5.0, "c sharp": 5.0,
            "asp.net": 4.5, "asp.net core": 4.5, ".net core": 4.5,
            ".net framework": 4.0, "dotnet": 4.0, ".net": 3.5,
            "entity framework": 3.5, "ef core": 3.5, "mvc": 3.0,
            "wpf": 2.5, "winforms": 2.5, "web api": 2.5,
            "linq": 2.0, "xamarin": 2.0, "blazor": 2.0,
        }
    ),
    (
        "Python",
        {
            "python": 5.0, "django": 4.0, "flask": 3.5, "fastapi": 3.5,
            "tornado": 2.5, "pyramid": 2.0,
            "pandas": 3.0, "numpy": 2.5, "scipy": 2.0,
            "matplotlib": 2.0, "seaborn": 1.5, "sqlalchemy": 2.5,
            "celery": 2.0, "redis": 1.5, "asyncio": 1.5,
        }
    ),
    (
        "JavaScript",
        {
            "javascript": 5.0, "js": 4.5, "typescript": 4.5,
            "node.js": 4.5, "node": 4.0,
            "react": 4.0, "angular": 3.5, "vue": 3.5,
            "es6": 3.0, "jquery": 2.5, "express": 3.0,
            "next.js": 3.0, "nuxt.js": 2.5, "nuxt": 2.5,
            "webpack": 2.0, "babel": 1.5,
        }
    ),
    (
        "Full Stack",
        {
            "full stack": 5.0, "fullstack": 4.5,
            "mern": 4.0, "mean": 4.0, "mevn": 3.5, "lamp": 3.0,
            "next.js": 3.0, "nuxt": 2.5,
            "react": 2.0, "node.js": 2.0, "express": 2.0,
            "mongodb": 2.0, "postgresql": 1.5, "mysql": 1.5,
        }
    ),
    (
        "DevOps",
        {
            "devops": 5.0, "ci/cd": 4.0, "cicd": 4.0,
            "kubernetes": 4.5, "docker": 4.0, "helm": 3.0,
            "terraform": 3.5, "ansible": 3.0, "chef": 2.5, "puppet": 2.5,
            "jenkins": 3.0, "gitlab ci": 2.5, "github actions": 2.5,
            "monitoring": 2.0, "prometheus": 2.0, "grafana": 1.5,
        }
    ),
    (
        "Data Engineering",
        {
            "data engineer": 5.0, "data engineering": 4.5,
            "etl": 4.0, "elt": 3.5, "airflow": 3.5,
            "snowflake": 3.5, "spark": 3.5, "pyspark": 3.0,
            "hadoop": 3.0, "hdfs": 2.5, "databricks": 3.0,
            "kafka": 2.5, "flume": 2.0, "sqoop": 2.0,
            "redshift": 2.0, "bigquery": 2.0,
        }
    ),
    (
        "Data Science",
        {
            "data science": 5.0, "machine learning": 4.5,
            "deep learning": 3.5, "ml engineer": 4.0,
            "ai engineer": 4.0, "artificial intelligence": 3.5,
            "llm": 3.5, "nlp": 3.0, "computer vision": 2.5,
            "tensorflow": 2.5, "pytorch": 2.5, "scikit-learn": 2.0,
        }
    ),
    (
        "Testing / QA",
        {
            "qa": 4.0, "quality assurance": 4.5,
            "manual testing": 3.0, "automation testing": 4.0,
            "selenium": 3.5, "cypress": 3.0, "playwright": 2.5,
            "junit": 2.5, "testng": 2.5,
            "postman": 2.0, "api testing": 2.0,
            "performance testing": 2.0, "jmeter": 2.0,
        }
    ),
    (
        "SAP",
        {
            "sap": 5.0, "abap": 4.0,
            "sap hana": 3.5, "s/4hana": 3.5, "hana": 3.5,
            "fiori": 2.5, "sap mm": 2.5, "sap sd": 2.5,
            "sap fico": 2.5, "successfactors": 3.0,
            "ariba": 2.5, "sap basis": 2.0,
        }
    ),
    (
        "ERP",
        {
            "erp": 4.0,
            "oracle apps": 3.5, "oracle e-business": 3.5, "oracle fusion": 3.0,
            "d365": 3.0, "dynamics 365": 3.0, "dynamics": 3.0,
            "business central": 2.5, "netsuite": 2.5,
            "workday": 2.5, "odoo": 2.0, "peopleSoft": 2.0,
        }
    ),
    (
        "Cloud / Infra",
        {
            "aws": 4.0, "azure": 4.0, "gcp": 3.5, "google cloud": 3.5,
            "ec2": 2.5, "s3": 2.5, "lambda": 2.0,
            "iam": 2.0, "eks": 2.5, "aks": 2.5,
            "cloud architect": 4.5, "cloud engineer": 4.0,
            "terraform": 3.0, "cloudformation": 2.0, "vmware": 2.0,
        }
    ),
    (
        "Business Intelligence (BI)",
        {
            "power bi": 5.0, "tableau": 4.5, "qlik": 4.0, "looker": 3.5,
            "ssis": 3.0, "ssrs": 3.0, "business intelligence": 4.0,
            "bi": 3.5, "data visualization": 2.5, "dashboard": 2.0,
        }
    ),
    (
        "Microsoft Power Platform",
        {
            "power platform": 5.0, "power apps": 4.5, "power automate": 4.0,
            "power bi": 4.0, "power virtual agents": 3.5,
            "dataverse": 3.5, "power fx": 3.0, "canvas app": 2.5,
            "model-driven app": 2.5, "power pages": 2.0,
        }
    ),
    (
        "RPA",
        {
            "rpa": 5.0, "robotic process automation": 4.5,
            "ui path": 4.5, "uipath": 4.5,
            "automation anywhere": 4.0, "blue prism": 3.5,
            "rpa developer": 4.0, "process automation": 3.0,
        }
    ),
    (
        "Cyber Security",
        {
            "cyber security": 5.0, "cybersecurity": 5.0, "information security": 4.5,
            "soc": 4.0, "security operations center": 4.0,
            "siem": 3.5, "security information": 3.5,
            "penetration testing": 3.5, "pen testing": 3.5,
            "ethical hacking": 3.0, "vapt": 3.0, "vulnerability assessment": 3.0,
            "security analyst": 3.5, "security engineer": 3.5,
        }
    ),
    (
        "Mobile Development",
        {
            "android": 4.5, "ios": 4.5, "mobile development": 5.0,
            "kotlin": 4.0, "swift": 4.0,
            "flutter": 3.5, "react native": 3.5, "xamarin": 3.0,
            "mobile app": 3.0, "ios development": 3.5, "android development": 3.5,
        }
    ),
    (
        "Salesforce",
        {
            "salesforce": 5.0, "sfdc": 4.0,
            "apex": 4.0, "visualforce": 3.5,
            "lightning": 3.5, "lightning web components": 3.5, "lwc": 3.5,
            "soql": 3.0, "sosl": 3.0, "salesforce admin": 3.0,
            "salesforce developer": 4.0, "sales cloud": 2.5, "service cloud": 2.5,
        }
    ),
    (
        "Low Code / No Code",
        {
            "low code": 5.0, "no code": 4.5, "low-code": 5.0, "no-code": 4.5,
            "appgyver": 3.5, "outsystems": 3.5, "mendix": 3.5,
            "low code platform": 4.0, "citizen developer": 2.5,
        }
    ),
    (
        "Database",
        {
            "database": 4.0, "database administrator": 4.5, "dba": 4.0,
            "mysql": 3.5, "postgresql": 3.5, "oracle db": 3.5, "oracle database": 3.5,
            "sql server": 3.5, "mongodb": 3.0, "redis": 2.5, "cassandra": 2.5,
            "database design": 3.0, "sql": 2.5, "nosql": 2.5,
        }
    ),
    (
        "Integration / APIs",
        {
            "integration": 4.0, "api": 3.5, "apis": 3.5,
            "rest api": 3.5, "restful api": 3.5, "soap": 3.0,
            "graphql": 3.0, "microservices": 3.0,
            "mule": 3.5, "mulesoft": 3.5, "boomi": 3.0,
            "api integration": 3.5, "enterprise integration": 3.0,
        }
    ),
    (
        "UI/UX",
        {
            "ui designer": 4.5, "ux designer": 4.5, "ui/ux": 5.0,
            "user interface": 4.0, "user experience": 4.0,
            "figma": 3.5, "adobe xd": 3.0, "sketch": 3.0,
            "wireframing": 3.0, "prototyping": 2.5, "ui design": 3.5,
            "ux design": 3.5, "interaction design": 2.5,
        }
    ),
    (
        "Support",
        {
            "technical support": 5.0, "it support": 4.5, "support engineer": 4.5,
            "help desk": 4.0, "desktop support": 3.5,
            "application support": 4.0, "production support": 4.0,
            "l1 support": 3.0, "l2 support": 3.5, "l3 support": 4.0,
            "customer support": 3.0, "support analyst": 3.5,
        }
    ),
    (
        "Business Development",
        {
            "business development": 5.0, "bd": 4.5, "business dev": 4.5,
            "bde": 4.0, "business development executive": 4.5,
            "business development manager": 4.5, "b2b sales": 4.0,
            "client acquisition": 3.5, "market expansion": 3.5,
            "partnership development": 3.5, "strategic partnerships": 3.5,
            "account development": 3.0,
        }
    ),
]

# Negative keywords to exclude false positives
NEGATIVE_KEYWORDS: Dict[str, Set[str]] = {
    "Java": {"javascript", "javac", "java island", "java coffee"},
    ".Net": {"avoid .net", "don't use .net", "not .net"},
    "Python": {"python snake", "monty python"},
}

# Pre-compiled regex patterns for word boundary matching
COMPILED_PATTERNS: Dict[str, Dict[str, re.Pattern]] = {}

DEFAULT_PROFILE_TYPE = "Generalist"


def canonicalize_profile_type(value: Optional[str]) -> str:
    """Normalize profile type labels to a consistent, canonical form."""
    if not value:
        return DEFAULT_PROFILE_TYPE
    
    normalized = str(value).strip()
    if not normalized:
        return DEFAULT_PROFILE_TYPE
    
    lowered = normalized.lower()
    
    # Special handling for .Net variations (net, .net, dotnet)
    # This ensures "Net", "net", ".net", "dotnet" all map to ".Net"
    if lowered in ("net", ".net", "dotnet"):
        return ".Net"
    
    # Check against canonical forms from PROFILE_TYPE_RULES
    for profile_type, _ in PROFILE_TYPE_RULES:
        if lowered == profile_type.lower():
            return profile_type
    
    # Keep known default if user explicitly passed it
    if lowered == DEFAULT_PROFILE_TYPE.lower():
        return DEFAULT_PROFILE_TYPE
    
    # If value doesn't match any known profile type, return DEFAULT_PROFILE_TYPE
    # This prevents skills from being stored as profile types
    return DEFAULT_PROFILE_TYPE


def canonicalize_profile_type_list(values: Optional[Iterable[str]]) -> List[str]:
    """Canonicalize and deduplicate a list of profile type labels."""
    if not values:
        return []
    
    canonicalized = []
    seen = set()
    for value in values:
        canonical = canonicalize_profile_type(value)
        if canonical and canonical not in seen:
            canonicalized.append(canonical)
            seen.add(canonical)
    return canonicalized

# Profile Type Compatibility Rules
# Defines which profile types can coexist (multi-profile candidates)
# This prevents incompatible combinations like "Python,Java" while allowing logical pairs
# Note: Compatibility is bidirectional - if A is compatible with B, then B is compatible with A
PROFILE_TYPE_COMPATIBILITY = {
    "Python": ["Data Science", "Full Stack", "Data Engineering", "DevOps"],
    "Java": [".Net", "Full Stack", "DevOps"],
    ".Net": ["Java", "Full Stack", "DevOps", "JavaScript"],  # Added JavaScript (Full Stack developers often have both)
    "JavaScript": ["Full Stack", "UI/UX", "Mobile Development", ".Net"],  # Added .Net (Full Stack developers often have both)
    "Full Stack": ["Python", "Java", ".Net", "JavaScript", "DevOps"],
    "Data Science": ["Python", "Data Engineering"],
    "Data Engineering": ["Python", "Data Science", "DevOps"],
    "DevOps": ["Python", "Java", ".Net", "Full Stack", "Cloud / Infra", "Data Engineering"],
    "Cloud / Infra": ["DevOps", "Data Engineering"],
    "Mobile Development": ["JavaScript", "Full Stack"],
    "UI/UX": ["JavaScript", "Full Stack"],
    "Testing / QA": ["DevOps", "Full Stack"],
    "SAP": ["ERP"],
    "ERP": ["SAP"],
    "Microsoft Power Platform": ["Low Code / No Code", "Integration / APIs"],
    "Integration / APIs": ["Microsoft Power Platform", "Full Stack"],
    "Low Code / No Code": ["Microsoft Power Platform"],
    "Salesforce": ["Integration / APIs"],
    "Database": ["Data Engineering", "DevOps", "Full Stack"],
    "Business Intelligence (BI)": ["Data Science", "Data Engineering"],
    "Cyber Security": ["DevOps", "Cloud / Infra"],
    "Business Development": [],  # Standalone profile
    "Support": [],  # Standalone profile
}

def are_profile_types_compatible(profile1: str, profile2: str) -> bool:
    """
    Check if two profile types are compatible (can coexist in multi-profile candidate).
    
    Uses bidirectional checking - if A is compatible with B, then B is compatible with A.
    This ensures consistency regardless of which profile is checked first.
    
    Args:
        profile1: First profile type
        profile2: Second profile type
        
    Returns:
        True if profiles are compatible, False otherwise
    """
    profile1 = canonicalize_profile_type(profile1)
    profile2 = canonicalize_profile_type(profile2)
    
    if profile1 == profile2:
        return True
    
    # Bidirectional check: A compatible with B OR B compatible with A
    # This ensures consistency regardless of order
    compatible_list_1 = PROFILE_TYPE_COMPATIBILITY.get(profile1, [])
    compatible_list_2 = PROFILE_TYPE_COMPATIBILITY.get(profile2, [])
    
    # Check both directions for compatibility
    return (profile2 in compatible_list_1) or (profile1 in compatible_list_2)


def _compile_keyword_patterns():
    """Pre-compile regex patterns for all keywords with word boundaries."""
    global COMPILED_PATTERNS
    if COMPILED_PATTERNS:
        return
    
    for profile_type, keyword_weights in PROFILE_TYPE_RULES_ENHANCED:
        COMPILED_PATTERNS[profile_type] = {}
        for keyword in keyword_weights.keys():
            escaped = re.escape(keyword.lower())
            if ' ' in keyword:
                pattern = r'\b' + escaped.replace(r'\ ', r'\s+') + r'\b'
            else:
                pattern = r'\b' + escaped + r'\b'
            COMPILED_PATTERNS[profile_type][keyword] = re.compile(pattern, re.IGNORECASE)

_compile_keyword_patterns()

def _normalize_text_blob(*parts: str) -> str:
    """Lower-case, concatenated blob for keyword detection."""
    normalized_parts = []
    for part in parts:
        if not part:
            continue
        normalized_parts.append(str(part).lower())
    return " ".join(normalized_parts)


def _count_keyword_matches(keyword: str, text: str, profile_type: str) -> int:
    """Count keyword matches using word boundaries."""
    pattern = COMPILED_PATTERNS.get(profile_type, {}).get(keyword)
    if not pattern:
        return 0
    return len(pattern.findall(text))

def _has_negative_context(keyword: str, text: str, profile_type: str) -> bool:
    """Check if keyword appears in negative context."""
    negative_patterns = NEGATIVE_KEYWORDS.get(profile_type, set())
    pattern = COMPILED_PATTERNS.get(profile_type, {}).get(keyword)
    if not pattern:
        return False
    
    for match in pattern.finditer(text):
        start, end = match.span()
        context = text[max(0, start-50):min(len(text), end+50)].lower()
        negative_indicators = [
            "don't", "dont", "not", "avoid", "never", "no experience",
            "not familiar", "unfamiliar", "don't know", "dont know"
        ]
        for indicator in negative_indicators:
            if indicator in context:
                return True
    return False

def detect_profile_types_from_text(*parts: str) -> List[str]:
    """
    Return a prioritized list of profile types detected inside the provided text parts.
    Multiple profile types may apply (e.g., Full Stack + JavaScript).
    Uses word boundary matching to avoid false positives.
    """
    text_blob = _normalize_text_blob(*parts)
    if not text_blob:
        return []
    
    matches = []
    for profile_type, keywords in PROFILE_TYPE_RULES:
        for keyword in keywords:
            if _count_keyword_matches(keyword, text_blob, profile_type) > 0:
                matches.append(profile_type)
                break
    return matches


def determine_profile_types_enhanced(
    primary_skills: str = "",
    secondary_skills: str = "",
    resume_text: str = "",
    ai_client=None,
    ai_model: str = None,
    min_confidence: float = 0.4,  # Increased from 0.3 to 0.4 for stricter filtering
    equal_score_threshold: float = 0.15,
    max_profiles: int = 2  # Reduced from 3 to 2 to prevent too many profiles
) -> Tuple[List[str], float, Dict[str, Any]]:
    """
    Enhanced profile type detection with multi-profile support.
    
    Args:
        primary_skills: Comma-separated primary skills
        secondary_skills: Comma-separated secondary skills
        resume_text: Full resume text content
        ai_client: Optional AI/LLM client
        ai_model: Optional AI model name
        min_confidence: Minimum confidence threshold (0.0-1.0)
        equal_score_threshold: Score difference ratio for equal profiles (0.0-1.0)
        max_profiles: Maximum number of profiles to return
        
    Returns:
        (profile_types, overall_confidence, metadata)
    """
    metadata = {'method': 'keyword', 'scores': {}, 'matched_keywords': {}}
    
    # AI/LLM detection disabled - always use keyword-based detection for consistency and accuracy
    # Keyword-based detection provides more reliable and consistent results for profile type classification
    
    # Keyword-based detection
    text_blob = _normalize_text_blob(primary_skills, secondary_skills, resume_text)
    if not text_blob:
        return ([DEFAULT_PROFILE_TYPE], 0.0, metadata)
    
    profile_scores = _calculate_normalized_scores(text_blob, primary_skills, secondary_skills, resume_text)
    
    if not profile_scores:
        return ([DEFAULT_PROFILE_TYPE], 0.0, metadata)
    
    # Store metadata
    for ps in profile_scores[:max_profiles]:
        metadata['scores'][ps.profile_type] = {
            'normalized': ps.normalized_score,
            'raw': ps.raw_score,
            'confidence': ps.confidence
        }
        metadata['matched_keywords'][ps.profile_type] = ps.matched_keywords
    
    # Filter by confidence
    valid_scores = [ps for ps in profile_scores if ps.confidence >= min_confidence]
    if not valid_scores:
        return ([DEFAULT_PROFILE_TYPE], 0.0, metadata)
    
    # Multi-profile logic with adaptive requirements based on confidence and score strength
    top_score = valid_scores[0]
    equal_profiles = [top_score.profile_type]
    top_normalized = top_score.normalized_score
    top_confidence = top_score.confidence
    
    # Log detailed scoring for debugging
    logger.info(f"Profile type scoring details:")
    for i, ps in enumerate(valid_scores[:max_profiles]):
        score_diff = top_normalized - ps.normalized_score if i > 0 else 0.0
        score_diff_ratio = (score_diff / top_normalized * 100) if top_normalized > 0 and i > 0 else 0.0
        score_ratio = (ps.normalized_score / top_normalized * 100) if top_normalized > 0 else 0.0
        logger.info(
            f"  {i+1}. {ps.profile_type}: "
            f"normalized={ps.normalized_score:.4f} ({score_ratio:.1f}% of top), "
            f"raw={ps.raw_score:.2f}, "
            f"confidence={ps.confidence:.3f}, "
            f"diff={score_diff_ratio:.1f}%, "
            f"keywords={ps.matched_keywords[:3]}"
        )
    
    # Adaptive multi-profile inclusion criteria based on top score strength and confidence
    # Tier 1: High confidence (>=0.75) or strong top score (>=0.7) → Single profile preferred
    # Tier 2: Moderate confidence (0.6-0.75) and moderate score (0.5-0.7) → Allow multi-profile with strict criteria
    # Tier 3: Low confidence (<0.6) or weak score (<0.5) → Single profile only (most reliable)
    
    if top_confidence >= 0.75 or top_normalized >= 0.7:
        # Tier 1: High confidence/strong score - very strict for second profile
        tier = "Tier 1 (High Confidence/Strong Score)"
        min_keywords_required = 5  # Increased from 4 to 5 - require more keywords for higher accuracy
        min_score_ratio = 0.75  # Increased from 0.65 to 0.75 - second must be at least 75% of top
        min_raw_score_for_inclusion = 30.0  # Increased from 25.0 to 30.0 - higher raw score required
        dominant_score_threshold = 0.15  # Increased from 0.01 to 0.15 (15%) - exclude if top is >15% higher
        min_second_confidence = 0.70  # Increased from 0.65 to 0.70 - higher confidence required
    elif top_confidence >= 0.6 and top_normalized >= 0.5:
        # Tier 2: Moderate confidence/score - moderate strictness
        tier = "Tier 2 (Moderate Confidence/Score)"
        min_keywords_required = 4  # Increased from 3 to 4 - require more keywords for higher accuracy
        min_score_ratio = 0.65  # Increased from 0.55 to 0.65 - second must be at least 65% of top
        min_raw_score_for_inclusion = 25.0  # Increased from 20.0 to 25.0 - higher raw score required
        dominant_score_threshold = 0.10  # Increased from 0.01 to 0.10 (10%) - exclude if top is >10% higher
        min_second_confidence = 0.60  # Increased from 0.55 to 0.60 - higher confidence required
    else:
        # Tier 3: Low confidence/weak score - single profile only
        tier = "Tier 3 (Low Confidence/Weak Score)"
        min_keywords_required = 999  # Effectively disables multi-profile
        min_score_ratio = 1.0
        min_raw_score_for_inclusion = 999.0
        dominant_score_threshold = 0.0
        min_second_confidence = 1.0
    
    logger.info(f"Using {tier} criteria for multi-profile inclusion")
    
    for score in valid_scores[1:max_profiles]:
        score_diff_ratio = (top_normalized - score.normalized_score) / top_normalized if top_normalized > 0 else 1.0
        score_ratio = score.normalized_score / top_normalized if top_normalized > 0 else 0.0
        
        # Count only real keywords (exclude phrase matches)
        real_keywords = [kw for kw in score.matched_keywords if not kw.startswith("phrase_match_")]
        keyword_count = len(real_keywords)
        
        # Check if top score is significantly dominant
        is_top_dominant = score_diff_ratio > dominant_score_threshold
        
        # Check profile type compatibility
        is_compatible = are_profile_types_compatible(top_score.profile_type, score.profile_type)
        
        # Check confidence requirement for second profile
        has_sufficient_confidence = score.confidence >= min_second_confidence
        
        # Check ALL strict inclusion criteria
        meets_diff_threshold = score_diff_ratio <= equal_score_threshold
        meets_min_ratio = score_ratio >= min_score_ratio
        has_significant_raw_score = score.raw_score >= min_raw_score_for_inclusion
        has_min_keywords = keyword_count >= min_keywords_required
        
        # Include ONLY if ALL conditions are met:
        # 1. Top score is not significantly dominant
        # 2. Profile types are compatible
        # 3. Second profile has sufficient confidence
        # 4. All other criteria met
        if (not is_top_dominant and is_compatible and has_sufficient_confidence and 
            meets_diff_threshold and meets_min_ratio and has_significant_raw_score and has_min_keywords):
            equal_profiles.append(score.profile_type)
            logger.info(
                f"  → Including {score.profile_type} "
                f"(keywords={keyword_count}, raw={score.raw_score:.1f}, "
                f"ratio={score_ratio*100:.1f}%, diff={score_diff_ratio*100:.1f}%, "
                f"confidence={score.confidence:.3f}, compatible={is_compatible})"
            )
        else:
            reason = []
            if is_top_dominant:
                reason.append(f"top score is dominant (diff {score_diff_ratio*100:.1f}% > {dominant_score_threshold*100:.1f}%)")
            if not is_compatible:
                reason.append(f"profile types incompatible ({top_score.profile_type} + {score.profile_type})")
            if not has_sufficient_confidence:
                reason.append(f"confidence {score.confidence:.3f} < {min_second_confidence:.3f}")
            if not has_min_keywords:
                reason.append(f"keywords {keyword_count} < {min_keywords_required}")
            if not has_significant_raw_score:
                reason.append(f"raw score {score.raw_score:.1f} < {min_raw_score_for_inclusion}")
            if not meets_min_ratio:
                reason.append(f"ratio {score_ratio*100:.1f}% < {min_score_ratio*100:.1f}%")
            if not meets_diff_threshold:
                reason.append(f"diff {score_diff_ratio*100:.1f}% > {equal_score_threshold*100:.1f}%")
            logger.info(f"  → Excluding {score.profile_type} ({', '.join(reason)})")
            break
    
    overall_confidence = sum(ps.confidence for ps in valid_scores[:len(equal_profiles)]) / len(equal_profiles)
    # Canonicalize and deduplicate profile types (remove duplicates while preserving order)
    canonical_profiles = []
    seen = set()
    for pt in equal_profiles:
        canonical = canonicalize_profile_type(pt)
        if canonical and canonical not in seen:
            canonical_profiles.append(canonical)
            seen.add(canonical)
    
    logger.info(
        f"Detected profiles: {canonical_profiles} "
        f"(scores: {[f'{ps.normalized_score:.2f}' for ps in valid_scores[:len(equal_profiles)]]}, "
        f"confidence: {overall_confidence:.2f})"
    )
    
    return (canonical_profiles, overall_confidence, metadata)

def determine_primary_profile_type(primary_skills: str = "", secondary_skills: str = "", resume_text: str = "", ai_client=None, ai_model: str = None) -> str:
    """
    Determine the canonical profile type (backward compatible - returns single profile).
    For multi-profile support, use determine_profile_types_enhanced().
    
    Args:
        primary_skills: Comma-separated primary skills
        secondary_skills: Comma-separated secondary skills  
        resume_text: Full resume text content
        ai_client: Optional AI/LLM client for intelligent analysis
        ai_model: Optional AI model name
        
    Returns:
        Canonical profile type string (comma-separated if multiple)
    """
    profile_types, confidence, _ = determine_profile_types_enhanced(
        primary_skills, secondary_skills, resume_text, ai_client, ai_model
    )
    # Return comma-separated string for backward compatibility
    return ", ".join(profile_types) if len(profile_types) > 1 else profile_types[0] if profile_types else DEFAULT_PROFILE_TYPE


def _determine_profile_type_with_llm_enhanced(
    primary_skills: str,
    secondary_skills: str,
    resume_text: str,
    ai_client,
    ai_model: str
) -> Optional[Tuple[List[str], float]]:
    """Enhanced LLM detection with multi-profile support."""
    skills_context = f"Primary Skills: {primary_skills}\nSecondary Skills: {secondary_skills}"
    resume_snippet = resume_text[:4000] if len(resume_text) > 4000 else resume_text
    profile_types_list = [pt for pt, _ in PROFILE_TYPE_RULES_ENHANCED] + [DEFAULT_PROFILE_TYPE]
    
    prompt = f"""Analyze the resume and determine profile types. Return comma-separated if multiple apply.

{skills_context}

Resume Content:
{resume_snippet}

Available Profile Types: {', '.join(profile_types_list)}

Examples:
- "Java, Spring Boot, Hibernate" → "Java"
- "C#, ASP.NET, Entity Framework, Java, Spring" → ".Net, Java"
- "React, Node.js, MongoDB" → "Full Stack"

Return format: Comma-separated profile types (e.g., ".Net, Java" or "Python")
If uncertain, return "{DEFAULT_PROFILE_TYPE}".
"""
    
    try:
        response = ai_client.chat.completions.create(
            model=ai_model,
            messages=[
                {"role": "system", "content": "Expert technical recruiter identifying profile types."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        result = response.choices[0].message.content.strip()
        result = result.strip('"\'').strip('.').strip()
        
        profiles = [p.strip() for p in result.split(',')]
        # Filter to only valid profile types (canonicalize returns DEFAULT_PROFILE_TYPE for invalid ones)
        canonical_profiles = [
            pt for p in profiles if p 
            for pt in [canonicalize_profile_type(p)]
            if pt != DEFAULT_PROFILE_TYPE
        ]
        
        # Limit LLM output to max 2 profiles to prevent too many profiles
        if canonical_profiles:
            limited_profiles = canonical_profiles[:2]  # Strict limit of 2 profiles
            return (limited_profiles, 0.85)
        
    except Exception as e:
        logger.error(f"LLM profile type determination failed: {e}")
    
    return None

def _determine_profile_type_with_llm(primary_skills: str, secondary_skills: str, resume_text: str, ai_client, ai_model: str) -> str:
    """
    Use LLM to analyze overall resume content and determine profile type.
    """
    
    # Prepare context for LLM analysis
    skills_context = f"Primary Skills: {primary_skills}\nSecondary Skills: {secondary_skills}"
    resume_snippet = resume_text[:4000] if len(resume_text) > 4000 else resume_text  # Limit text for token efficiency
    
    profile_types_list = [pt for pt, _ in PROFILE_TYPE_RULES] + [DEFAULT_PROFILE_TYPE]
    
    prompt = f"""Analyze the following resume content and skill set to determine the candidate's primary profile type.

{skills_context}

Resume Content (snippet):
{resume_snippet}

Available Profile Types:
{', '.join(profile_types_list)}

Instructions:
1. Analyze the overall resume content, skills, experience, and context
2. Consider the dominant technology stack, frameworks, and tools mentioned
3. Identify the PRIMARY profile type that best describes this candidate
4. Consider skill weights - if candidate has C#, ASP.NET, .NET Core, ADO.NET, Entity Framework - they are clearly a .NET developer, not Java
5. Return ONLY the profile type name (one of the available types), nothing else

Return format: Just the profile type name (e.g., ".Net", "Java", "Python", etc.)
If uncertain, return "{DEFAULT_PROFILE_TYPE}".
"""
    
    try:
        response = ai_client.chat.completions.create(
            model=ai_model,
            messages=[
                {"role": "system", "content": "You are an expert technical recruiter who accurately identifies candidate profile types based on resume analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent results
            max_tokens=50
        )
        
        result = response.choices[0].message.content.strip()
        # Clean up the response - remove quotes, periods, etc.
        result = result.strip('"\'').strip('.').strip()
        
        logger.info(f"LLM determined profile type: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in LLM profile type determination: {e}")
        return None


class ProfileScore(NamedTuple):
    """Profile type scoring result."""
    profile_type: str
    raw_score: float
    normalized_score: float
    confidence: float
    matched_keywords: List[str]
    keyword_details: List[Tuple[str, float, str]]  # (keyword, score, location)

def _extract_first_lines_of_skills(primary_skills: str, num_lines: int = 2) -> str:
    """Extract first 1-2 lines of technical skills for higher weightage."""
    if not primary_skills:
        return ""
    
    # Split by comma and take first few skills (assuming comma-separated)
    skills_list = [s.strip() for s in primary_skills.split(',')]
    if len(skills_list) <= num_lines:
        return primary_skills
    
    # Return first 1-2 skills
    return ', '.join(skills_list[:num_lines])

def _detect_phrases(text: str) -> Dict[str, float]:
    """Detect specific phrases that indicate profile types with high confidence."""
    text_lower = text.lower()
    phrase_scores = {}
    
    # Python phrases
    python_phrases = [
        (r"with solid foundation.*?python", 8.0),
        (r"strong experience.*?python", 8.0),
        (r"solid foundation in python", 8.0),
        (r"strong experience in python", 8.0),
        (r"expertise.*?python", 6.0),
        (r"proficient.*?python", 6.0),
    ]
    
    for pattern, weight in python_phrases:
        if re.search(pattern, text_lower, re.IGNORECASE):
            phrase_scores['Python'] = max(phrase_scores.get('Python', 0.0), weight)
    
    # Java phrases
    java_phrases = [
        (r"with solid foundation.*?java", 8.0),
        (r"strong experience.*?java", 8.0),
        (r"solid foundation in java", 8.0),
        (r"strong experience in java", 8.0),
        (r"expertise.*?java", 6.0),
        (r"proficient.*?java", 6.0),
    ]
    
    for pattern, weight in java_phrases:
        if re.search(pattern, text_lower, re.IGNORECASE):
            phrase_scores['Java'] = max(phrase_scores.get('Java', 0.0), weight)
    
    return phrase_scores

def _check_business_development(text: str) -> bool:
    """Check if profile indicates Business Development role."""
    text_lower = text.lower()
    bd_keywords = [
        "business development", "bd", "business dev", "bde",
        "business development executive", "business development manager"
    ]
    return any(keyword in text_lower for keyword in bd_keywords)

def _calculate_normalized_scores(
    text_blob: str,
    primary_skills: str,
    secondary_skills: str,
    resume_text: str
) -> List[ProfileScore]:
    """Calculate normalized scores with confidence for all profile types."""
    profile_scores = []
    primary_lower = primary_skills.lower()
    secondary_lower = secondary_skills.lower()
    
    # Extract first 1-2 lines of skills for higher weightage
    first_lines_skills = _extract_first_lines_of_skills(primary_skills, num_lines=2)
    first_lines_lower = first_lines_skills.lower()
    
    # Detect specific phrases
    phrase_scores = _detect_phrases(resume_text + " " + primary_skills)
    
    # Check for Business Development - if present, prioritize it
    is_business_dev = _check_business_development(resume_text + " " + primary_skills)
    
    for profile_type, keyword_weights in PROFILE_TYPE_RULES_ENHANCED:
        # If Business Development detected, only process that profile type
        if is_business_dev and profile_type != "Business Development":
            continue
        if is_business_dev and profile_type == "Business Development":
            # Give very high score for Business Development
            profile_scores.append(ProfileScore(
                profile_type="Business Development",
                raw_score=50.0,  # Very high score
                normalized_score=1.0,
                confidence=0.95,
                matched_keywords=["business development"],
                keyword_details=[("business development", 50.0, "resume")]
            ))
            continue
        
        raw_score = 0.0
        matched_keywords = []
        keyword_details = []
        
        for keyword, base_weight in keyword_weights.items():
            if _has_negative_context(keyword, text_blob, profile_type):
                continue
            
            count = _count_keyword_matches(keyword, text_blob, profile_type)
            if count == 0:
                continue
            
            location_multiplier = 1.0
            # Check if keyword is in first 1-2 lines of skills (highest priority)
            if keyword.lower() in first_lines_lower:
                location_multiplier = 5.0  # Highest weight for first lines
            elif keyword.lower() in primary_lower:
                location_multiplier = 3.0
            elif keyword.lower() in secondary_lower:
                location_multiplier = 2.0
            
            keyword_score = count * base_weight * location_multiplier
            raw_score += keyword_score
            
            matched_keywords.append(keyword)
            location = ("first_lines" if location_multiplier == 5.0
                       else "primary" if location_multiplier == 3.0 
                       else "secondary" if location_multiplier == 2.0 
                       else "resume")
            keyword_details.append((keyword, keyword_score, location))
        
        # Add phrase-based bonus
        if profile_type in phrase_scores:
            phrase_bonus = phrase_scores[profile_type]
            raw_score += phrase_bonus
            matched_keywords.append(f"phrase_match_{profile_type.lower()}")
            keyword_details.append((f"phrase_match", phrase_bonus, "resume"))
        
        if raw_score > 0:
            # Max possible uses 5.0 multiplier to account for first_lines weightage
            max_possible = sum(keyword_weights.values()) * 5.0
            normalized_score = min(1.0, raw_score / max_possible) if max_possible > 0 else 0.0
            confidence = _calculate_confidence(
                normalized_score, len(matched_keywords), len(keyword_weights), keyword_details
            )
            profile_scores.append(ProfileScore(
                profile_type=profile_type,
                raw_score=raw_score,
                normalized_score=normalized_score,
                confidence=confidence,
                matched_keywords=matched_keywords,
                keyword_details=keyword_details
            ))
    
    return sorted(profile_scores, key=lambda x: x.normalized_score, reverse=True)

def _calculate_confidence(
    normalized_score: float,
    matched_count: int,
    total_keywords: int,
    keyword_details: List[Tuple[str, float, str]]
) -> float:
    """Calculate confidence score (0.0 to 1.0)."""
    base_confidence = normalized_score
    coverage_ratio = matched_count / total_keywords if total_keywords > 0 else 0
    coverage_bonus = min(0.2, coverage_ratio * 0.2)
    
    primary_bonus = 0.0
    for _, score, location in keyword_details:
        if location == "primary" and score > 5.0:
            primary_bonus += 0.1
    
    confidence = min(1.0, base_confidence + coverage_bonus + min(0.1, primary_bonus))
    return round(confidence, 3)

def _determine_profile_type_with_keywords(primary_skills: str, secondary_skills: str, resume_text: str) -> str:
    """
    Fallback: Weighted keyword-based profile type detection (legacy method).
    Uses enhanced scoring but returns single profile for backward compatibility.
    """
    profile_types, confidence, _ = determine_profile_types_enhanced(
        primary_skills, secondary_skills, resume_text, None, None
    )
    return profile_types[0] if profile_types else DEFAULT_PROFILE_TYPE

def format_profile_types_for_storage(profile_types: List[str]) -> str:
    """Format profile types for database storage (comma-separated)."""
    if not profile_types:
        return DEFAULT_PROFILE_TYPE
    
    # Filter to only valid profile types (exclude DEFAULT_PROFILE_TYPE and invalid entries)
    canonical = [
        pt for profile_type in profile_types
        for pt in [canonicalize_profile_type(profile_type)]
        if pt != DEFAULT_PROFILE_TYPE
    ]
    
    if not canonical:
        return DEFAULT_PROFILE_TYPE
    
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for pt in canonical:
        if pt not in seen:
            seen.add(pt)
            unique.append(pt)
    
    # Use comma without space for MySQL FIND_IN_SET compatibility
    # Stored format: "Microsoft Power Platform,Integration / APIs"
    return ",".join(unique)


def infer_profile_type_from_requirements(required_skills: List[str], jd_text: str = "") -> List[str]:
    """
    Infer one or more target profile types from job requirements / JD text.
    """
    skill_blob = ", ".join(required_skills or [])
    combined = _normalize_text_blob(skill_blob, jd_text)
    matches = detect_profile_types_from_text(combined)
    # Deduplicate while preserving order
    seen = set()
    ordered_matches = []
    for match in matches:
        if match not in seen:
            ordered_matches.append(match)
            seen.add(match)
    return ordered_matches

