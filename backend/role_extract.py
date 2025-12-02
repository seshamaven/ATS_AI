"""
Lightweight utilities to work with high-level roles and subroles.

This module is intentionally **category-agnostic** â€“ it only knows about
`role` and `subrole` (no category column).
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Sequence, Tuple, Dict

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Static role / subrole definitions
# -----------------------------------------------------------------------------

# NOTE: Legacy ROLE_SUBROLE_PAIRS removed; roles now come from ROLE_HIERARCHY
# and subroles are derived from skills, not a static mapping.
ROLE_SUBROLE_PAIRS: List[Tuple[str, str]] = []


# -----------------------------------------------------------------------------
# Role Hierarchy (Highest to Lowest Priority)
# -----------------------------------------------------------------------------

ROLE_HIERARCHY = [
    # Level 0: Highest (Executive/Director)
    ["Technical Director", "Vice President", "VP", "Director", "Director Technology", "Director - Technology"],
    # Level 2: Senior Management
    ["Delivery Manager", "Program Manager"],
    # Level 3: Management
    ["Sr Manager", "Senior Manager", "Account Manager", "Sr Project Manager", "Senior Project Manager"],
    # Level 4: Project Management
    ["Project Manager"],
    # Level 5: Lead Roles
    ["Project Lead", "Technical Lead", "Tech Lead", "Module Lead", "Team Lead", "Lead"],
    # Level 6: Analyst
    ["Analyst", "Business Analyst", "Data Analyst", "System Analyst"],
    # Level 7: Senior Engineer
    ["Sr Software Engineer", "Senior Software Engineer", "Sr Engineer", "Senior Engineer"],
    # Level 8: Engineer (Lowest)
    ["Software Engineer", "Engineer", "Developer"],
]

def get_role_priority(role_or_subrole: str) -> int:
    """
    Get priority level for a role/subrole (lower number = higher priority).
    Returns 999 if not found in hierarchy.
    """
    text_lower = role_or_subrole.lower()
    for priority, role_list in enumerate(ROLE_HIERARCHY):
        for role in role_list:
            if role.lower() in text_lower or text_lower in role.lower():
                return priority
    return 999  # Not in hierarchy - lowest priority


# -----------------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------------

def list_roles() -> List[str]:
    """Return the distinct list of high-level roles."""
    return sorted({role for role, _ in ROLE_SUBROLE_PAIRS})


def list_subroles(role: Optional[str] = None) -> List[str]:
    """
    Return all subroles, or subroles for a given role if provided.
    """
    if role is None:
        return sorted({sub for _, sub in ROLE_SUBROLE_PAIRS})
    return sorted({sub for r, sub in ROLE_SUBROLE_PAIRS if r.lower() == role.lower()})


def _normalize(text: str) -> str:
    """Normalize text for matching: lowercase, remove special chars, normalize whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[/,&]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_role_subrole(
    title_or_text: str,
    candidates: Sequence[Tuple[str, str]] = ROLE_SUBROLE_PAIRS,
) -> Optional[Tuple[str, str]]:
    """
    Detect role and subrole from text using exact values from ROLE_SUBROLE_PAIRS.
    When multiple matches are found, returns the highest priority role based on hierarchy.
    
    Matching strategy:
    1. Finds ALL matching roles/subroles
    2. Selects the one with highest priority (lowest hierarchy number)
    3. Returns exact values from ROLE_SUBROLE_PAIRS
    
    Returns exact values from ROLE_SUBROLE_PAIRS - no modifications.
    Category is deliberately ignored (only role and subrole are used).
    
    Args:
        title_or_text: Job title, designation, or text to search
        candidates: List of (role, subrole) tuples to match against
        
    Returns:
        Tuple of (role, subrole) if match found, None otherwise
    """
    if not title_or_text:
        return None

    text_norm = _normalize(title_or_text)
    matches = []  # List of (priority, role, subrole) tuples

    # Strategy 1: Check hierarchy first for high-priority roles (Director, VP, etc.)
    # This ensures "Director - Technology" matches before generic roles
    for priority, hierarchy_roles in enumerate(ROLE_HIERARCHY):
        for hierarchy_role in hierarchy_roles:
            hierarchy_norm = _normalize(hierarchy_role)
            # Check if hierarchy role appears in text
            if hierarchy_norm and hierarchy_norm in text_norm:
                # Find matching role/subrole pair that contains this hierarchy role
                for role, subrole in candidates:
                    role_norm = _normalize(role)
                    subrole_norm = _normalize(subrole)
                    if hierarchy_norm in role_norm or hierarchy_norm in subrole_norm:
                        matches.append((priority, role, subrole))
                        logger.debug(f"Matched hierarchy role '{hierarchy_role}' -> '{role}' / '{subrole}' (priority: {priority})")
                        break
                # If found a high-priority match, break early
                if matches and matches[-1][0] == 0:  # Priority 0 = highest
                    break
        if matches and matches[-1][0] == 0:  # Found highest priority, stop searching
            break

    # Strategy 2: Look for exact subrole phrase matches (more specific)
    for role, subrole in candidates:
        subrole_norm = _normalize(subrole)
        if subrole_norm and subrole_norm in text_norm:
            priority = get_role_priority(subrole)
            # Only add if not already found in hierarchy search
            if not any(r == role and s == subrole for _, r, s in matches):
                matches.append((priority, role, subrole))
                logger.debug(f"Matched subrole '{subrole}' (role: '{role}', priority: {priority})")

    # Strategy 3: Look for role phrase matches (less specific)
    for role, subrole in candidates:
        role_norm = _normalize(role)
        if role_norm and role_norm in text_norm:
            priority = get_role_priority(role)
            # Only add if not already found
            if not any(r == role and s == subrole for _, r, s in matches):
                matches.append((priority, role, subrole))
                logger.debug(f"Matched role '{role}' (subrole: '{subrole}', priority: {priority})")

    if not matches:
        logger.debug("No role/subrole match for text: %r", title_or_text[:100])
        return None

    # Sort by priority (lower number = higher priority) and return the highest
    matches.sort(key=lambda x: x[0])
    best_priority, best_role, best_subrole = matches[0]
    
    if len(matches) > 1:
        logger.debug(f"Found {len(matches)} role matches, selected highest priority: '{best_role}' / '{best_subrole}' (priority: {best_priority})")
    
    return best_role, best_subrole


def detect_all_roles(title_or_text: str) -> List[Tuple[int, str, str]]:
    """
    Find ALL roles in text using ROLE_HIERARCHY and return with priorities.
    Returns list of (priority, role, subrole) tuples sorted by priority.
    Subrole will be determined later from skills, so we return None for subrole here.
    """
    if not title_or_text:
        return []
    
    text_norm = _normalize(title_or_text)
    matches = []  # List of (priority, role, subrole) tuples
    
    # Map hierarchy roles to role_type names
    # This maps job titles from hierarchy to the actual role_type we want to store
    hierarchy_to_role_map = {
        # Level 0: Executive/Director -> Management & Product
        "technical director": "Management & Product",
        "vice president": "Management & Product",
        "vp": "Management & Product",
        "director": "Management & Product",
        "director technology": "Management & Product",
        "director - technology": "Management & Product",
        # Level 2: Senior Management -> Management & Product
        "delivery manager": "Management & Product",
        "program manager": "Management & Product",
        # Level 3: Management -> Management & Product
        "sr manager": "Management & Product",
        "senior manager": "Management & Product",
        "account manager": "Management & Product",
        "sr project manager": "Management & Product",
        "senior project manager": "Management & Product",
        # Level 4: Project Management -> Management & Product
        "project manager": "Management & Product",
        # Level 5: Lead Roles -> Software Engineer (or Management & Product for some)
        "project lead": "Software Engineer",
        "technical lead": "Software Engineer",
        "tech lead": "Software Engineer",
        "module lead": "Software Engineer",
        "team lead": "Software Engineer",
        "lead": "Software Engineer",
        # Level 6: Analyst -> Business Analyst or Data Analyst
        "analyst": "Business Analyst",
        "business analyst": "Business Analyst",
        "data analyst": "Data Analyst",
        "system analyst": "Business Analyst",
        # Level 7: Senior Engineer -> Software Engineer
        "sr software engineer": "Software Engineer",
        "senior software engineer": "Software Engineer",
        "sr engineer": "Software Engineer",
        "senior engineer": "Software Engineer",
        # Level 8: Engineer -> Software Engineer
        "software engineer": "Software Engineer",
        "engineer": "Software Engineer",
        "developer": "Software Engineer",
    }
    
    # Check hierarchy roles directly
    for priority, hierarchy_roles in enumerate(ROLE_HIERARCHY):
        for hierarchy_role in hierarchy_roles:
            hierarchy_norm = _normalize(hierarchy_role)
            if hierarchy_norm and hierarchy_norm in text_norm:
                # Map hierarchy role to role_type
                role_type = hierarchy_to_role_map.get(hierarchy_norm, "Software Engineer")
                # Subrole will be determined from skills later, so use None here
                if not any(r == role_type for _, r, _ in matches):
                    matches.append((priority, role_type, None))
    
    # Sort by priority (lower = higher priority)
    matches.sort(key=lambda x: x[0])
    return matches


def match_subrole_from_skills(role: str, primary_skills: str, secondary_skills: str = "") -> Optional[str]:
    """
    Match subrole based on skills. Always returns one of: "Backend Developer", 
    "Full Stack Developer", or "Frontend Developer".
    
    Args:
        role: The selected role
        primary_skills: Comma-separated primary/technical skills
        secondary_skills: Comma-separated secondary skills
        
    Returns:
        One of: "Backend Developer", "Full Stack Developer", or "Frontend Developer"
    """
    if not primary_skills:
        return "Backend Developer"  # Default
    
    # Combine all skills for matching
    all_skills_text = f"{primary_skills}, {secondary_skills}".lower()
    all_skills_list = [s.strip().lower() for s in all_skills_text.split(',') if s.strip()]
    
    # Define skill keywords for each subrole type
    frontend_keywords = [
        'react', 'angular', 'vue', 'javascript', 'typescript', 'html', 'css', 
        'sass', 'scss', 'less', 'jsx', 'tsx', 'next.js', 'nuxt', 'gatsby',
        'webpack', 'vite', 'frontend', 'front-end', 'ui', 'ux', 'jquery',
        'bootstrap', 'tailwind', 'material-ui', 'ant design'
    ]
    
    backend_keywords = [
        'node.js', 'nodejs', 'express', 'django', 'flask', 'fastapi', 'spring',
        'spring boot', 'hibernate', 'api', 'rest', 'graphql', 'microservices',
        'server', 'backend', 'back-end', 'serverless', 'lambda', 'asp.net',
        '.net', 'c#', 'java', 'python', 'php', 'ruby', 'rails', 'go', 'golang',
        'rust', 'scala', 'kotlin', 'database', 'sql', 'nosql', 'mongodb',
        'postgresql', 'mysql', 'redis', 'elasticsearch', 'kafka', 'rabbitmq'
    ]
    
    # Count matches for frontend and backend
    frontend_score = sum(1 for keyword in frontend_keywords if keyword in all_skills_text)
    backend_score = sum(1 for keyword in backend_keywords if keyword in all_skills_text)
    
    # Determine subrole based on scores
    if frontend_score > 0 and backend_score > 0:
        # Has both frontend and backend skills -> Full Stack
        return "Full Stack Developer"
    elif frontend_score > backend_score:
        # More frontend skills -> Frontend
        return "Frontend Developer"
    else:
        # More backend skills or equal -> Backend (default)
        return "Backend Developer"


def determine_subrole_type_from_profile_and_skills(
    profile_type: str,
    primary_skills: str,
    secondary_skills: str = "",
) -> str:
    """
    Determine subrole_type based on profile_type and skills.
    Always returns one of: "Backend Developer", "Full Stack Developer", or "Frontend Developer".
    This is used to populate subrole_type (not sub_profile_type).
    """
    if not profile_type or not primary_skills:
        return "Backend Developer"  # Default
    
    # Combine all skills for matching
    all_skills_text = f"{primary_skills}, {secondary_skills}".lower()
    
    # Define skill keywords
    frontend_keywords = [
        'react', 'angular', 'vue', 'javascript', 'typescript', 'html', 'css', 
        'sass', 'scss', 'less', 'jsx', 'tsx', 'next.js', 'nuxt', 'gatsby',
        'webpack', 'vite', 'frontend', 'front-end', 'ui', 'ux', 'jquery',
        'bootstrap', 'tailwind', 'material-ui', 'ant design'
    ]
    
    backend_keywords = [
        'node.js', 'nodejs', 'express', 'django', 'flask', 'fastapi', 'spring',
        'spring boot', 'hibernate', 'api', 'rest', 'graphql', 'microservices',
        'server', 'backend', 'back-end', 'serverless', 'lambda', 'asp.net',
        '.net', 'c#', 'java', 'python', 'php', 'ruby', 'rails', 'go', 'golang',
        'rust', 'scala', 'kotlin', 'database', 'sql', 'nosql', 'mongodb',
        'postgresql', 'mysql', 'redis', 'elasticsearch', 'kafka', 'rabbitmq'
    ]
    
    # Count matches for frontend and backend skills
    frontend_score = sum(1 for keyword in frontend_keywords if keyword in all_skills_text)
    backend_score = sum(1 for keyword in backend_keywords if keyword in all_skills_text)
    
    # Normalize profile_type for comparison
    profile_lower = profile_type.lower()
    
    # Categorize profile types
    backend_profiles = ['java', '.net', 'python', 'c#', 'php', 'ruby', 'go', 'golang', 'rust', 'scala', 'kotlin']
    frontend_profiles = ['javascript', 'typescript', 'ui/ux', 'ui', 'ux', 'html', 'css']
    # Non-technical profiles that shouldn't get developer subroles
    non_technical_profiles = ['business development', 'sales', 'marketing', 'hr', 'recruitment', 'accounting', 'finance', 'legal', 'support', 'operations']
    
    # Check if profile_type is backend-focused
    is_backend_profile = any(bp in profile_lower for bp in backend_profiles)
    # Check if profile_type is frontend-focused
    is_frontend_profile = any(fp in profile_lower for fp in frontend_profiles)
    # Check if profile_type is non-technical
    is_non_technical = any(ntp in profile_lower for ntp in non_technical_profiles)
    
    # Determine subrole_type based on profile_type and skills
    if is_backend_profile:
        # Backend profile (Java, .Net, Python, etc.)
        if frontend_score > 0:
            # Has frontend skills -> Full Stack
            return "Full Stack Developer"
        else:
            # No frontend skills -> Backend
            return "Backend Developer"
    elif is_frontend_profile:
        # Frontend profile (JavaScript, UI/UX, etc.)
        if backend_score > 0:
            # Has backend skills -> Full Stack
            return "Full Stack Developer"
        else:
            # No backend skills -> Frontend
            return "Frontend Developer"
    elif is_non_technical:
        # Non-technical profiles - default to Backend Developer (or could be None, but we need one of the three)
        # If they have strong technical skills, determine from skills
        if frontend_score > 0 and backend_score > 0:
            return "Full Stack Developer"
        elif frontend_score > backend_score and frontend_score >= 2:  # Need at least 2 frontend skills
            return "Frontend Developer"
        else:
            return "Backend Developer"  # Default for non-technical
    else:
        # Other technical profiles (Data Science, DevOps, etc.) - determine from skills
        if frontend_score > 0 and backend_score > 0:
            return "Full Stack Developer"
        elif frontend_score > backend_score:
            return "Frontend Developer"
        else:
            return "Backend Developer"


def is_non_it_profile(profile_type: str) -> bool:
    """
    Check if a profile_type is non-IT related (Business Development, Sales, Marketing, etc.).
    
    Args:
        profile_type: The profile type to check
        
    Returns:
        True if non-IT profile, False otherwise
    """
    if not profile_type:
        return False
    
    profile_lower = profile_type.lower()
    
    # Non-IT profile types
    non_it_profiles = [
        'business development', 'sales', 'marketing', 'hr', 'human resources',
        'recruitment', 'recruiter', 'talent acquisition', 'accounting', 'finance',
        'financial', 'legal', 'support', 'operations', 'customer service',
        'customer support', 'admin', 'administration', 'executive assistant',
        'business analyst', 'functional analyst', 'process analyst', 'product analyst',
        'teacher', 'lecturer', 'professor', 'education', 'doctor', 'physician',
        'nurse', 'medical', 'healthcare', 'banking', 'insurance'
    ]
    
    return any(ntp in profile_lower for ntp in non_it_profiles)


def infer_role_from_skills(primary_skills: str, secondary_skills: str = "", profile_type: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """
    Infer role and subrole purely from skills (fallback when no role found in resume).
    Subrole is always one of: "Backend Developer", "Full Stack Developer", or "Frontend Developer".
    
    Args:
        primary_skills: Comma-separated primary/technical skills
        secondary_skills: Comma-separated secondary skills
        profile_type: Optional profile type to check if non-IT
        
    Returns:
        Tuple of (role, subrole) if inferred, None otherwise (returns None for non-IT profiles)
    """
    # If profile_type is non-IT, don't infer IT roles
    if profile_type and is_non_it_profile(profile_type):
        return None
    
    if not primary_skills:
        return None
    
    all_skills_text = f"{primary_skills}, {secondary_skills}".lower()
    all_skills_list = [s.strip().lower() for s in all_skills_text.split(',') if s.strip()]
    
    # Determine subrole first (always one of the three)
    frontend_keywords = [
        'react', 'angular', 'vue', 'javascript', 'typescript', 'html', 'css', 
        'sass', 'scss', 'less', 'jsx', 'tsx', 'next.js', 'nuxt', 'gatsby',
        'webpack', 'vite', 'frontend', 'front-end', 'ui', 'ux', 'jquery',
        'bootstrap', 'tailwind', 'material-ui', 'ant design'
    ]
    
    backend_keywords = [
        'node.js', 'nodejs', 'express', 'django', 'flask', 'fastapi', 'spring',
        'spring boot', 'hibernate', 'api', 'rest', 'graphql', 'microservices',
        'server', 'backend', 'back-end', 'serverless', 'lambda', 'asp.net',
        '.net', 'c#', 'java', 'python', 'php', 'ruby', 'rails', 'go', 'golang',
        'rust', 'scala', 'kotlin', 'database', 'sql', 'nosql', 'mongodb',
        'postgresql', 'mysql', 'redis', 'elasticsearch', 'kafka', 'rabbitmq'
    ]
    
    frontend_score = sum(1 for keyword in frontend_keywords if keyword in all_skills_text)
    backend_score = sum(1 for keyword in backend_keywords if keyword in all_skills_text)
    
    # Determine subrole
    if frontend_score > 0 and backend_score > 0:
        subrole = "Full Stack Developer"
    elif frontend_score > backend_score:
        subrole = "Frontend Developer"
    else:
        subrole = "Backend Developer"
    
    # Skill-based role inference patterns (role only, subrole is already determined)
    skill_patterns = {
        # Data Science / ML
        'Data Scientist': ['python', 'machine learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn', 'keras', 'neural network', 'deep learning', 'ai', 'artificial intelligence', 'data science'],
        'Data Engineer': ['etl', 'data pipeline', 'airflow', 'spark', 'hadoop', 'kafka', 'snowflake', 'data engineering'],
        'Data Analyst': ['data analysis', 'pandas', 'numpy', 'sql', 'excel', 'tableau', 'power bi', 'bi', 'business intelligence'],
        
        # Software Engineering
        'Software Engineer': ['.net', 'c#', 'csharp', 'asp.net', 'dotnet', 'java', 'spring', 'hibernate', 'python', 'django', 'flask', 'javascript', 'typescript', 'react', 'angular', 'vue', 'node.js', 'express', 'software', 'developer', 'programming'],
        
        # DevOps / Cloud
        'DevOps Engineer': ['devops', 'ci/cd', 'jenkins', 'gitlab', 'docker', 'kubernetes', 'terraform', 'ansible', 'chef', 'puppet'],
        'Cloud Engineer': ['aws', 'azure', 'gcp', 'cloud', 'ec2', 's3', 'lambda', 'cloud computing'],
        
        # QA / Testing
        'QA Engineer': ['selenium', 'cypress', 'test automation', 'qa', 'testing', 'junit', 'pytest', 'quality assurance'],
        
        # Database
        'Database Administrator': ['sql', 'mysql', 'postgresql', 'oracle', 'database', 'dba', 'database administration'],
        
        # Mobile
        'Mobile Developer': ['android', 'ios', 'flutter', 'react native', 'mobile', 'swift', 'kotlin', 'xamarin'],
        
        # SAP / ERP
        'SAP Consultant': ['sap', 'abap', 'hana', 'sap fico', 'sap mm', 'sap sd'],
    }
    
    # Score each role pattern
    role_scores = []
    for role, keywords in skill_patterns.items():
        score = sum(10 if keyword in all_skills_text else 0 for keyword in keywords)
        if score > 0:
            role_scores.append((score, role))
    
    if not role_scores:
        # Don't default to Software Engineer - return None if no IT skills found
        # This prevents assigning IT roles to non-IT profiles
        return None
    
    # Return highest scoring role with the determined subrole
    role_scores.sort(key=lambda x: x[0], reverse=True)
    return role_scores[0][1], subrole


def detect_role_only(title_or_text: str) -> Optional[str]:
    """Convenience helper to get just the role from text."""
    result = detect_role_subrole(title_or_text)
    return result[0] if result else None


def detect_subrole_only(title_or_text: str) -> Optional[str]:
    """Convenience helper to get just the subrole from text."""
    result = detect_role_subrole(title_or_text)
    return result[1] if result else None


__all__ = [
    "ROLE_SUBROLE_PAIRS",
    "ROLE_HIERARCHY",
    "list_roles",
    "list_subroles",
    "detect_role_subrole",
    "detect_all_roles",
    "match_subrole_from_skills",
    "determine_subrole_type_from_profile_and_skills",
    "is_non_it_profile",
    "infer_role_from_skills",
    "detect_role_only",
    "detect_subrole_only",
]


