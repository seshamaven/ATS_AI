"""
Utility helpers to derive and reuse candidate profile types (Java, .Net, SAP, etc.).

The goal is to keep a single source of truth for profile type detection so that
resume parsing, SQL filtering, and search/ranking code stay in sync.
"""

from typing import Iterable, List, Optional

# Ordered list to keep deterministic precedence (first match wins)
PROFILE_TYPE_RULES = [
    ("Java", {"java", "spring", "spring boot", "hibernate", "j2ee", "maven", "gradle"}),
    (".Net", {".net", "dotnet", "c#", "asp.net", "entity framework", "wpf", "winforms"}),
    ("Python", {"python", "django", "flask", "fastapi", "pandas"}),
    ("JavaScript", {"javascript", "node", "node.js", "react", "angular", "vue", "typescript"}),
    ("Full Stack", {"full stack", "mern", "mean", "next.js", "nuxt"}),
    ("DevOps", {"devops", "kubernetes", "docker", "terraform", "ansible", "cicd"}),
    ("Data Engineering", {"data engineer", "etl", "snowflake", "spark", "hadoop", "databricks"}),
    ("Data Science", {"machine learning", "data science", "ml engineer", "ai engineer", "llm"}),
    ("Testing / QA", {"qa", "quality assurance", "automation testing", "selenium", "cypress"}),
    ("SAP", {"sap", "abap", "hana", "successfactors", "ariba"}),
    ("ERP", {"erp", "oracle apps", "oracle e-business", "d365", "dynamics", "netsuite", "workday"}),
    ("Cloud / Infra", {"aws", "azure", "gcp", "cloud architect", "cloud engineer"}),
]

DEFAULT_PROFILE_TYPE = "Generalist"


def canonicalize_profile_type(value: Optional[str]) -> str:
    """Normalize profile type labels to a consistent, canonical form."""
    if not value:
        return DEFAULT_PROFILE_TYPE
    
    normalized = str(value).strip()
    if not normalized:
        return DEFAULT_PROFILE_TYPE
    
    lowered = normalized.lower()
    for profile_type, _ in PROFILE_TYPE_RULES:
        if lowered == profile_type.lower():
            return profile_type
    
    # Keep known default if user explicitly passed it
    if lowered == DEFAULT_PROFILE_TYPE.lower():
        return DEFAULT_PROFILE_TYPE
    
    # Otherwise Title-case the custom entry to avoid duplicate variants
    return normalized.title()


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


def _normalize_text_blob(*parts: str) -> str:
    """Lower-case, concatenated blob for keyword detection."""
    normalized_parts = []
    for part in parts:
        if not part:
            continue
        normalized_parts.append(str(part).lower())
    return " ".join(normalized_parts)


def detect_profile_types_from_text(*parts: str) -> List[str]:
    """
    Return a prioritized list of profile types detected inside the provided text parts.
    Multiple profile types may apply (e.g., Full Stack + JavaScript).
    """
    text_blob = _normalize_text_blob(*parts)
    if not text_blob:
        return []
    
    matches = []
    for profile_type, keywords in PROFILE_TYPE_RULES:
        for keyword in keywords:
            if keyword in text_blob:
                matches.append(profile_type)
                break
    return matches


def determine_primary_profile_type(primary_skills: str = "", secondary_skills: str = "", resume_text: str = "") -> str:
    """
    Determine the canonical profile type for storage on the candidate record.
    """
    matches = detect_profile_types_from_text(primary_skills, secondary_skills, resume_text)
    if matches:
        return canonicalize_profile_type(matches[0])
    
    # Fallback to the first primary skill token if available
    if primary_skills:
        first_token = primary_skills.split(",")[0].strip()
        if first_token:
            return canonicalize_profile_type(first_token)
    
    return DEFAULT_PROFILE_TYPE


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

