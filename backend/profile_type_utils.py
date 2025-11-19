"""
Utility helpers to derive and reuse candidate profile types (Java, .Net, SAP, etc.).

The goal is to keep a single source of truth for profile type detection so that
resume parsing, SQL filtering, and search/ranking code stay in sync.
"""

import logging
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

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


def determine_primary_profile_type(primary_skills: str = "", secondary_skills: str = "", resume_text: str = "", ai_client=None, ai_model: str = None) -> str:
    """
    Determine the canonical profile type using NLM (Natural Language Model) approach.
    Analyzes overall resume content, skill set, and context to determine the most appropriate profile type.
    
    Uses LLM-based analysis if AI client is available, otherwise falls back to weighted keyword matching.
    
    Args:
        primary_skills: Comma-separated primary skills
        secondary_skills: Comma-separated secondary skills  
        resume_text: Full resume text content
        ai_client: Optional AI/LLM client for intelligent analysis
        ai_model: Optional AI model name
        
    Returns:
        Canonical profile type string
    """
    # Try LLM-based analysis if client is available
    if ai_client and ai_model and resume_text:
        try:
            profile_type = _determine_profile_type_with_llm(
                primary_skills, secondary_skills, resume_text, ai_client, ai_model
            )
            if profile_type and profile_type != DEFAULT_PROFILE_TYPE:
                return canonicalize_profile_type(profile_type)
        except Exception as e:
            logger.warning(f"LLM-based profile type detection failed: {e}. Falling back to keyword matching.")
    
    # Fallback: Weighted keyword-based analysis
    return _determine_profile_type_with_keywords(primary_skills, secondary_skills, resume_text)


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


def _determine_profile_type_with_keywords(primary_skills: str, secondary_skills: str, resume_text: str) -> str:
    """
    Fallback: Weighted keyword-based profile type detection.
    Analyzes skill frequency and weights to determine the best match.
    """
    text_blob = _normalize_text_blob(primary_skills, secondary_skills, resume_text)
    if not text_blob:
        return DEFAULT_PROFILE_TYPE
    
    # Calculate scores for each profile type based on keyword matches
    profile_scores = {}
    
    for profile_type, keywords in PROFILE_TYPE_RULES:
        score = 0
        matched_keywords = []
        
        for keyword in keywords:
            # Count occurrences in text
            count = text_blob.count(keyword.lower())
            if count > 0:
                matched_keywords.append(keyword)
                # Weight primary skills more heavily
                if keyword.lower() in primary_skills.lower():
                    score += count * 3  # Primary skills weighted 3x
                elif keyword.lower() in secondary_skills.lower():
                    score += count * 2  # Secondary skills weighted 2x
                else:
                    score += count  # Resume text weighted 1x
        
        if score > 0:
            profile_scores[profile_type] = {
                'score': score,
                'keywords': matched_keywords
            }
    
    # Return profile type with highest score
    if profile_scores:
        best_profile = max(profile_scores.items(), key=lambda x: x[1]['score'])
        logger.info(f"Keyword-based profile type: {best_profile[0]} (score: {best_profile[1]['score']}, matched: {best_profile[1]['keywords']})")
        return canonicalize_profile_type(best_profile[0])
    
    # Fallback: try to match first primary skill token (but filter single letters)
    if primary_skills:
        skills_list = [s.strip() for s in primary_skills.split(',') if s.strip() and len(s.strip()) > 1]
        if skills_list:
            first_token = skills_list[0]
            # Only use if it matches a known profile type
            matches = detect_profile_types_from_text(first_token)
            if matches:
                return canonicalize_profile_type(matches[0])
            # Otherwise, just canonicalize the token (might be a technology)
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

