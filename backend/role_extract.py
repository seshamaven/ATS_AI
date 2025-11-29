"""
Lightweight utilities to work with high-level roles and subroles.

This module is intentionally **category-agnostic** – it only knows about
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

# NOTE: Only Role and Subrole are kept here (no Category).
ROLE_SUBROLE_PAIRS: List[Tuple[str, str]] = [
    ("Software Engineer", "Backend Developer"),
    ("Software Engineer", "Frontend Developer"),
    ("Software Engineer", "Full Stack Developer"),
    ("Software Engineer", ".NET Developer"),
    ("Software Engineer", "Java Developer"),
    ("Software Engineer", "Python Developer"),
    ("Software Engineer", "Mobile App Developer"),
    ("Software Engineer", "Embedded Systems Developer"),
    ("Software Engineer", "Game Developer"),
    ("Software Engineer", "Dev Tools / SDK Developer"),
    ("Web Developer", "Frontend Developer"),
    ("Web Developer", "Backend Developer"),
    ("Web Developer", "Full Stack Developer"),
    ("Web Developer", "UI/UX Developer"),
    ("Web Developer", "WordPress / CMS Developer"),
    ("Mobile Developer", "iOS Developer"),
    ("Mobile Developer", "Android Developer"),
    ("Mobile Developer", "Cross-Platform Developer"),
    ("Mobile Developer", "Hybrid App Developer"),
    ("Data Scientist", "Machine Learning Engineer"),
    ("Data Scientist", "AI Engineer"),
    ("Data Scientist", "Data Analyst"),
    ("Data Scientist", "Big Data Engineer"),
    ("Data Scientist", "NLP Engineer"),
    ("Data Scientist", "Research Scientist"),
    ("Data Analyst", "Business Intelligence Analyst"),
    ("Data Analyst", "Reporting Analyst"),
    ("Data Analyst", "Marketing Data Analyst"),
    ("Data Analyst", "Financial Data Analyst"),
    ("Data Engineer", "ETL Developer"),
    ("Data Engineer", "Data Pipeline Engineer"),
    ("Data Engineer", "Database Developer"),
    ("Data Engineer", "Cloud Data Engineer"),
    ("DevOps Engineer", "Cloud Engineer"),
    ("DevOps Engineer", "CI/CD Engineer"),
    ("DevOps Engineer", "Infrastructure Engineer"),
    ("DevOps Engineer", "Site Reliability Engineer"),
    ("DevOps Engineer", "Build & Release Engineer"),
    ("Cloud Engineer", "Cloud Architect"),
    ("Cloud Engineer", "Cloud Security Engineer"),
    ("Cloud Engineer", "Cloud Operations Engineer"),
    ("Cloud Engineer", "Cloud Migration Specialist"),
    ("QA Engineer", "Automation Tester"),
    ("QA Engineer", "Manual Tester"),
    ("QA Engineer", "Performance Tester"),
    ("QA Engineer", "Security Tester"),
    ("QA Engineer", "Test Lead"),
    ("Project Manager", "Technical Project Manager"),
    ("Project Manager", "Agile Project Manager"),
    ("Project Manager", "IT Project Manager"),
    ("Project Manager", "Program Manager"),
    ("Project Manager", "Scrum Master"),
    ("Business Analyst", "System Analyst"),
    ("Business Analyst", "Functional Analyst"),
    ("Business Analyst", "Product Analyst"),
    ("Business Analyst", "Business Systems Analyst"),
    ("Business Analyst", "Process Analyst"),
    ("Product Manager", "Technical Product Manager"),
    ("Product Manager", "Platform Product Manager"),
    ("Product Manager", "Growth Product Manager"),
    ("Cybersecurity Analyst", "Security Analyst"),
    ("Cybersecurity Analyst", "Ethical Hacker"),
    ("Cybersecurity Analyst", "Penetration Tester"),
    ("Cybersecurity Analyst", "Network Security Engineer"),
    ("Cybersecurity Analyst", "Information Security Officer"),
    ("Cybersecurity Analyst", "SOC Analyst"),
    ("Network Engineer", "Network Administrator"),
    ("Network Engineer", "Network Architect"),
    ("Network Engineer", "Wireless Network Engineer"),
    ("Network Engineer", "Cloud Network Engineer"),
    ("Network Engineer", "VPN / Security Specialist"),
    ("Database Administrator", "SQL DBA"),
    ("Database Administrator", "NoSQL DBA"),
    ("Database Administrator", "Data Warehouse Specialist"),
    ("Database Administrator", "Database Architect"),
    ("Big Data Engineer", "Hadoop Developer"),
    ("Big Data Engineer", "Spark Engineer"),
    ("Big Data Engineer", "Data Lake Engineer"),
    ("SAP Consultant", "SAP FI/CO Consultant"),
    ("SAP Consultant", "SAP MM Consultant"),
    ("SAP Consultant", "SAP SD Consultant"),
    ("SAP Consultant", "SAP HCM Consultant"),
    ("SAP Consultant", "SAP Basis Consultant"),
    ("System Administrator", "Windows System Admin"),
    ("System Administrator", "Linux/Unix Admin"),
    ("System Administrator", "Network Admin"),
    ("System Administrator", "Cloud System Admin"),
    ("System Administrator", "Virtualization Specialist"),
    ("IT Support", "Helpdesk Technician"),
    ("IT Support", "Desktop Support"),
    ("IT Support", "IT Support Engineer"),
    ("IT Support", "Technical Support Specialist"),
    ("Accountant", "Tax Accountant"),
    ("Accountant", "Cost Accountant"),
    ("Accountant", "Audit Specialist"),
    ("Accountant", "Financial Accountant"),
    ("Financial Analyst", "Business Finance Analyst"),
    ("Financial Analyst", "Investment Analyst"),
    ("Financial Analyst", "Risk Analyst"),
    ("Financial Analyst", "Budget Analyst"),
    ("Banking / Insurance", "Credit Analyst"),
    ("Banking / Insurance", "Loan Officer"),
    ("Banking / Insurance", "Insurance Underwriter"),
    ("Banking / Insurance", "Claims Analyst"),
    ("Teacher / Lecturer", "School Teacher"),
    ("Teacher / Lecturer", "College Lecturer / Professor"),
    ("Teacher / Lecturer", "Corporate Trainer"),
    ("Teacher / Lecturer", "Online Tutor"),
    ("Teacher / Lecturer", "Education Consultant"),
    ("Doctor / Physician", "General Practitioner"),
    ("Doctor / Physician", "Specialist (Cardiologist, Neurologist, etc.)"),
    ("Doctor / Physician", "Surgeon"),
    ("Nurse", "Registered Nurse"),
    ("Nurse", "Nurse Practitioner"),
    ("Nurse", "ICU / Critical Care Nurse"),
    ("Medical Technician", "Lab Technician"),
    ("Medical Technician", "Radiology Technician"),
    ("Medical Technician", "Physiotherapist"),
    ("Marketing Manager", "Digital Marketing Specialist"),
    ("Marketing Manager", "SEO / SEM Specialist"),
    ("Marketing Manager", "Social Media Manager"),
    ("Marketing Manager", "Content Marketing Manager"),
    ("Sales Executive", "Inside Sales"),
    ("Sales Executive", "Field Sales"),
    ("Sales Executive", "Account Manager"),
    ("Sales Executive", "Business Development Manager"),
    ("Legal", "Corporate Lawyer"),
    ("Legal", "Legal Advisor"),
    ("Legal", "Compliance Officer"),
    ("HR / People Operations", "HR Generalist"),
    ("HR / People Operations", "Recruiter / Talent Acquisition"),
    ("HR / People Operations", "HR Manager"),
    ("HR / People Operations", "Payroll Specialist"),
    ("Creative / Design", "Graphic Designer"),
    ("Creative / Design", "UX/UI Designer"),
    ("Creative / Design", "Animator"),
    ("Creative / Design", "Video Editor"),
]


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
    text = text.lower()
    text = re.sub(r"[/,&]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_role_subrole(
    title_or_text: str,
    candidates: Sequence[Tuple[str, str]] = ROLE_SUBROLE_PAIRS,
) -> Optional[Tuple[str, str]]:
    """
    Very simple heuristic matcher:

    - Looks for exact subrole phrases inside the text first.
    - If no subrole matches, falls back to looking for role phrases.
    - Returns the first best match as (role, subrole).

    Only Role and Subrole are used – category is deliberately ignored.
    """
    if not title_or_text:
        return None

    text_norm = _normalize(title_or_text)

    # 1) Exact subrole phrase inside the text
    for role, subrole in candidates:
        if _normalize(subrole) in text_norm:
            return role, subrole

    # 2) Role phrase inside the text (no specific subrole)
    for role, subrole in candidates:
        if _normalize(role) in text_norm:
            return role, subrole

    logger.debug("No role/subrole match for text: %r", title_or_text)
    return None


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
    "list_roles",
    "list_subroles",
    "detect_role_subrole",
    "detect_role_only",
    "detect_subrole_only",
]


