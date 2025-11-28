#!/usr/bin/env python3
"""
Regression test for explicit experience extraction on Suresh Kavili's resume.
"""

import os
import pytest

from experience_extractor import extract_experience

try:
    from PyPDF2 import PdfReader
except ImportError:  # pragma: no cover - PyPDF2 is part of requirements, but skip gracefully
    PdfReader = None


def _load_resume_text() -> str:
    """Extract text from the bundled PDF resume."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, "Suresh_Kavili_Profile-TechnicalDirector.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Resume PDF not found at {pdf_path}")

    if PdfReader is None:
        pytest.skip("PyPDF2 is not installed in this environment.")

    with open(pdf_path, "rb") as handle:
        reader = PdfReader(handle)
        return "\n".join((page.extract_text() or "") for page in reader.pages)


def test_suresh_resume_explicit_experience():
    """
    The resume contains the phrase "Having 23 years of balanced experience".
    Ensure the updated direct experience patterns capture this value.
    """
    text = _load_resume_text()
    result = extract_experience(text)

    assert result["explicit_experience_used"], "Expected explicit experience detection."
    assert result["total_experience_years"] >= 23.0, "Should detect at least 23 years of experience."






