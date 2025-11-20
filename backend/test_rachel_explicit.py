#!/usr/bin/env python3
"""Test explicit experience extraction on Rachel Morris resume"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experience_extractor import extract_experience

# Extract text from PDF
def extract_text_from_pdf(file_path: str) -> str:
    try:
        import PyPDF2
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        return f"Error: {e}"

# Test Rachel Morris resume
file_path = "524691788.pdf"
if not os.path.exists(file_path):
    file_path = os.path.join("backend", file_path)

text = extract_text_from_pdf(file_path)

# Find the explicit experience mention
import re
patterns = [
    r"25\s*\+\s*years",
    r"25\s*\+\s*years.*experience",
    r"25.*years.*experience",
]
print("Searching for '25+ years' patterns in text:")
for pattern in patterns:
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        print(f"  Pattern '{pattern}': Found {len(matches)} matches")
        for match in matches[:3]:
            # Find context
            idx = text.lower().find(match.lower())
            if idx >= 0:
                context = text[max(0, idx-50):idx+len(match)+50]
                print(f"    Context: ...{context}...")

print("\n" + "="*80)
print("Testing experience extraction:")
print("="*80)
result = extract_experience(text)
print(f"Total Experience: {result.get('total_experience_years')} years")
print(f"Explicit Used: {result.get('explicit_experience_used')}")
print(f"Segments: {result.get('segments')}")

# Test explicit extraction directly
from experience_extractor import ExperienceExtractor
extractor = ExperienceExtractor(text)
explicit = extractor._extract_explicit_experience()
print(f"\nDirect explicit extraction: {explicit}")

