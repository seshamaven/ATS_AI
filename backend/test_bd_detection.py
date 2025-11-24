#!/usr/bin/env python3
"""Test why Business Development is being detected for Suresh Kavili"""

from profile_type_utils import _check_business_development, determine_profile_types_enhanced, get_all_profile_type_scores
import PyPDF2
import pathlib
import re

# Load resume
pdf_path = pathlib.Path('Suresh_Kavili_Profile-TechnicalDirector.pdf')
pdf = PyPDF2.PdfReader(open(pdf_path, 'rb'))
resume_text = '\n'.join((page.extract_text() or '') for page in pdf.pages)

primary_skills = 'Java, spring, Python, react, Azure, AWS, Docker, Kubernetes, ci/cd pipelines, DevOps, resources, performance tuning, disaster recovery, code coverage, environments, automation, monitoring, power apps, tensorflow, providers, hibernate, firebase, rest api, firewall, reports, laravel, kanban, oracle, agile, scrum, mysql, mongo, json, saas, jira, etl, sql, pwa, php, ai, js, r'

print("=" * 80)
print("TESTING BUSINESS DEVELOPMENT DETECTION")
print("=" * 80)

# Test 1: Check resume text alone
print("\n1. Testing resume text alone:")
is_bd_resume = _check_business_development(resume_text)
print(f"   BD detected: {is_bd_resume}")

# Test 2: Check primary skills alone
print("\n2. Testing primary skills alone:")
is_bd_skills = _check_business_development(primary_skills)
print(f"   BD detected: {is_bd_skills}")

# Test 3: Check combined text (as used in actual function)
print("\n3. Testing combined text (resume + skills):")
combined = resume_text + " " + primary_skills
is_bd_combined = _check_business_development(combined)
print(f"   BD detected: {is_bd_combined}")

# Test 4: Find all BD pattern matches in combined text
print("\n4. Searching for BD patterns in combined text:")
patterns = [
    r"\bbusiness\s+development\b",
    r"\bbusiness\s+dev\b",
    r"\bbusiness\s+development\s+executive\b",
    r"\bbusiness\s+development\s+manager\b",
    r"\bb2b\s+sales\b",
    r"\bclient\s+acquisition\b",
    r"\bmarket\s+expansion\b",
    r"\bpartnership\s+development\b",
    r"\bstrategic\s+partnerships\b",
    r"\baccount\s+development\b",
    r"\bbd\b",
    r"\bbde\b",
]

text_lower = combined.lower()
for pattern in patterns:
    matches = list(re.finditer(pattern, text_lower))
    if matches:
        print(f"   ✓ Pattern '{pattern}' found {len(matches)} times:")
        for i, match in enumerate(matches[:3], 1):
            start = max(0, match.start() - 30)
            end = min(len(text_lower), match.end() + 30)
            context = text_lower[start:end]
            print(f"      Match {i}: '{context}'")

# Test 5: Check what profile types are detected
print("\n5. Profile type detection result:")
profiles, confidence, metadata = determine_profile_types_enhanced(
    primary_skills=primary_skills,
    resume_text=resume_text
)
print(f"   Detected profiles: {profiles}")
print(f"   Confidence: {confidence}")

# Test 6: Check all scores
print("\n6. All profile type scores:")
all_scores = get_all_profile_type_scores(
    primary_skills=primary_skills,
    resume_text=resume_text
)
sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
non_zero = [(k, v) for k, v in sorted_scores if v > 0]
if non_zero:
    print(f"   Found {len(non_zero)} profiles with non-zero scores:")
    for profile, score in non_zero[:10]:
        print(f"      {profile}: {score:.2f}")
else:
    print("   ⚠️  All scores are 0.00!")

print("\n" + "=" * 80)

