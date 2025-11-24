#!/usr/bin/env python3
"""Test profile type scoring to understand why Generalist is returned"""

from profile_type_utils import (
    _calculate_normalized_scores,
    determine_profile_types_enhanced,
    PROFILE_TYPE_RULES_ENHANCED
)

# Candidate's actual skills
primary_skills = "Python, MySQL, Django, HTML, CSS, pycharm, notebook, jupyter, salesforce, ai"

print("=" * 80)
print("TESTING PROFILE TYPE DETECTION")
print("=" * 80)
print(f"\nPrimary Skills: {primary_skills}\n")

# Calculate all scores before filtering
from profile_type_utils import _normalize_text_blob, _count_keyword_matches, COMPILED_PATTERNS
text_blob = _normalize_text_blob(primary_skills, "", "")
print(f"Normalized text blob: {text_blob[:100]}...")
print(f"\nTesting keyword matches:")
print(f"  'python' in text: {'python' in text_blob}")
print(f"  'django' in text: {'django' in text_blob}")
print(f"  'salesforce' in text: {'salesforce' in text_blob}")

# Test pattern matching
if "Python" in COMPILED_PATTERNS:
    python_pattern = COMPILED_PATTERNS["Python"].get("python")
    if python_pattern:
        matches = python_pattern.findall(text_blob)
        print(f"  Pattern matches for 'python': {matches}")

# Debug: Check what's happening in scoring
from profile_type_utils import _extract_first_lines_of_skills, _check_business_development
first_lines = _extract_first_lines_of_skills(primary_skills, num_lines=2)
print(f"\nFirst 2 lines of skills: '{first_lines}'")
print(f"Primary skills lower: '{primary_skills.lower()[:50]}...'")
is_bd = _check_business_development(primary_skills)
print(f"Is Business Development detected: {is_bd}")
if is_bd:
    print("  ⚠️  This would cause only BD profile to be processed!")

# Manual test: Calculate Python score manually
print("\n" + "=" * 80)
print("MANUAL CALCULATION TEST for Python:")
print("-" * 80)
python_weights = next((kw for pt, kw in PROFILE_TYPE_RULES_ENHANCED if pt == "Python"), None)
if python_weights:
    print(f"Python keywords and weights: {list(python_weights.items())[:5]}...")
    # Check which keywords match
    matched = []
    for keyword, weight in python_weights.items():
        count = _count_keyword_matches(keyword, text_blob, "Python")
        if count > 0:
            matched.append((keyword, weight, count))
            print(f"  ✓ '{keyword}' (weight={weight}): {count} matches")
    print(f"\nTotal matched keywords: {len(matched)}")
    if matched:
        # Calculate raw score manually
        primary_lower = primary_skills.lower()
        first_lines_lower = first_lines.lower()
        raw = 0.0
        for keyword, weight, count in matched:
            if keyword.lower() in first_lines_lower:
                multiplier = 5.0
            elif keyword.lower() in primary_lower:
                multiplier = 3.0
            else:
                multiplier = 1.0
            score = count * weight * multiplier
            raw += score
            print(f"    '{keyword}': {count} × {weight} × {multiplier} = {score:.2f}")
        print(f"\n  Manual raw score: {raw:.2f}")

scores = _calculate_normalized_scores(text_blob, primary_skills, "", "")

print(f"\nTotal scores calculated: {len(scores)}")
print("All calculated scores (BEFORE confidence filtering):")
print("-" * 80)
for i, s in enumerate(scores[:10], 1):
    print(f"{i}. {s.profile_type:20s} | "
          f"raw={s.raw_score:6.2f} | "
          f"normalized={s.normalized_score:.4f} | "
          f"confidence={s.confidence:.3f} | "
          f"keywords={len(s.matched_keywords)} | "
          f"matched={s.matched_keywords[:3]}")

print("\n" + "=" * 80)
print("Filtering by min_confidence=0.4:")
print("-" * 80)
valid_scores = [s for s in scores if s.confidence >= 0.4]
if valid_scores:
    print(f"Found {len(valid_scores)} profiles above confidence threshold:")
    for s in valid_scores:
        print(f"  ✓ {s.profile_type}: confidence={s.confidence:.3f}")
else:
    print("❌ NO profiles meet confidence threshold (0.4)")
    print("   This is why 'Generalist' is returned!")

print("\n" + "=" * 80)
print("Final result from determine_profile_types_enhanced():")
print("-" * 80)
profiles, confidence, metadata = determine_profile_types_enhanced(
    primary_skills=primary_skills,
    resume_text=""
)
print(f"Detected profiles: {profiles}")
print(f"Confidence: {confidence}")
print(f"Metadata: {metadata}")

# Calculate max_possible for Python and Salesforce
print("\n" + "=" * 80)
print("Max Possible Scores Calculation:")
print("-" * 80)
for profile_type, keyword_weights in PROFILE_TYPE_RULES_ENHANCED:
    if profile_type in ["Python", "Salesforce"]:
        sum_weights = sum(keyword_weights.values())
        max_possible = sum_weights * 5.0  # Highest multiplier
        print(f"{profile_type}:")
        print(f"  Sum of keyword weights: {sum_weights:.1f}")
        print(f"  Max possible (with 5.0x multiplier): {max_possible:.1f}")
        # Find actual raw score
        actual_score = next((s.raw_score for s in scores if s.profile_type == profile_type), 0.0)
        if actual_score > 0:
            normalized = actual_score / max_possible
            print(f"  Actual raw score: {actual_score:.2f}")
            print(f"  Normalized score: {normalized:.4f}")
            print(f"  This explains why confidence is low!")

