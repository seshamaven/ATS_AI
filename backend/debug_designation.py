#!/usr/bin/env python3
"""Debug designation extraction for the problematic resume."""

import PyPDF2
from designation_extraction import (
    normalize_text, 
    crop_to_experience_section,
    _find_roles_with_dates,
    parse_date_range_end_date,
    extract_designation
)

# Read PDF
pdf = open('738249660.pdf', 'rb')
reader = PyPDF2.PdfReader(pdf)
text = ''.join([page.extract_text() for page in reader.pages])
pdf.close()

print("=" * 80)
print("TESTING DATE PARSING")
print("=" * 80)
test_dates = ['May 2023 to December 2024', 'January 2025 to May 2025', 'May 2022 to May 2023']
for d in test_dates:
    parsed = parse_date_range_end_date(d)
    print(f'{d} -> {parsed}')

print("\n" + "=" * 80)
print("EXPERIENCE SECTION")
print("=" * 80)
normalized = normalize_text(text)
exp = crop_to_experience_section(normalized)
lines = exp.split('\n')
print(f"Total lines: {len(lines)}")
print("\nFirst 60 lines:")
for i, line in enumerate(lines[:60]):
    print(f"{i:3}: {line[:100]}")

print("\n" + "=" * 80)
print("FINDING ROLES WITH DATES")
print("=" * 80)
roles = _find_roles_with_dates(exp, lines)
print(f"Found {len(roles)} roles:")
for end_date, designation, source_line, line_idx in roles:
    print(f"  {designation} (end: {end_date.strftime('%Y-%m')}, line: {line_idx}): {source_line[:80]}")

print("\n" + "=" * 80)
print("FINAL EXTRACTION RESULT")
print("=" * 80)
result = extract_designation(text)
print(f"Result: {result}")
