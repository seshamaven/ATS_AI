# Role Normalization Usage Guide

## Overview

The role normalization system matches original job roles from resumes against the `role_processor` table and returns standardized normalized roles.

## Table Structure

- **Table**: `role_processor`
- **Columns**:
  - `Id`: Primary key
  - `Normalised_roles`: Unique normalized role name (e.g., "Software Engineer")
  - `roles`: JSON array containing all original role variations

## Function: `normalize_role_from_resume()`

### Description
Normalizes a job role from resume by matching against the `role_processor` table.

### Process
1. Reads the candidate's current designation/original role from resume
2. Matches it against entries in `role_processor` table
3. If match found in `roles` JSON array, returns corresponding `Normalised_role`
4. If no match found, returns `"Others"`

### Parameters
- `original_role` (str): The original role/designation from resume
- `fuzzy_threshold` (float): Similarity threshold for fuzzy matching (default: 0.75)

### Returns
- `str`: Normalized role name or `"Others"` if no match found

## Usage Examples

### Example 1: Using the Class Method

```python
from role_processor import RoleProcessor

# Initialize with database config
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Reset@123',
    'database': 'ats_db',
    'port': 3306
}

with RoleProcessor(config=db_config) as rp:
    # Normalize a role
    normalized = rp.normalize_role_from_resume("backend java developer")
    print(normalized)  # Output: "Software Engineer"
    
    # Normalize another role
    normalized = rp.normalize_role_from_resume("Unknown Role")
    print(normalized)  # Output: "Others"
```

### Example 2: Using the Standalone Function

```python
from role_processor import normalize_role

# Simple usage (uses default config)
normalized = normalize_role("asp.net web developer")
print(normalized)  # Output: "Software Engineer"

# With custom config
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Reset@123',
    'database': 'ats_db',
    'port': 3306
}

normalized = normalize_role("Senior Software Engineer", config=db_config)
print(normalized)  # Output: "Software Engineer"
```

### Example 3: Processing Multiple Roles

```python
from role_processor import normalize_role

resume_roles = [
    "Java Developer",
    "Senior .NET Developer",
    "Python Web Developer",
    "Solution Architect",
    "Unknown Role"
]

normalized_roles = []
for role in resume_roles:
    normalized = normalize_role(role)
    normalized_roles.append(normalized)
    print(f"{role} -> {normalized}")

# Output:
# Java Developer -> Software Engineer
# Senior .NET Developer -> Software Engineer
# Python Web Developer -> Software Engineer
# Solution Architect -> Architect
# Unknown Role -> Others
```

## Matching Logic

### 1. Exact Match
- First tries exact match (case-insensitive)
- Normalizes text (removes special chars, handles variations like "Sr." -> "sr", ".NET" -> "net")

### 2. Fuzzy Match
- If no exact match, uses fuzzy matching with similarity scoring
- Combines string similarity (70%) and word overlap (30%)
- Only returns match if similarity >= threshold (default: 0.75)

### 3. No Match
- Returns `"Others"` if no match found

## Normalized Role Categories

The system currently supports these normalized roles:
- Software Engineer
- Architect
- Consultant
- Data Analyst
- Analyst
- Database Administrator
- Project Manager
- Program Manager
- Product Manager
- Engineering Manager
- Jr Project Manager
- ProgramManager
- Others (for unmatched roles)

## Integration with Resume Processing

```python
from role_processor import normalize_role
from designation_extraction import extract_designation

# Extract designation from resume text
resume_text = """
John Doe
Senior Software Engineer
...
"""

# Extract original role
original_role = extract_designation(resume_text)
print(f"Original role: {original_role}")

# Normalize the role
normalized_role = normalize_role(original_role)
print(f"Normalized role: {normalized_role}")

# Use normalized_role as role_type in your system
```

## Error Handling

The function handles errors gracefully:
- Invalid input (None, empty string) → Returns `"Others"`
- Database connection errors → Returns `"Others"` and logs error
- All exceptions are caught and logged

## Performance Notes

- Uses database connection pooling (via context manager)
- Caches connection for multiple lookups
- Fuzzy matching searches all roles (consider indexing for large datasets)

## Testing

Run the test script to verify functionality:

```bash
python test_role_normalization.py
```

