# Skill-Based Role Detection for Fresher Resumes

## Overview

The role normalization system now supports **skill-based role detection** for fresher resumes that don't have a current designation. When a designation is missing, the system automatically extracts skills from the resume and infers the appropriate normalized role.

## How It Works

### Process Flow

1. **Try Designation First** (if provided)
   - Matches original_role against `role_processor` table
   - Uses exact and fuzzy matching

2. **Fallback to Skills** (if designation is missing/empty)
   - Extracts skills from resume text (if provided)
   - Uses provided primary_skills and secondary_skills
   - Infers role using `infer_role_from_skills()` from `role_extract.py`
   - Maps inferred role to normalized role from `role_processor` table

3. **Return Result**
   - Returns normalized role if match found
   - Returns "Others" if no match found

## Usage Examples

### Example 1: Fresher Resume with Skills

```python
from role_processor import normalize_role

# Fresher resume - no designation, but has skills
resume_text = """
EDUCATION:
B.Tech Computer Science

SKILLS:
Java, Spring Boot, Hibernate, MySQL, REST API

PROJECTS:
E-commerce application using Spring Boot
"""

normalized = normalize_role(
    original_role=None,  # No designation
    resume_text=resume_text,
    primary_skills="Java, Spring Boot, Hibernate, MySQL, REST API"
)

print(normalized)  # Output: "Software Engineer"
```

### Example 2: Using Pre-extracted Skills

```python
from role_processor import normalize_role

# If you already have extracted skills
normalized = normalize_role(
    original_role=None,
    primary_skills="Python, Django, React, JavaScript, PostgreSQL",
    secondary_skills="Communication, Teamwork"
)

print(normalized)  # Output: "Software Engineer"
```

### Example 3: With Designation (Normal Flow)

```python
from role_processor import normalize_role

# Resume with designation - uses designation first
normalized = normalize_role(original_role="Senior Java Developer")

print(normalized)  # Output: "Software Engineer"
```

### Example 4: Mixed - Designation Missing, Skills Available

```python
from role_processor import normalize_role

# Designation is empty/None, but skills are available
normalized = normalize_role(
    original_role="",  # Empty designation
    primary_skills="SQL, MySQL, PostgreSQL, Oracle, Database Design"
)

print(normalized)  # Output: "Database Administrator"
```

## Function Signature

```python
def normalize_role(
    original_role: Optional[str] = None,
    resume_text: Optional[str] = None,
    primary_skills: Optional[str] = None,
    secondary_skills: Optional[str] = None,
    config: Dict = None,
    fuzzy_threshold: float = 0.75
) -> str
```

### Parameters

- **original_role** (optional): The original role/designation from resume
- **resume_text** (optional): Full resume text (used to extract skills if primary_skills not provided)
- **primary_skills** (optional): Comma-separated primary/technical skills
- **secondary_skills** (optional): Comma-separated secondary skills
- **config** (optional): Database configuration dict
- **fuzzy_threshold** (optional): Similarity threshold for fuzzy matching (default: 0.75)

### Returns

- Normalized role name (e.g., "Software Engineer", "Data Analyst", "Database Administrator")
- "Others" if no match found

## Skill-Based Role Inference

The system uses `infer_role_from_skills()` from `role_extract.py` which recognizes:

### Software Engineering Roles
- **Keywords**: Java, Python, C#, .NET, JavaScript, React, Angular, Node.js, etc.
- **Maps to**: "Software Engineer"

### Data Roles
- **Data Analyst**: SQL, Excel, Tableau, Power BI, Data Analysis
- **Data Scientist**: Python, Machine Learning, TensorFlow, Pandas, NumPy
- **Data Engineer**: ETL, Spark, Hadoop, Kafka
- **Maps to**: "Data Analyst" or "Software Engineer"

### Database Roles
- **Keywords**: SQL, MySQL, PostgreSQL, Oracle, Database Design, DBA
- **Maps to**: "Database Administrator"

### Other Roles
- **DevOps Engineer**: DevOps, CI/CD, Docker, Kubernetes
- **QA Engineer**: Selenium, Test Automation, QA
- **Mobile Developer**: Android, iOS, React Native, Flutter
- **SAP Consultant**: SAP, ABAP, HANA
- **Maps to**: "Software Engineer" or "Consultant"

## Integration with Resume Processing

```python
from role_processor import normalize_role
from designation_extraction import extract_designation
from skill_extractor import extract_skills

# Process resume
resume_text = """..."""

# Extract designation
original_role = extract_designation(resume_text)

# Extract skills
skills_result = extract_skills(resume_text, return_categories=True)
primary_skills = ', '.join(skills_result.get('tech_skills', []))
secondary_skills = ', '.join(skills_result.get('soft_skills', []))

# Normalize role (handles both designation and skills)
normalized_role = normalize_role(
    original_role=original_role,
    resume_text=resume_text,
    primary_skills=primary_skills,
    secondary_skills=secondary_skills
)

# Use normalized_role as role_type
print(f"Role Type: {normalized_role}")
```

## Test Results

All test cases passed successfully:

✅ Fresher with Java/Spring skills → Software Engineer
✅ Fresher with Python/ML skills → Software Engineer
✅ Fresher with Frontend skills → Software Engineer
✅ Fresher with Database skills → Database Administrator
✅ Fresher with Full Stack skills → Software Engineer
✅ Fresher with Data Analyst skills → Data Analyst
✅ Fresher with no IT skills → Others

## Benefits

1. **Handles Fresher Resumes**: Automatically detects roles for candidates without work experience
2. **Skill-Based Matching**: Uses technical skills to infer appropriate roles
3. **Backward Compatible**: Still works with existing designation-based matching
4. **Flexible Input**: Accepts resume text, pre-extracted skills, or both
5. **Robust Fallback**: Returns "Others" if no match found

## Notes

- The system prioritizes designation over skills (if both are available)
- Skill extraction is automatic if `resume_text` is provided and `primary_skills` is not
- The inferred role is mapped to normalized roles from the `role_processor` table
- All skill-based inferences are logged for debugging

