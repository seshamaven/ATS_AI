# Extraction Method Calls - Flow Diagram

## Answer: **Methods are CALLED** (not creating new data)

The extraction process **calls existing Python functions/methods** to **extract/detect** information from the job description text. It does NOT create new data - it finds patterns and matches in the provided text.

---

## Method Call Flow

```
API Request (/api/profileRankingByJD)
    ↓
Extract Job Metadata (Python-only, NO AI)
    ↓
    ├─→ 1. ROLE Extraction
    │       ├─→ Pattern Matching (inline code)
    │       └─→ detect_subrole_from_query()  [CALLED]
    │           └─→ Uses ROLE_MAPPING dictionary
    │
    ├─→ 2. SUB_ROLE Extraction
    │       └─→ detect_subrole_frontend_backend()  [CALLED]
    │           └─→ Keyword matching (Frontend/Backend)
    │
    ├─→ 3. PROFILE_TYPE Extraction
    │       ├─→ Pattern Matching (inline code for BI/Data Science)
    │       ├─→ infer_profile_type_from_requirements()  [CALLED]
    │       │   └─→ Uses PROFILE_TYPE_RULES dictionary
    │       └─→ detect_profile_types_from_text()  [CALLED]
    │           └─→ Uses PROFILE_TYPE_RULES dictionary
    │
    ├─→ 4. PROFILE_SUB_TYPE Extraction
    │       └─→ extract_specific_technologies()  [CALLED]
    │           └─→ Uses Technology Map dictionary
    │
    ├─→ 5. PRIMARY_SKILLS Extraction
    │       └─→ extract_skills_from_text()  [CALLED]
    │           └─→ extract_skills_advanced() from skill_extractor.py
    │               └─→ Uses TECH_SKILLS dictionary
    │
    └─→ 6. SECONDARY_SKILLS Extraction
            └─→ Split from extracted_skills (inline code)
```

---

## Detailed Method Calls

### 1. **ROLE Extraction**

**Location**: `ats_api.py` lines 2624-2647

**Methods Called**:
```python
# Method 1: detect_subrole_from_query()
role_info = detect_subrole_from_query(searchable_text)
# Returns: {'sub_role': '...', 'main_role': '...', 'profile_type': '...'}
```

**Function Definition**: `ats_api.py` lines 2058-2093
- Searches job description for role keywords
- Matches against `ROLE_MAPPING` dictionary
- Returns role information or None

**What it does**: 
- ✅ **CALLS** a method
- ✅ **EXTRACTS** from job description text
- ❌ Does NOT create new data

---

### 2. **SUB_ROLE Extraction**

**Location**: `ats_api.py` lines 2649-2661

**Methods Called**:
```python
# Method 1: detect_subrole_frontend_backend()
detected_subrole = detect_subrole_frontend_backend(searchable_text)
# Returns: 'Frontend', 'Backend', 'Full Stack Developer', or None
```

**Function Definition**: `ats_api.py` lines 1910-1961
- Counts frontend vs backend keyword matches
- Returns appropriate sub-role

**What it does**:
- ✅ **CALLS** a method
- ✅ **DETECTS** from keywords in text
- ❌ Does NOT create new data

---

### 3. **PROFILE_TYPE Extraction**

**Location**: `ats_api.py` lines 2663-2711

**Methods Called**:
```python
# Method 1: infer_profile_type_from_requirements()
inferred_profile_types = infer_profile_type_from_requirements(
    required_skills_list, 
    job_description
)
# Returns: List[str] like ['Java', 'Python']

# Method 2: detect_profile_types_from_text()
detected_profiles = detect_profile_types_from_text(
    job_description, 
    ', '.join(required_skills_list)
)
# Returns: List[str] like ['Java']
```

**Function Definitions**:
- `infer_profile_type_from_requirements()`: `profile_type_utils.py` lines 1246-1260
- `detect_profile_types_from_text()`: `profile_type_utils.py` lines 585-601

**What it does**:
- ✅ **CALLS** methods
- ✅ **INFERS/DETECTS** from skills and text
- ❌ Does NOT create new data

---

### 4. **PROFILE_SUB_TYPE Extraction**

**Location**: `ats_api.py` lines 2713-2719

**Methods Called**:
```python
# Method: extract_specific_technologies()
specific_techs = extract_specific_technologies(
    searchable_text, 
    ', '.join(required_skills_list)
)
# Returns: List[str] like ['Spring Boot', 'MySQL', 'React']
```

**Function Definition**: `ats_api.py` lines 1964-2023
- Searches for technology keywords
- Matches against Technology Map dictionary
- Returns up to 3 technologies

**What it does**:
- ✅ **CALLS** a method
- ✅ **EXTRACTS** technologies from text
- ❌ Does NOT create new data

---

### 5. **PRIMARY_SKILLS Extraction**

**Location**: `ats_api.py` lines 2485-2545

**Methods Called**:
```python
# Method: extract_skills_from_text()
from resume_parser import extract_skills_from_text
extracted_skills = extract_skills_from_text(job_description)
# Returns: List[str] like ['Java', 'Spring Boot', 'MySQL', ...]
```

**Function Definition**: `resume_parser.py` lines 1661-1667
- Calls `extract_skills_advanced()` from `skill_extractor.py`
- Matches against `TECH_SKILLS` dictionary
- Returns list of detected skills

**What it does**:
- ✅ **CALLS** a method
- ✅ **EXTRACTS** skills from text
- ❌ Does NOT create new data

---

### 6. **SECONDARY_SKILLS Extraction**

**Location**: `ats_api.py` lines 2537-2545

**Methods Called**:
```python
# No method call - inline code
# Splits extracted_skills list
if len(extracted_skills) > 10:
    preferred_skills = extracted_skills[10:]  # Skills after first 10
```

**What it does**:
- ✅ **SPLITS** existing extracted skills
- ❌ Does NOT call a method (inline code)
- ❌ Does NOT create new data

---

## Summary: Methods Called vs Inline Code

| Field | Method Called? | Method Name | Location |
|-------|---------------|-------------|----------|
| `role` | ✅ Yes | `detect_subrole_from_query()` | `ats_api.py:2058` |
| `sub_role` | ✅ Yes | `detect_subrole_frontend_backend()` | `ats_api.py:1910` |
| `profile_type` | ✅ Yes | `infer_profile_type_from_requirements()` | `profile_type_utils.py:1246` |
| `profile_type` | ✅ Yes | `detect_profile_types_from_text()` | `profile_type_utils.py:585` |
| `profile_sub_type` | ✅ Yes | `extract_specific_technologies()` | `ats_api.py:1964` |
| `primary_skills` | ✅ Yes | `extract_skills_from_text()` | `resume_parser.py:1661` |
| `secondary_skills` | ❌ No | Inline code (splitting) | `ats_api.py:2537` |

---

## Key Points

### ✅ **Methods ARE Called**
- 6 different methods/functions are called during extraction
- Each method has a specific purpose (role detection, skill extraction, etc.)
- Methods use predefined dictionaries (ROLE_MAPPING, PROFILE_TYPE_RULES, etc.)

### ✅ **Data is EXTRACTED (not created)**
- All extraction is based on the job description text provided
- Methods search for patterns, keywords, and matches
- No new data is generated - only existing information is found and formatted

### ✅ **No AI/LLM Used**
- All methods use Python-only code
- Pattern matching, keyword detection, dictionary lookups
- Deterministic results (same input = same output)

### ✅ **Dictionaries Used**
- `ROLE_MAPPING` - for role detection
- `PROFILE_TYPE_RULES` - for profile type detection
- `Technology Map` - for profile_sub_type
- `TECH_SKILLS` - for skill extraction

---

## Example Flow

**Input**:
```json
{
  "job_description": "We are looking for a Java Developer with Spring Boot and MySQL experience...",
  "required_skills": "Java, Spring Boot, MySQL"
}
```

**Method Calls**:
1. `detect_subrole_from_query()` → Finds "Java Developer" → Returns `{'main_role': 'Software Engineer', 'sub_role': 'Java Developer'}`
2. `detect_subrole_frontend_backend()` → Finds backend keywords → Returns `'Backend'`
3. `infer_profile_type_from_requirements()` → Finds "Java" in skills → Returns `['Java']`
4. `extract_specific_technologies()` → Finds "Spring Boot", "MySQL" → Returns `['Spring Boot', 'MySQL']`
5. `extract_skills_from_text()` → Finds skills in text → Returns `['Java', 'Spring Boot', 'MySQL', ...]`

**Output**:
```json
{
  "role": "Software Engineer",
  "sub_role": "Backend",
  "profile_type": "Java",
  "profile_sub_type": "Spring Boot, MySQL",
  "primary_skills": ["Java", "Spring Boot", "MySQL"],
  "secondary_skills": [...]
}
```

---

## Conclusion

**Answer**: Methods ARE called. The extraction process calls 6 different Python functions to extract information from the job description text. No new data is created - only existing information is detected and extracted using pattern matching and dictionary lookups.



