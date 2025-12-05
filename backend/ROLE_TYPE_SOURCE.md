# Where Does `role_type` Come From?

## Overview

The `role_type` field in the ATS system comes from the **`role_processor` table** via the `normalize_role_from_resume()` function.

---

## 🔄 Current Flow (After Integration)

### Step 1: Extract Current Designation
```python
current_designation = extract_designation(resume_text)
# Example: "Senior Java Developer" or "Project Manager"
```

### Step 2: Normalize Designation Using `role_processor` Table
```python
from role_processor import normalize_role

role_type = normalize_role(
    original_role=current_designation,
    resume_text=resume_text,
    primary_skills=primary_skills,
    secondary_skills=secondary_skills
)
# Example: "Senior Java Developer" -> "Software Engineer"
# Example: "Project Manager" -> "Project Manager"
# Example: "Unknown Role" -> "Others"
```

### Step 3: Fallback if No Designation (Fresher Resumes)
```python
# If current_designation is empty/None
# Extract skills and infer role, then normalize
if not role_type:
    inferred_role = infer_role_from_skills(primary_skills, secondary_skills)
    role_type = normalize_role(
        original_role=inferred_role,
        primary_skills=primary_skills,
        secondary_skills=secondary_skills
    )
```

---

## 📊 Source: `role_processor` Table

The `role_type` comes from the **`Normalised_roles`** column in the `role_processor` table.

### Table Structure:
```sql
role_processor
├── Id (Primary Key)
├── Normalised_roles (e.g., "Software Engineer", "Architect", "Consultant")
└── roles (JSON array of original roles)
```

### Example Mappings:
- **Original**: ".NET Developer", "Java Developer", "Python Developer"
- **Normalized**: "Software Engineer"

- **Original**: "Solution Architect", "Enterprise Architect"
- **Normalized**: "Architect"

- **Original**: "SAP Consultant", "Business Consultant"
- **Normalized**: "Consultant"

---

## 🎯 How It Works

### 1. **With Designation** (Most Common)
```
Resume has: "Senior Java Developer"
    ↓
normalize_role("Senior Java Developer")
    ↓
Searches role_processor table
    ↓
Finds match in roles JSON array
    ↓
Returns: "Software Engineer"
    ↓
role_type = "Software Engineer"
```

### 2. **Without Designation** (Fresher Resume)
```
Resume has: No designation, but skills: "Java, Spring Boot, MySQL"
    ↓
infer_role_from_skills() → "Software Engineer"
    ↓
normalize_role("Software Engineer")
    ↓
Searches role_processor table
    ↓
Finds match
    ↓
Returns: "Software Engineer"
    ↓
role_type = "Software Engineer"
```

### 3. **Unknown Role**
```
Resume has: "Unknown Job Title XYZ"
    ↓
normalize_role("Unknown Job Title XYZ")
    ↓
Searches role_processor table
    ↓
No match found
    ↓
Returns: "Others"
    ↓
role_type = "Others"
```

---

## 📝 Available Normalized Roles

The `role_type` can be one of these values (from `role_processor` table):

1. **Software Engineer** (most common)
2. **Architect**
3. **Consultant**
4. **Data Analyst**
5. **Analyst**
6. **Database Administrator**
7. **Project Manager**
8. **Program Manager**
9. **Product Manager**
10. **Engineering Manager**
11. **Jr Project Manager**
12. **ProgramManager**
13. **Others** (for unmatched roles)

---

## 🔧 Code Location

The normalization happens in `resume_parser.py` around **line 1738-1786**:

```python
# Normalize current designation
if current_designation:
    role_type = normalize_role(
        original_role=current_designation,
        resume_text=resume_text,
        primary_skills=primary_skills,
        secondary_skills=secondary_skills_str
    )

# Fallback: infer from skills if no designation
if not role_type:
    inferred_role = infer_role_from_skills(...)
    role_type = normalize_role(
        original_role=inferred_role,
        ...
    )
```

---

## ✅ Summary

**`role_type` comes from:**
1. ✅ **`role_processor` table** (via `normalize_role_from_resume()`)
2. ✅ Normalized from original designation or inferred from skills
3. ✅ Returns standardized role names (e.g., "Software Engineer", "Architect")
4. ✅ Returns "Others" if no match found

**It does NOT come from:**
- ❌ Direct designation extraction (raw designation is normalized first)
- ❌ Hardcoded values
- ❌ Pattern matching only (uses database lookup)

---

## 🚀 Benefits

1. **Consistency**: All roles are normalized to standard categories
2. **Database-Driven**: Easy to update mappings without code changes
3. **Handles Variations**: ".NET Developer", "Java Developer" → "Software Engineer"
4. **Fresher Support**: Infers role from skills when designation is missing
5. **Fallback**: Returns "Others" for unmatched roles

