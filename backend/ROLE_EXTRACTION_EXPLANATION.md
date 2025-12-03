# Role Extraction - Single Role vs Multiple Roles

## Answer: **Returns ONLY ONE ROLE** (Not All Roles)

---

## Key Points

### ✅ **1. API Returns ONE Role Only**

**Location**: `ats_api.py` lines 2624-2647

```python
# Only ONE role is extracted and returned
extracted_job_metadata['role'] = 'Business Analyst'  # Single role, not a list
```

**Response Structure**:
```json
{
  "extracted_job_metadata": {
    "role": "Business Analyst",  ← Only ONE role (string, not array)
    "sub_role": null,
    "profile_type": "Business Intelligence (BI)",
    "primary_skills": ["SQL", "Excel", "data visualization"],
    "secondary_skills": ["data visualization"]
  }
}
```

### ✅ **2. For Business Analyst - Skills ARE Collected**

**Yes, skills are collected for Business Analyst role!**

**How it works**:
1. **Role Detection** (line 2636-2637):
   ```python
   if 'business analyst' in searchable_lower:
       extracted_job_metadata['role'] = 'Business Analyst'
   ```

2. **Skills Extraction** (line 2486-2487):
   ```python
   from resume_parser import extract_skills_from_text
   extracted_skills = extract_skills_from_text(job_description)
   ```
   - Uses `TECH_SKILLS` dictionary from `skill_extractor.py`
   - Matches skills from job description against `TECH_SKILLS`
   - Returns all matching skills

3. **BI Skills Addition** (lines 2506-2552):
   ```python
   if bi_keywords_detected:
       # Add BI skills if mentioned in job description
       bi_skills_to_check = ['SQL', 'Excel', 'Power BI', 'Tableau', ...]
       # If very few skills, add default BI skills
       default_bi_skills = ['SQL', 'Excel', 'data visualization']
   ```

---

## Role Extraction Logic

### **Priority Order**:

1. **From Request** (`job_title` parameter):
   ```python
   job_title = data.get('job_title', '')
   if job_title:
       extracted_job_metadata['role'] = job_title  # Use provided role
   ```

2. **Pattern Matching** (for specific roles):
   ```python
   if 'data analyst' in searchable_lower:
       role = 'Data Analyst'
   elif 'business analyst' in searchable_lower:
       role = 'Business Analyst'  # ← Returns this if found
   elif 'business intelligence' in searchable_lower:
       role = 'Business Intelligence Analyst'
   ```

3. **ROLE_MAPPING Dictionary** (for other roles):
   ```python
   role_info = detect_subrole_from_query(searchable_text)
   if role_info:
       role = role_info.get('main_role') or role_info.get('sub_role')
   ```

4. **Default Fallback**:
   ```python
   else:
       role = 'Software Engineer'  # Default if nothing found
   ```

---

## Example Scenarios

### **Scenario 1: Business Analyst JD**

**Input**:
```json
{
  "job_description": "We are looking for a Business Analyst with SQL, Excel, and data visualization skills..."
}
```

**Output**:
```json
{
  "extracted_job_metadata": {
    "role": "Business Analyst",  ← Only ONE role
    "sub_role": null,
    "profile_type": "Business Intelligence (BI)",
    "primary_skills": ["SQL", "Excel", "data visualization"],
    "secondary_skills": []
  }
}
```

**What Happened**:
1. ✅ Detected "Business Analyst" in job description
2. ✅ Extracted skills: SQL, Excel, data visualization
3. ✅ Detected BI keywords → Set profile_type to "Business Intelligence (BI)"
4. ✅ Returned ONE role: "Business Analyst"

---

### **Scenario 2: Multiple Roles Mentioned in JD**

**Input**:
```json
{
  "job_description": "We need a Business Analyst and Data Analyst to work together..."
}
```

**Output**:
```json
{
  "extracted_job_metadata": {
    "role": "Business Analyst",  ← Only FIRST role found (not both)
    "sub_role": null,
    "profile_type": "Business Intelligence (BI)",
    "primary_skills": ["SQL", "Excel", "data visualization"],
    "secondary_skills": []
  }
}
```

**What Happened**:
1. ✅ Found "Business Analyst" first (line 2636)
2. ✅ Returned "Business Analyst" (stops checking after first match)
3. ❌ "Data Analyst" is ignored (not returned)

**Code Logic**:
```python
if 'data analyst' in searchable_lower:
    role = 'Data Analyst'
elif 'business analyst' in searchable_lower:  # ← Checks this second
    role = 'Business Analyst'  # ← Returns this if found
```

---

### **Scenario 3: Generic JD (No Specific Role)**

**Input**:
```json
{
  "job_description": "We need someone with Java, Spring Boot, and MySQL experience..."
}
```

**Output**:
```json
{
  "extracted_job_metadata": {
    "role": "Software Engineer",  ← Default role
    "sub_role": "Backend",
    "profile_type": "Java",
    "primary_skills": ["Java", "Spring Boot", "MySQL"],
    "secondary_skills": []
  }
}
```

**What Happened**:
1. ❌ No "Business Analyst" or "Data Analyst" found
2. ✅ Used `detect_subrole_from_query()` to find role from ROLE_MAPPING
3. ✅ Found "Java Developer" or similar → Returns main role "Software Engineer"
4. ✅ Extracted skills: Java, Spring Boot, MySQL

---

## ROLE_MAPPING Dictionary

**Location**: `ats_api.py` lines 1630-1810

**Contains**:
```python
ROLE_MAPPING = [
    ("Software Engineer", {
        "Java Developer", "Python Developer", "Full Stack Developer", ...
    }),
    ("Analyst", {
        "Data Analyst", "Business Analyst", "Business Intelligence Analyst", ...
    }),
    ("Business Analyst", {
        "Business Analyst", "IT Analyst", "Functional Analyst", ...
    }),
    # ... more roles ...
]
```

**How it works**:
- Searches job description for sub-roles (like "Java Developer", "Business Analyst")
- Returns the **main role** (like "Software Engineer" or "Analyst")
- **Returns only ONE main role**, not all matching roles

---

## Skills Collection for Business Analyst

### ✅ **YES, Skills ARE Collected**

**Process**:
1. **Extract from Job Description**:
   ```python
   extracted_skills = extract_skills_from_text(job_description)
   # Uses TECH_SKILLS dictionary
   ```

2. **Filter Non-Technical Words**:
   ```python
   non_skill_words = {'dashboards', 'datasets', 'insights', ...}
   extracted_skills = [s for s in extracted_skills 
                      if s.lower() not in non_skill_words]
   ```

3. **Add BI Skills if BI Keywords Detected**:
   ```python
   if bi_keywords_detected:
       # Check if BI skills mentioned
       bi_skills_to_check = ['SQL', 'Excel', 'Power BI', 'Tableau', ...]
       
       # If very few skills, add default
       if len(real_skills) < 3:
           default_bi_skills = ['SQL', 'Excel', 'data visualization']
           extracted_skills.extend(default_bi_skills)
   ```

4. **Split into Primary and Secondary**:
   ```python
   primary_skills = extracted_skills[:10]  # First 10
   secondary_skills = extracted_skills[10:]  # Rest
   ```

---

## Summary Table

| Question | Answer |
|----------|--------|
| **Returns all roles?** | ❌ **NO** - Returns only ONE role |
| **Returns multiple roles?** | ❌ **NO** - Single role string, not array |
| **Collects skills for Business Analyst?** | ✅ **YES** - Uses TECH_SKILLS dictionary |
| **What if multiple roles in JD?** | Returns FIRST role found (priority order) |
| **What if no role found?** | Returns default: "Software Engineer" |
| **Skills source?** | TECH_SKILLS dictionary in skill_extractor.py |

---

## Code Locations

1. **Role Extraction**: `ats_api.py` lines 2624-2647
2. **Skills Extraction**: `ats_api.py` lines 2486-2572
3. **ROLE_MAPPING**: `ats_api.py` lines 1630-1810
4. **TECH_SKILLS**: `skill_extractor.py` lines 20-1000+

---

## Conclusion

### ✅ **For Business Analyst (or ANY role)**:
- ✅ Skills ARE collected from `TECH_SKILLS` dictionary
- ✅ Skills are extracted from job description text
- ✅ Additional BI skills may be added if BI keywords detected

### ❌ **Role Return**:
- ❌ Returns ONLY ONE role (not all roles)
- ❌ Not an array/list of roles
- ❌ If multiple roles in JD, returns FIRST one found

### 📝 **Example Response**:
```json
{
  "extracted_job_metadata": {
    "role": "Business Analyst",  ← Single role (string)
    "primary_skills": ["SQL", "Excel", "data visualization"],  ← Skills collected
    "secondary_skills": []
  }
}
```

**The API extracts and returns ONE role per job description, but collects ALL matching skills from the TECH_SKILLS dictionary.**



