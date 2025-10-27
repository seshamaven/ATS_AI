# Name Extraction Fix - Section Headers

## Problem Identified

The resume parser was incorrectly extracting section headers and academic degrees as candidate names instead of the actual person's name.

**Examples:**
- Actual candidate name: "Daniel Mindlin"
- Extracted name: "Education" ❌ (Section header)
- Extracted name: "B.A. in History" ❌ (Academic degree)

## Root Cause

The AI extraction prompt was not explicit enough about avoiding section headers and academic degrees. The parser was confusing:
1. **Section headers** like "Education", "Experience", "Skills"
2. **Academic degrees** like "B.A. in History", "M.S. in Computer Science"

These were being extracted as candidate names instead of the actual person's name.

## Solution Implemented

### 1. Enhanced AI Prompt
Updated the AI extraction prompt to be more explicit:

```
1. full_name – Identify the candidate's ACTUAL PERSONAL NAME (e.g., "Daniel Mindlin", "John Smith"). 
   CRITICAL: Do NOT confuse section headers or labels (like "Education", "Experience", "Skills", "Contact Information") with the person's name.
   The candidate's name is typically at the top of the resume, often centered or on the left.
   It is NEVER a section header. If the first prominent text is a section header, look for the actual name below or above it.
```

### 2. Post-Extraction Validation
Added validation to reject section headers AND academic degrees after AI extraction:

```python
# Validate extracted name - reject section headers and academic degrees
full_name = ai_result.get('full_name', '')
if full_name:
    invalid_names = ['education', 'experience', 'skills', 'contact', 'objective', ...]
    
    # Check for academic degree patterns
    degree_patterns = [
        r'\b([BM]\.?[AS]\.?|MBA|PhD|MD|JD|B\.?Tech|M\.?Tech)\b',
        r'\bin\s+[A-Z][a-z]+',
        r'degree|diploma|certificate'
    ]
    
    is_invalid = False
    if full_name.lower() in invalid_names:
        is_invalid = True
    
    # Check for degree patterns
    for pattern in degree_patterns:
        if re.search(pattern, full_name, re.IGNORECASE):
            is_invalid = True
            break
    
    if is_invalid:
        logger.warning(f"AI extracted invalid name '{full_name}', trying regex fallback...")
        ai_result['full_name'] = self.extract_name(text) or 'Unknown'
```

### 3. Improved Regex Fallback
Enhanced the regex-based name extraction:

- Checks first 10 lines (increased from 5)
- Explicitly skips section headers
- Rejects ALL-CAPS text (likely section headers)
- Validates NLP results too

```python
def extract_name(self, text: str) -> Optional[str]:
    """Extract candidate name from resume text."""
    invalid_names = {'education', 'experience', 'skills', 'contact', 'objective', ...}
    
    lines = text.split('\n')
    for line in lines[:10]:
        line = line.strip()
        # Skip section headers
        if not line or line.lower() in invalid_names:
            continue
        
        # Name validation: 2-4 words, alphabetic, not ALL CAPS
        if line and len(line.split()) <= 4 and len(line) < 50:
            if all(word.isalpha() for word in words):
                if not line.isupper():  # Reject ALL CAPS (section headers)
                    return line
```

### 4. Updated Validation Rules

Added explicit rule in the prompt:
```
- The name field should NEVER contain words like "Education", "Experience", "Skills", "Contact", "Objective", "Summary"
```

## Invalid Names Blacklist

The system now explicitly rejects these common section headers and academic degrees:

**Section Headers:**
- education, experience, skills, contact, objective
- summary, qualifications, work history, professional summary
- references, certifications, projects, achievements

**Academic Degree Patterns:**
- B.A., M.S., B.S., M.A., MBA, PhD, MD, JD
- "in History", "in Computer Science", "in Business"
- "degree", "major", "minor", "diploma", "certificate"

**Validation Checks:**
1. Exact match with section headers
2. Regex match for degree patterns (B.A., M.S., etc.)
3. Keywords like "in", "degree", "major" in the text

## How It Works Now

### Step 1: AI Extraction
- AI analyzes the resume text
- Extracts candidate information
- Returns structured JSON

### Step 2: Validation
- Checks if extracted name is in invalid_names list
- If yes → Uses regex fallback
- If no → Uses AI result

### Step 3: Regex Fallback (if needed)
- Checks first 10 lines
- Skips section headers
- Rejects ALL-CAPS text
- Returns first valid name found

### Step 4: Final Validation
- NLP results are also validated
- Section headers are filtered out
- Only real names pass through

## Testing

To test the fix, try uploading a resume where:
- The candidate name is "Daniel Mindlin"
- The resume has section headers "Education", "Experience", etc.

**Expected result:**
- Extracted name: "Daniel Mindlin" ✅
- NOT "Education" ❌

## Benefits

1. **Accuracy**: Correctly identifies actual candidate names
2. **Reliability**: Multiple layers of validation prevent errors
3. **Robustness**: Handles various resume formats
4. **Fallback**: Even if AI fails, regex backup works

## Backward Compatibility

- Existing resumes in database are unaffected
- New uploads will use improved extraction
- No breaking changes to API

## Future Enhancements

Consider:
- Machine learning model trained specifically on resume names
- Pattern recognition for common name formats
- Confidence scoring for extracted names

