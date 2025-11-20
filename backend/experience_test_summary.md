# Experience Extraction Test Results

## Test Date: 2025-11-20

### Summary
- **Total Files Tested:** 6
- **Successful Extractions:** 5
- **Average Experience:** 7.54 years

### Detailed Results

#### 1. 524538938.pdf (Carol Forsberg)
- **Extracted Experience:** 0.0 years
- **Segments Found:** 0
- **Status:** ✅ Correct (Fresher resume - no experience section)
- **Notes:** This is the fresher resume we tested earlier. Correctly identified as 0 years.

#### 2. 524573810.pdf
- **Extracted Experience:** 17.89 years
- **Segments Found:** 1
- **Date Range:** 2008-01 to 2025-11
- **Status:** ✅ Looks correct
- **Notes:** ~18 years of experience calculated from date ranges.

#### 3. 524691788.pdf (Rachel Morris)
- **Extracted Experience:** 32.89 years
- **Segments Found:** 1
- **Date Range:** 1993-01 to 2025-11
- **Status:** ⚠️ **ISSUE FOUND**
- **Expected:** 25 years (from "25+ years' experience" in Overview)
- **Actual:** 32.89 years (calculated from dates)
- **Notes:** 
  - Resume mentions "25+ years' experience" in Overview section
  - Explicit experience should take priority but is not being used
  - Calculated experience (32.89) is higher than explicit (25), so explicit is ignored
  - **Action Needed:** Review explicit experience priority logic

#### 4. 524574773.pdf
- **Extracted Experience:** 0.33 years
- **Segments Found:** 1
- **Date Range:** 2020-01 to 2020-04
- **Status:** ✅ Looks correct
- **Notes:** Very short experience period (4 months).

#### 5. 524695733.pdf
- **Extracted Experience:** 6.0 years
- **Segments Found:** 1
- **Date Range:** 2014-01 to 2019-12
- **Status:** ✅ Looks correct
- **Notes:** 6 years of experience from 2014 to 2019.

#### 6. 524697389.pdf
- **Extracted Experience:** 6.0 years
- **Segments Found:** 1
- **Date Range:** 2014-01 to 2019-12
- **Status:** ✅ Looks correct
- **Notes:** Same as file 5 (duplicate?).

### Issues Found

1. **Explicit Experience Priority Issue (524691788.pdf)**
   - Problem: Explicit experience (25 years) is not being used when calculated experience (32.89 years) is higher
   - Current Logic: Only uses explicit if `explicit > total_years`
   - Recommendation: Should always prioritize explicit experience when found, regardless of calculated value
   - Location: `backend/experience_extractor.py` line 419

2. **Encoding Issue**
   - Problem: Unicode characters in PDF text cause encoding errors when printing
   - Status: Fixed in test script (using error handling)
   - Impact: Minor - doesn't affect extraction, only display

### Recommendations

1. **Fix Explicit Experience Priority:**
   - Change logic to always use explicit experience when found
   - Current: `if explicit and explicit > total_years:`
   - Suggested: `if explicit: total_years = explicit`

2. **Test Apostrophe Pattern:**
   - Verify that "25+ years' experience" pattern is being matched correctly
   - Check if apostrophe character encoding is causing issues

3. **Add Logging:**
   - Log when explicit experience is found but not used
   - Log when explicit experience takes priority

