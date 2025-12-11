-- ============================================
-- UPDATED NAME SEARCH QUERY WITH WORD-BY-WORD SOUNDEX
-- ============================================
-- This query matches names like:
-- - "hasena" → "shaik Hasin Begum" (matches "Hasin" word)
-- - "seshu" → "Sesha Kumar" (matches "Sesha" word)
-- 
-- Performance: Optimized for 180k+ records with proper indexes

-- Example query structure (for reference):
SELECT 
    rm.candidate_id,
    rm.name,
    rm.email,
    rm.phone,
    rm.total_experience,
    -- ... other fields ...
    CASE 
        -- Exact match (highest priority)
        WHEN LOWER(rm.name) = :query_lower THEN 10
        
        -- Partial match (substring)
        WHEN LOWER(rm.name) LIKE CONCAT('%', :query_lower, '%') THEN 8
        
        -- Full name SOUNDEX match
        WHEN SOUNDEX(rm.name) = SOUNDEX(:query) THEN 7
        
        -- Word-by-word SOUNDEX matching (NEW)
        -- First word
        WHEN SOUNDEX(SUBSTRING_INDEX(rm.name, ' ', 1)) = SOUNDEX(:query) THEN 6
        -- Second word (if exists)
        WHEN (LENGTH(rm.name) - LENGTH(REPLACE(rm.name, ' ', '')) >= 1 
              AND SOUNDEX(SUBSTRING_INDEX(SUBSTRING_INDEX(rm.name, ' ', 2), ' ', -1)) = SOUNDEX(:query)) THEN 6
        -- Third word (if exists)
        WHEN (LENGTH(rm.name) - LENGTH(REPLACE(rm.name, ' ', '')) >= 2
              AND SOUNDEX(SUBSTRING_INDEX(SUBSTRING_INDEX(rm.name, ' ', 3), ' ', -1)) = SOUNDEX(:query)) THEN 6
        -- Last word
        WHEN SOUNDEX(SUBSTRING_INDEX(rm.name, ' ', -1)) = SOUNDEX(:query) THEN 6
        
        ELSE 0
    END AS name_match_score
FROM resume_metadata rm
WHERE rm.status = 'active'
  AND (
      -- Exact match
      LOWER(rm.name) = :query_lower
      
      -- Partial match (substring)
      OR LOWER(rm.name) LIKE CONCAT('%', :query_lower, '%')
      
      -- Full name SOUNDEX
      OR SOUNDEX(rm.name) = SOUNDEX(:query)
      
      -- Word-by-word SOUNDEX (NEW)
      -- First word
      OR SOUNDEX(SUBSTRING_INDEX(rm.name, ' ', 1)) = SOUNDEX(:query)
      -- Second word (if exists)
      OR (LENGTH(rm.name) - LENGTH(REPLACE(rm.name, ' ', '')) >= 1 
          AND SOUNDEX(SUBSTRING_INDEX(SUBSTRING_INDEX(rm.name, ' ', 2), ' ', -1)) = SOUNDEX(:query))
      -- Third word (if exists)
      OR (LENGTH(rm.name) - LENGTH(REPLACE(rm.name, ' ', '')) >= 2
          AND SOUNDEX(SUBSTRING_INDEX(SUBSTRING_INDEX(rm.name, ' ', 3), ' ', -1)) = SOUNDEX(:query))
      -- Last word
      OR SOUNDEX(SUBSTRING_INDEX(rm.name, ' ', -1)) = SOUNDEX(:query)
  )
ORDER BY name_match_score DESC, rm.total_experience DESC
LIMIT 25;

-- ============================================
-- SCORING BREAKDOWN:
-- ============================================
-- 10 = Exact match: "hasena" = "hasena"
-- 8  = Partial match: "hasena" in "hasena khan"
-- 7  = Full name SOUNDEX: SOUNDEX("hasena") = SOUNDEX("hasina")
-- 6  = Word-by-word SOUNDEX: "hasena" matches "Hasin" in "shaik Hasin Begum"
-- 0  = No match

-- ============================================
-- PERFORMANCE NOTES:
-- ============================================
-- 1. Requires indexes: idx_status_name (composite)
-- 2. SOUNDEX functions can't use indexes (full table scan on filtered results)
-- 3. With 180k records: Expected time 1-3 seconds (with indexes)
-- 4. Without indexes: 2-8 seconds (too slow)


