-- Migration Script: Normalize .Net Profile Type Variations
-- This script fixes the mismatch between "Net" and ".Net" profile types
-- Run this once to update existing database records

USE ats_db;

-- Update all variations of .Net to the canonical form ".Net"
UPDATE resume_metadata 
SET profile_type = '.Net' 
WHERE LOWER(profile_type) IN ('net', '.net', 'dotnet')
   OR LOWER(REPLACE(profile_type, '.', '')) = 'net';

-- Verify the changes
SELECT 
    profile_type,
    COUNT(*) as count
FROM resume_metadata
WHERE LOWER(profile_type) LIKE '%net%'
GROUP BY profile_type;

-- Expected result: Should show only ".Net" (with dot) after migration

