-- ATS Database Schema for Resume Metadata and Embeddings
-- MySQL 8.0+ required for VECTOR support (or use JSON/BLOB as fallback)

CREATE DATABASE IF NOT EXISTS ats_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE ats_db;

-- Resume Metadata Table
CREATE TABLE IF NOT EXISTS resume_metadata (
    candidate_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    phone VARCHAR(50),
    
    -- Experience and Skills
    total_experience FLOAT DEFAULT 0.0 COMMENT 'Total years of experience',
    primary_skills TEXT COMMENT 'Comma-separated primary skills',
    secondary_skills TEXT COMMENT 'Comma-separated secondary skills',
    all_skills TEXT COMMENT 'All extracted skills',
    profile_type VARCHAR(100) DEFAULT 'Generalist' COMMENT 'Primary role classification (Java, .Net, SAP, etc.)',
    role_type VARCHAR(100) COMMENT 'Role type classification',
 
    subrole_type VARCHAR(100) COMMENT 'Sub-role type classification',
    sub_profile_type VARCHAR(100) COMMENT 'Sub-profile type classification',
    
    -- Domain and Industry
    domain VARCHAR(255) COMMENT 'Primary domain/industry',
    sub_domain VARCHAR(255) COMMENT 'Specialization area',
    
    -- Education
    education VARCHAR(500) COMMENT 'Highest education qualification',
    education_details TEXT COMMENT 'All education information',
    
    -- Additional Information
    current_location VARCHAR(255),
    preferred_locations TEXT COMMENT 'Comma-separated preferred locations',
    current_company VARCHAR(255),
    current_designation VARCHAR(255),
    notice_period VARCHAR(100),
    expected_salary VARCHAR(100),
    current_salary VARCHAR(100),
    
    -- Resume Content
    resume_summary TEXT COMMENT 'Auto-generated summary',
    
    -- File Information
    file_name VARCHAR(500),
    file_type VARCHAR(50),
    file_size_kb INT,
    file_base64 LONGTEXT COMMENT 'Base64-encoded resume file content',
    
    -- AI Analysis Fields
    ai_primary_skills JSON COMMENT 'AI-extracted primary skills with experience and weights',
    ai_secondary_skills JSON COMMENT 'AI-extracted secondary skills with experience and weights',
    ai_project_details JSON COMMENT 'AI-extracted project details and skill usage',
    ai_extraction_used BOOLEAN DEFAULT FALSE COMMENT 'Whether AI extraction was used',
    
    -- Status and Metadata
    status VARCHAR(50) DEFAULT 'active' COMMENT 'active, archived, blacklisted',
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes for fast querying
    INDEX idx_name (name),
    INDEX idx_email (email),
    INDEX idx_domain (domain),
    INDEX idx_profile_type (profile_type),
    INDEX idx_role_type (role_type),
    INDEX idx_subrole_type (subrole_type),
    INDEX idx_sub_profile_type (sub_profile_type),
    INDEX idx_experience (total_experience),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at),
    FULLTEXT INDEX idx_skills (primary_skills, secondary_skills, all_skills)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Job Descriptions Table
CREATE TABLE IF NOT EXISTS job_descriptions (
    job_id VARCHAR(100) PRIMARY KEY,
    job_title VARCHAR(255) NOT NULL,
    job_description LONGTEXT NOT NULL,
    
    -- Job Requirements
    required_skills TEXT COMMENT 'Comma-separated required skills',
    preferred_skills TEXT COMMENT 'Comma-separated preferred skills',
    min_experience FLOAT DEFAULT 0.0,
    max_experience FLOAT,
    
    -- Job Details
    domain VARCHAR(255),
    sub_domain VARCHAR(255),
    education_required VARCHAR(500),
    location VARCHAR(255),
    employment_type VARCHAR(50) COMMENT 'Full-time, Part-time, Contract',
    
    -- Extracted Job Metadata (AI-extracted)
    role VARCHAR(255) COMMENT 'Main job title/role (e.g., Software Engineer, Data Scientist)',
    sub_role VARCHAR(50) COMMENT 'Sub-role: Frontend, Backend, or Full Stack',
    profile_type VARCHAR(100) COMMENT 'Primary technology profile (e.g., Java, Python, .Net)',
    profile_sub_type VARCHAR(500) COMMENT 'Specific technologies/frameworks (comma-separated)',
    primary_skills TEXT COMMENT 'Primary technical skills (comma-separated)',
    
    -- Compensation
    salary_range VARCHAR(100),
    
    -- JD Content
    jd_summary TEXT COMMENT 'Auto-generated summary',
    
    -- Embedding
    embedding JSON COMMENT '1536-dimension vector as JSON array',
    embedding_model VARCHAR(100) DEFAULT 'text-embedding-ada-002',
    
    -- Status
    status VARCHAR(50) DEFAULT 'active' COMMENT 'active, closed, on-hold',
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_domain (domain),
    INDEX idx_status (status),
    INDEX idx_experience_range (min_experience, max_experience),
    FULLTEXT INDEX idx_jd (job_description)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Candidate-Job Ranking History
CREATE TABLE IF NOT EXISTS ranking_history (
    ranking_id INT AUTO_INCREMENT PRIMARY KEY,
    job_id VARCHAR(100) NOT NULL,
    candidate_id INT NOT NULL,
    
    -- Scores
    total_score FLOAT NOT NULL,
    match_percent FLOAT NOT NULL,
    
    -- Individual Scores
    skills_score FLOAT DEFAULT 0.0,
    experience_score FLOAT DEFAULT 0.0,
    domain_score FLOAT DEFAULT 0.0,
    education_score FLOAT DEFAULT 0.0,
    
    -- Match Details
    matched_skills TEXT,
    missing_skills TEXT,
    experience_match VARCHAR(50) COMMENT 'High, Medium, Low',
    domain_match VARCHAR(50) COMMENT 'High, Medium, Low',
    
    -- Rankings
    rank_position INT,
    
    -- Metadata
    ranking_algorithm_version VARCHAR(50) DEFAULT 'v1.0',
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign Keys
    FOREIGN KEY (candidate_id) REFERENCES resume_metadata(candidate_id) ON DELETE CASCADE,
    FOREIGN KEY (job_id) REFERENCES job_descriptions(job_id) ON DELETE CASCADE,
    
    -- Indexes
    INDEX idx_job_ranking (job_id, total_score DESC),
    INDEX idx_candidate_history (candidate_id, created_at DESC),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Skills Master Table (Optional - for skill standardization)
CREATE TABLE IF NOT EXISTS skills_master (
    skill_id INT AUTO_INCREMENT PRIMARY KEY,
    skill_name VARCHAR(255) NOT NULL UNIQUE,
    skill_category VARCHAR(100) COMMENT 'Technical, Soft, Domain',
    skill_aliases TEXT COMMENT 'Comma-separated alternative names',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_category (skill_category),
    FULLTEXT INDEX idx_skill_name (skill_name, skill_aliases)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Application Tracking (Optional - for workflow management)
CREATE TABLE IF NOT EXISTS applications (
    application_id INT AUTO_INCREMENT PRIMARY KEY,
    job_id VARCHAR(100) NOT NULL,
    candidate_id INT NOT NULL,
    
    -- Application Status
    status VARCHAR(50) DEFAULT 'applied' COMMENT 'applied, screening, interviewed, selected, rejected',
    stage VARCHAR(100),
    
    -- Scores from ranking
    initial_match_score FLOAT,
    
    -- Interview and Feedback
    interview_feedback TEXT,
    interviewer_rating FLOAT,
    
    -- Timestamps
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Foreign Keys
    FOREIGN KEY (candidate_id) REFERENCES resume_metadata(candidate_id) ON DELETE CASCADE,
    FOREIGN KEY (job_id) REFERENCES job_descriptions(job_id) ON DELETE CASCADE,
    
    -- Indexes
    INDEX idx_job_applications (job_id, status),
    INDEX idx_candidate_applications (candidate_id, applied_at DESC),
    UNIQUE KEY unique_application (job_id, candidate_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Role Processor Table (Stores unique normalized roles with all original roles in JSON)
CREATE TABLE IF NOT EXISTS role_processor (
    Id INT AUTO_INCREMENT PRIMARY KEY,
    Normalised_roles VARCHAR(255) NOT NULL UNIQUE COMMENT 'Normalized role name (e.g., Software Engineer)',
    roles JSON NOT NULL COMMENT 'JSON array of original role names (e.g., [".NET Developer", "Java Developer"])',
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes for fast lookups
    INDEX idx_normalised_role (Normalised_roles)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create Role_Processor Table
-- Stores unique normalized roles with all original roles in JSON array
-- Example: "Software Engineer" -> [".NET Developer", "Java Developer", "Python Developer"]

USE ats_db;

-- Create Role Processor Table
CREATE TABLE IF NOT EXISTS role_processor (
    Id INT AUTO_INCREMENT PRIMARY KEY,
    Normalised_roles VARCHAR(255) NOT NULL UNIQUE COMMENT 'Normalized role name (e.g., Software Engineer)',
    roles JSON NOT NULL COMMENT 'JSON array of original role names (e.g., [".NET Developer", "Java Developer"])',
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes for fast lookups
    INDEX idx_normalised_role (Normalised_roles)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Verify table creation
SELECT 'role_processor table created successfully!' AS status;
DESCRIBE role_processor;


-- Fix Job Description Table Issue
-- This script removes the incorrect 'job_description' table and ensures 'job_descriptions' table exists

USE ats_db;

-- Drop the incorrect 'job_description' table if it exists
DROP TABLE IF EXISTS job_description;

-- Verify that the correct 'job_descriptions' table exists with proper structure
-- If it doesn't exist, create it
CREATE TABLE IF NOT EXISTS job_descriptions (
    job_id VARCHAR(100) PRIMARY KEY,
    job_title VARCHAR(255) NOT NULL,
    job_description LONGTEXT NOT NULL,
    
    -- Job Requirements
    required_skills TEXT COMMENT 'Comma-separated required skills',
    preferred_skills TEXT COMMENT 'Comma-separated preferred skills',
    min_experience FLOAT DEFAULT 0.0,
    max_experience FLOAT,
    
    -- Job Details
    domain VARCHAR(255),
    sub_domain VARCHAR(255),
    education_required VARCHAR(500),
    location VARCHAR(255),
    employment_type VARCHAR(50) COMMENT 'Full-time, Part-time, Contract',
    
    -- Extracted Job Metadata (AI-extracted)
    role VARCHAR(255) COMMENT 'Main job title/role (e.g., Software Engineer, Data Scientist)',
    sub_role VARCHAR(50) COMMENT 'Sub-role: Frontend, Backend, or Full Stack',
    profile_type VARCHAR(100) COMMENT 'Primary technology profile (e.g., Java, Python, .Net)',
    profile_sub_type VARCHAR(500) COMMENT 'Specific technologies/frameworks (comma-separated)',
    primary_skills TEXT COMMENT 'Primary technical skills (comma-separated)',
    
    -- Compensation
    salary_range VARCHAR(100),
    
    -- JD Content
    jd_summary TEXT COMMENT 'Auto-generated summary',
    
    -- Embedding
    embedding JSON COMMENT '1536-dimension vector as JSON array',
    embedding_model VARCHAR(100) DEFAULT 'text-embedding-ada-002',
    
    -- Status
    status VARCHAR(50) DEFAULT 'active' COMMENT 'active, closed, on-hold',
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_domain (domain),
    INDEX idx_status (status),
    INDEX idx_experience_range (min_experience, max_experience),
    INDEX idx_role (role),
    INDEX idx_sub_role (sub_role),
    INDEX idx_profile_type (profile_type),
    FULLTEXT INDEX idx_jd (job_description)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Verify the table structure
SELECT 'job_descriptions table verified successfully!' AS status;
SELECT COUNT(*) AS total_jobs FROM job_descriptions;



-- Migration script to add job metadata columns to job_descriptions table
-- Run this script if you have an existing database
-- Note: This script will fail if columns already exist - that's okay, just means they're already added

-- Add new columns for extracted job metadata
ALTER TABLE job_descriptions
ADD COLUMN role VARCHAR(255) COMMENT 'Main job title/role (e.g., Software Engineer, Data Scientist)' AFTER employment_type,
ADD COLUMN sub_role VARCHAR(50) COMMENT 'Sub-role: Frontend, Backend, or Full Stack' AFTER role,
ADD COLUMN profile_type VARCHAR(100) COMMENT 'Primary technology profile (e.g., Java, Python, .Net)' AFTER sub_role,
ADD COLUMN profile_sub_type VARCHAR(500) COMMENT 'Specific technologies/frameworks (comma-separated)' AFTER profile_type,
ADD COLUMN primary_skills TEXT COMMENT 'Primary technical skills (comma-separated)' AFTER profile_sub_type;

-- Add indexes for better query performance (will fail if indexes already exist - that's okay)
ALTER TABLE job_descriptions
ADD INDEX idx_role (role),
ADD INDEX idx_sub_role (sub_role),
ADD INDEX idx_profile_type (profile_type);

-- Migration Script: Add role_type, subrole_type, sub_profile_type columns to resume_metadata
-- Date: 2024
-- Description: Adds three new columns for role classification hierarchy

USE ats_db;

-- Add role_type column
ALTER TABLE resume_metadata 
ADD COLUMN role_type VARCHAR(100) NULL COMMENT 'Primary role type classification' 
AFTER profile_type;

-- Add subrole_type column
ALTER TABLE resume_metadata 
ADD COLUMN subrole_type VARCHAR(100) NULL COMMENT 'Sub-role type classification' 
AFTER role_type;

-- Add sub_profile_type column
ALTER TABLE resume_metadata 
ADD COLUMN sub_profile_type VARCHAR(100) NULL COMMENT 'Sub-profile type classification' 
AFTER subrole_type;

-- Add indexes for faster querying
CREATE INDEX idx_role_type ON resume_metadata(role_type);
CREATE INDEX idx_subrole_type ON resume_metadata(subrole_type);
CREATE INDEX idx_sub_profile_type ON resume_metadata(sub_profile_type);

-- Verify the changes
DESCRIBE resume_metadata;


-- Chat History Table for storing AI Search conversations
-- Run this migration to create the Chat_history table

CREATE TABLE IF NOT EXISTS Chat_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    Candidate_id INT NULL COMMENT 'Reference to candidate if applicable',
    Chat_msg TEXT NOT NULL COMMENT 'User input/query message',
    role VARCHAR(100) COMMENT 'User role classification',
    sub_role VARCHAR(100) COMMENT 'User sub-role classification',
    profile_type VARCHAR(100) COMMENT 'Profile type from search context',
    sub_profile_type VARCHAR(100) COMMENT 'Sub-profile type from search context',
    response LONGTEXT COMMENT 'AI/system response to the user query',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for faster querying
    INDEX idx_candidate_id (Candidate_id),
    INDEX idx_role (role),
    INDEX idx_profile_type (profile_type),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Add comment to describe the table purpose
ALTER TABLE Chat_history COMMENT = 'Stores user queries and AI responses from the AI Search API';

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

-- Migrate Role_Processor Table to New Structure
-- Changes from: One row per mapping (normalized_role, original_role)
-- To: One row per normalized role with all original roles stored as JSON

USE ats_db;

-- Step 1: Create backup of existing data
CREATE TABLE IF NOT EXISTS role_processor_backup AS SELECT * FROM role_processor;

-- Step 2: Drop existing table
DROP TABLE IF EXISTS role_processor;

-- Step 3: Create new table structure
CREATE TABLE role_processor (
    Id INT AUTO_INCREMENT PRIMARY KEY,
    Normalised_roles VARCHAR(255) NOT NULL UNIQUE COMMENT 'Normalized role name (e.g., Software Engineer)',
    roles JSON NOT NULL COMMENT 'JSON array of original role names (e.g., [".NET Developer", "Java Developer"])',
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes for fast lookups
    INDEX idx_normalised_role (Normalised_roles)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Step 4: Migrate data from backup to new structure
-- Group by normalized role and aggregate original roles into JSON array
INSERT INTO role_processor (Normalised_roles, roles)
SELECT 
    Normalised_roles,
    JSON_ARRAYAGG(roles) as roles
FROM role_processor_backup
GROUP BY Normalised_roles
ORDER BY Normalised_roles;

-- Step 5: Verify migration
SELECT 
    'Migration completed!' AS status,
    COUNT(*) AS total_normalized_roles,
    (SELECT SUM(JSON_LENGTH(roles)) FROM role_processor) AS total_original_roles
FROM role_processor;

-- Step 6: Show sample data
SELECT 
    Id,
    Normalised_roles,
    JSON_LENGTH(roles) AS number_of_roles,
    JSON_PRETTY(roles) AS roles
FROM role_processor
LIMIT 5;

-- Optional: Drop backup table after verification (uncomment if satisfied)
-- DROP TABLE IF EXISTS role_processor_backup;

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


