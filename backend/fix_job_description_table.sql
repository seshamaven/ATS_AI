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

