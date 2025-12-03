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

