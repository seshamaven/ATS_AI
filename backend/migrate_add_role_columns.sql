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


