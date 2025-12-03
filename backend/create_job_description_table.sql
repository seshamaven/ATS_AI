-- Create job_description table (singular) with proper structure
-- This table stores extracted job metadata

USE ats_db;

-- Drop table if exists and recreate with correct structure
DROP TABLE IF EXISTS job_description;

-- Create job_description table with proper column sizes
CREATE TABLE job_description (
    job_id INT AUTO_INCREMENT PRIMARY KEY,
    role VARCHAR(255) COMMENT 'Main job title/role (e.g., Software Engineer, Data Scientist)',
    sub_role VARCHAR(50) COMMENT 'Sub-role: Frontend, Backend, or Full Stack',
    profile_type VARCHAR(100) COMMENT 'Primary technology profile (e.g., Java, Python, .Net)',
    profile_sub_type VARCHAR(500) COMMENT 'Specific technologies/frameworks (comma-separated)',
    primary_skills TEXT COMMENT 'Primary technical skills (comma-separated)',
    secondary_skills TEXT COMMENT 'Secondary technical skills (comma-separated)',
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_role (role),
    INDEX idx_sub_role (sub_role),
    INDEX idx_profile_type (profile_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Verify table creation
SELECT 'job_description table created successfully!' AS status;
DESCRIBE job_description;

