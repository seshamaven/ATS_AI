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
    resume_text LONGTEXT COMMENT 'Full parsed resume text',
    resume_summary TEXT COMMENT 'Auto-generated summary',
    
    -- File Information
    file_name VARCHAR(500),
    file_type VARCHAR(50),
    file_size_kb INT,
    
    -- Embedding (Store as JSON for compatibility)
    embedding JSON COMMENT '1536-dimension vector as JSON array',
    embedding_model VARCHAR(100) DEFAULT 'text-embedding-ada-002',
    
    -- Status and Metadata
    status VARCHAR(50) DEFAULT 'active' COMMENT 'active, archived, blacklisted',
    source VARCHAR(100) COMMENT 'Upload source',
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes for fast querying
    INDEX idx_name (name),
    INDEX idx_email (email),
    INDEX idx_domain (domain),
    INDEX idx_experience (total_experience),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at),
    FULLTEXT INDEX idx_skills (primary_skills, secondary_skills, all_skills),
    FULLTEXT INDEX idx_resume_text (resume_text)
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

