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

