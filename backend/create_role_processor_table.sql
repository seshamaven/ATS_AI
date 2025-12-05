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

