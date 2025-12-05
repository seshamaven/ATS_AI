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

