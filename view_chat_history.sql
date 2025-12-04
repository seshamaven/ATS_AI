-- ==========================================
-- Chat History Viewer for MySQL Workbench
-- ==========================================
-- Copy and paste this entire file into MySQL Workbench
-- Then click Execute (or press Ctrl+Enter)

USE ats_db;

-- Step 1: Verify you're in the correct database
SELECT DATABASE() as 'Current Database';

-- Step 2: Check table exists
SHOW TABLES LIKE 'Chat_history';

-- Step 3: Count total records
SELECT COUNT(*) as 'Total Records' FROM Chat_history;

-- Step 4: View all data (simple format)
SELECT 
    id,
    Chat_msg AS 'Your Query',
    role AS 'Role',
    profile_type AS 'Profile Type',
    DATE_FORMAT(created_at, '%Y-%m-%d %H:%i:%s') AS 'Date & Time'
FROM Chat_history 
ORDER BY created_at DESC;

-- Step 5: View latest record with all details
SELECT * FROM Chat_history ORDER BY id DESC LIMIT 1;

-- Step 6: View all columns for all records
SELECT * FROM Chat_history ORDER BY created_at DESC LIMIT 10;


