-- Test script to verify Chat_history data
-- Run this in MySQL Workbench

USE ats_db;

-- Check if table exists
SHOW TABLES LIKE 'Chat_history';

-- Count records
SELECT COUNT(*) as total_records FROM Chat_history;

-- Show all records
SELECT 
    id,
    Chat_msg AS 'Your Query',
    role AS 'Role',
    profile_type AS 'Profile Type',
    DATE_FORMAT(created_at, '%Y-%m-%d %H:%i:%s') AS 'Date & Time'
FROM Chat_history 
ORDER BY created_at DESC;

-- Show full details of latest record
SELECT * FROM Chat_history ORDER BY created_at DESC LIMIT 1;

