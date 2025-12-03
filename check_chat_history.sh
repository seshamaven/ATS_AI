#!/bin/bash
# Quick script to check Chat_history data

echo "=========================================="
echo "Checking Chat_history Table"
echo "=========================================="
echo ""

/usr/bin/mysql -u root -p'Mst@2026' ats_db << EOF
SELECT 
    id,
    Chat_msg as 'User Query',
    role as 'Role',
    profile_type as 'Profile Type',
    DATE_FORMAT(created_at, '%Y-%m-%d %H:%i:%s') as 'Created At'
FROM Chat_history 
ORDER BY created_at DESC 
LIMIT 10;

SELECT COUNT(*) as 'Total Records' FROM Chat_history;
EOF

echo ""
echo "=========================================="

