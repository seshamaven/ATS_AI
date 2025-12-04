#!/usr/bin/env python3 adding to the github
"""
Quick script to check Chat_history data
"""
from ats_database import ATSDatabase
from datetime import datetime

print("=" * 60)
print("Chat History Checker")
print("=" * 60)
print()

db = ATSDatabase()
if db.connect():
    # Get total count
    db.cursor.execute("SELECT COUNT(*) as count FROM Chat_history")
    total = db.cursor.fetchone()['count']
    print(f"📊 Total Records: {total}")
    print()
    
    if total > 0:
        # Get latest 10 records
        db.cursor.execute("""
            SELECT 
                id, Chat_msg, role, profile_type, 
                DATE_FORMAT(created_at, '%Y-%m-%d %H:%i:%s') as created_at
            FROM Chat_history 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        records = db.cursor.fetchall()
        
        print("📋 Latest Chat History:")
        print("-" * 60)
        print(f"{'ID':<5} {'Query':<25} {'Role':<15} {'Profile':<15} {'Created At'}")
        print("-" * 60)
        
        for r in records:
            print(f"{r['id']:<5} {r['Chat_msg'][:24]:<25} {str(r['role'])[:14]:<15} {str(r['profile_type'])[:14]:<15} {r['created_at']}")
        
        print("-" * 60)
    else:
        print("⚠️  No records found in Chat_history table")
    
    db.disconnect()
else:
    print("❌ Failed to connect to database")

print()
print("=" * 60)


