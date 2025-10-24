#!/usr/bin/env python3
"""
MySQL Connection Test Script for Railway
Tests connection to MySQL database using Railway environment variables
"""

import os
import sys
import mysql.connector
from mysql.connector import Error
from datetime import datetime

def test_mysql_connection():
    """Test MySQL connection using Railway environment variables."""
    
    print("🔍 Testing MySQL Connection in Railway...")
    print("=" * 50)
    
    # Get Railway MySQL environment variables
    mysql_config = {
        'host': os.getenv('MYSQLHOST', 'localhost'),
        'user': os.getenv('MYSQLUSER', 'root'),
        'password': os.getenv('MYSQLPASSWORD', 'root'),
        'database': os.getenv('MYSQLDATABASE', 'ats_db'),
        'port': int(os.getenv('MYSQLPORT', '3306'))
    }
    
    print("📋 Connection Configuration:")
    print(f"   Host: {mysql_config['host']}")
    print(f"   User: {mysql_config['user']}")
    print(f"   Database: {mysql_config['database']}")
    print(f"   Port: {mysql_config['port']}")
    print(f"   Password: {'*' * len(mysql_config['password']) if mysql_config['password'] else 'Not set'}")
    print()
    
    connection = None
    cursor = None
    
    try:
        print("🔄 Attempting to connect...")
        connection = mysql.connector.connect(**mysql_config)
        
        if connection.is_connected():
            print("✅ SUCCESS: Connected to MySQL!")
            
            # Get server info
            cursor = connection.cursor()
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            print(f"📊 MySQL Version: {version[0]}")
            
            # Test database operations
            cursor.execute("SHOW DATABASES")
            databases = cursor.fetchall()
            print(f"📁 Available Databases: {len(databases)}")
            
            # Test current database
            cursor.execute("SELECT DATABASE()")
            current_db = cursor.fetchone()
            print(f"🎯 Current Database: {current_db[0]}")
            
            # Test table creation/access
            try:
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
                print(f"📋 Tables in current database: {len(tables)}")
                if tables:
                    print("   Tables:", [table[0] for table in tables])
            except Error as e:
                print(f"⚠️  Could not list tables: {e}")
            
            print()
            print("🎉 MySQL connection test PASSED!")
            return True
            
    except Error as e:
        print(f"❌ FAILED: Could not connect to MySQL")
        print(f"   Error: {e}")
        print()
        print("🔧 Troubleshooting Tips:")
        print("   1. Check if MySQL service is running in Railway")
        print("   2. Verify environment variables are set correctly")
        print("   3. Check Railway service logs for MySQL errors")
        print("   4. Ensure MySQL service is not in maintenance mode")
        return False
        
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()
            print("🔌 Connection closed.")

def test_environment_variables():
    """Test if Railway MySQL environment variables are available."""
    print("🔍 Checking Railway Environment Variables...")
    print("=" * 50)
    
    required_vars = ['MYSQLHOST', 'MYSQLUSER', 'MYSQLPASSWORD', 'MYSQLDATABASE', 'MYSQLPORT']
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {'*' * len(value) if 'PASSWORD' in var else value}")
        else:
            print(f"❌ {var}: Not set")
            missing_vars.append(var)
    
    print()
    if missing_vars:
        print("⚠️  Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print()
        print("🔧 To fix this:")
        print("   1. Go to Railway dashboard")
        print("   2. Click on your MySQL service")
        print("   3. Go to 'Variables' tab")
        print("   4. Ensure all MySQL variables are set")
        return False
    else:
        print("✅ All required environment variables are present!")
        return True

if __name__ == "__main__":
    print(f"🚀 MySQL Connection Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test environment variables first
    env_ok = test_environment_variables()
    print()
    
    if env_ok:
        # Test actual connection
        connection_ok = test_mysql_connection()
        
        if connection_ok:
            print("🎯 RESULT: MySQL is properly connected and working!")
            sys.exit(0)
        else:
            print("💥 RESULT: MySQL connection failed!")
            sys.exit(1)
    else:
        print("💥 RESULT: Environment variables are missing!")
        sys.exit(1)
