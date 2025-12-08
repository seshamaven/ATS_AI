"""
Simple test runner script for resume upload tests.
Run this script to execute all tests: python run_tests.py
"""
import subprocess
import sys
import os

def main():
    """Run pytest tests with verbose output."""
    # Change to backend directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run pytest
    result = subprocess.run(
        ['pytest', 'test_resume_upload.py', '-v', '--tb=short'],
        capture_output=False
    )
    
    return result.returncode

if __name__ == '__main__':
    sys.exit(main())

