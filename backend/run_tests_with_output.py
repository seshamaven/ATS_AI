"""
Test runner that captures and displays full output.
"""
import subprocess
import sys
import os

def main():
    """Run pytest tests and display output."""
    # Change to backend directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 80)
    print("Running Resume Upload Tests")
    print("=" * 80)
    print()
    
    # Run pytest with verbose output
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'test_resume_upload.py', '-v', '--tb=short', '--color=no'],
            capture_output=False,
            text=True
        )
        
        print()
        print("=" * 80)
        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print(f"❌ Tests failed with exit code: {result.returncode}")
        print("=" * 80)
        
        return result.returncode
        
    except FileNotFoundError:
        print("❌ Error: pytest not found. Please install it with: pip install pytest")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())

