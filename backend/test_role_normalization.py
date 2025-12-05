"""
Test script for role normalization functionality.
Tests the normalize_role_from_resume function.
"""

import logging
from role_processor import RoleProcessor, normalize_role

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Reset@123',
    'database': 'ats_db',
    'port': 3306
}


def test_role_normalization():
    """Test the role normalization function with various examples."""
    
    print("=" * 60)
    print("Testing Role Normalization")
    print("=" * 60)
    print()
    
    # Test cases: (original_role, expected_normalized_role)
    test_cases = [
        # Should match Software Engineer
        ("backend java developer", "Software Engineer"),
        ("asp.net web developer", "Software Engineer"),
        ("Senior .NET Developer", "Software Engineer"),
        ("Python Web Developer", "Software Engineer"),
        ("Full Stack Developer", "Software Engineer"),
        ("C# Developer", "Software Engineer"),
        
        # Should match Architect
        ("Solution Architect", "Architect"),
        ("Enterprise Architect", "Architect"),
        ("Software Architect", "Architect"),
        
        # Should match Consultant
        ("SAP Consultant", "Consultant"),
        ("Business Consultant", "Consultant"),
        ("HR Consultant", "Consultant"),
        
        # Should match Data Analyst
        ("Business Analyst", "Data Analyst"),  # Note: Some business analysts map to Data Analyst
        ("Data Analyst", "Data Analyst"),
        ("BI Business Analyst", "Data Analyst"),
        
        # Should match Database Administrator
        ("Oracle DBA", "Database Administrator"),
        ("Database Administrator", "Database Administrator"),
        ("SQL Server DBA", "Database Administrator"),
        
        # Should match Project Manager
        ("IT Project Manager", "Project Manager"),
        ("Project Manager", "Project Manager"),
        ("Technical Project Manager", "Project Manager"),
        
        # Should return "Others" for unknown roles
        ("Unknown Role XYZ", "Others"),
        ("Random Job Title", "Others"),
        ("", "Others"),
    ]
    
    try:
        with RoleProcessor(config=DB_CONFIG) as rp:
            passed = 0
            failed = 0
            
            for original_role, expected in test_cases:
                result = rp.normalize_role_from_resume(original_role)
                status = "✓ PASS" if result == expected else "✗ FAIL"
                
                if result == expected:
                    passed += 1
                else:
                    failed += 1
                
                print(f"{status}: '{original_role}'")
                print(f"    Expected: '{expected}'")
                print(f"    Got:      '{result}'")
                if result != expected:
                    print(f"    ⚠ Mismatch!")
                print()
            
            print("=" * 60)
            print(f"Test Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
            print("=" * 60)
            
            # Test standalone function
            print("\n" + "=" * 60)
            print("Testing Standalone Function")
            print("=" * 60)
            print()
            
            test_roles = [
                "Java Developer",
                "Senior Software Engineer",
                "Unknown Role",
                "Database Administrator"
            ]
            
            for role in test_roles:
                result = normalize_role(role, config=DB_CONFIG)
                print(f"normalize_role('{role}') -> '{result}'")
            
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        return 1
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(test_role_normalization())

