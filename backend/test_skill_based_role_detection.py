"""
Test script for skill-based role detection for fresher resumes.
Tests the normalize_role function when designation is missing.
"""

import logging
from role_processor import normalize_role

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


def test_skill_based_detection():
    """Test skill-based role detection for fresher resumes."""
    
    print("=" * 70)
    print("Testing Skill-Based Role Detection for Fresher Resumes")
    print("=" * 70)
    print()
    
    # Test cases: fresher resumes without designation
    test_cases = [
        {
            "name": "Fresher with Java/Spring skills",
            "resume_text": """
                EDUCATION:
                B.Tech Computer Science
                
                SKILLS:
                Java, Spring Boot, Hibernate, MySQL, REST API, Maven
                
                PROJECTS:
                E-commerce application using Spring Boot
            """,
            "primary_skills": "Java, Spring Boot, Hibernate, MySQL, REST API, Maven",
            "expected": "Software Engineer"
        },
        {
            "name": "Fresher with Python/ML skills",
            "resume_text": """
                EDUCATION:
                M.Tech Data Science
                
                SKILLS:
                Python, Machine Learning, TensorFlow, Pandas, NumPy, Scikit-learn
                
                PROJECTS:
                Image classification using CNN
            """,
            "primary_skills": "Python, Machine Learning, TensorFlow, Pandas, NumPy, Scikit-learn",
            "expected": "Software Engineer"  # Data Scientist maps to Software Engineer
        },
        {
            "name": "Fresher with Frontend skills",
            "resume_text": """
                EDUCATION:
                B.Tech Information Technology
                
                SKILLS:
                React, JavaScript, HTML, CSS, TypeScript, Redux
                
                PROJECTS:
                Todo app using React
            """,
            "primary_skills": "React, JavaScript, HTML, CSS, TypeScript, Redux",
            "expected": "Software Engineer"
        },
        {
            "name": "Fresher with Database skills",
            "resume_text": """
                EDUCATION:
                B.Tech Computer Science
                
                SKILLS:
                SQL, MySQL, PostgreSQL, Oracle, Database Design, ETL
                
                PROJECTS:
                Database optimization project
            """,
            "primary_skills": "SQL, MySQL, PostgreSQL, Oracle, Database Design, ETL",
            "expected": "Database Administrator"
        },
        {
            "name": "Fresher with Full Stack skills",
            "resume_text": """
                EDUCATION:
                B.Tech Computer Science
                
                SKILLS:
                Python, Django, React, JavaScript, PostgreSQL, REST API
                
                PROJECTS:
                Full stack web application
            """,
            "primary_skills": "Python, Django, React, JavaScript, PostgreSQL, REST API",
            "expected": "Software Engineer"
        },
        {
            "name": "Fresher with Data Analyst skills",
            "resume_text": """
                EDUCATION:
                B.Tech Statistics
                
                SKILLS:
                Python, SQL, Excel, Tableau, Power BI, Data Analysis
                
                PROJECTS:
                Sales data analysis dashboard
            """,
            "primary_skills": "Python, SQL, Excel, Tableau, Power BI, Data Analysis",
            "expected": "Data Analyst"
        },
        {
            "name": "Fresher with no IT skills",
            "resume_text": """
                EDUCATION:
                B.Com
                
                SKILLS:
                Communication, Leadership, Teamwork
            """,
            "primary_skills": None,
            "expected": "Others"
        },
    ]
    
    print("Test Cases:")
    print("-" * 70)
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Skills: {test_case.get('primary_skills', 'None')}")
        
        try:
            result = normalize_role(
                original_role=None,  # No designation (fresher)
                resume_text=test_case.get('resume_text'),
                primary_skills=test_case.get('primary_skills'),
                config=DB_CONFIG
            )
            
            expected = test_case['expected']
            status = "✓ PASS" if result == expected else "✗ FAIL"
            
            if result == expected:
                passed += 1
            else:
                failed += 1
            
            print(f"   Result: {result}")
            print(f"   Expected: {expected}")
            print(f"   Status: {status}")
            
            if result != expected:
                print(f"   ⚠ Mismatch!")
        
        except Exception as e:
            failed += 1
            print(f"   ✗ ERROR: {e}")
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 70)
    
    # Test with existing designation (should still work)
    print("\n" + "=" * 70)
    print("Testing with Existing Designation (should still work)")
    print("=" * 70)
    print()
    
    designation_tests = [
        ("Java Developer", "Software Engineer"),
        ("Senior Software Engineer", "Software Engineer"),
        ("Database Administrator", "Database Administrator"),
    ]
    
    for designation, expected in designation_tests:
        result = normalize_role(original_role=designation, config=DB_CONFIG)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{designation}' -> '{result}' (expected: '{expected}')")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(test_skill_based_detection())

