#!/usr/bin/env python3
"""
Test script for AI-enhanced resume parser.
Tests both traditional and AI-powered skill extraction.
"""

import json
import logging
from resume_parser import ResumeParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ai_parser():
    """Test the AI-enhanced resume parser."""
    
    # Sample resume text for testing
    sample_resume = """
    John Doe
    Software Engineer
    john.doe@example.com
    +1-555-123-4567
    
    EXPERIENCE:
    Senior Python Developer at TechCorp (2020-2023)
    - Developed web applications using Python, Django, and PostgreSQL
    - Built REST APIs and microservices architecture
    - Led a team of 3 developers on e-commerce platform
    
    Python Developer at StartupXYZ (2018-2020)
    - Created data analysis tools using pandas and numpy
    - Implemented machine learning models with scikit-learn
    - Worked with React.js for frontend development
    
    EDUCATION:
    Bachelor of Technology in Computer Science
    University of Technology (2014-2018)
    
    SKILLS:
    Python, Django, Flask, PostgreSQL, MySQL, React.js, JavaScript,
    Machine Learning, Data Science, REST APIs, Docker, AWS
    """
    
    print("🧪 Testing AI-Enhanced Resume Parser")
    print("=" * 50)
    
    # Test with AI extraction enabled
    print("\n🤖 Testing with AI extraction:")
    parser_ai = ResumeParser(use_ai_extraction=True)
    
    try:
        # Test AI skill extraction
        ai_skills = parser_ai.extract_skills_with_ai(sample_resume)
        print("✅ AI extraction successful!")
        print(f"Primary skills: {ai_skills['primary_skills']}")
        print(f"Secondary skills: {ai_skills['secondary_skills']}")
        
        if 'ai_analysis' in ai_skills:
            analysis = ai_skills['ai_analysis']
            print(f"AI-detected experience: {analysis.get('total_experience', 'N/A')} years")
            print(f"Project details: {len(analysis.get('project_details', []))} projects found")
        
    except Exception as e:
        print(f"❌ AI extraction failed: {e}")
        print("Falling back to traditional extraction...")
    
    # Test traditional extraction
    print("\n📝 Testing traditional extraction:")
    parser_traditional = ResumeParser(use_ai_extraction=False)
    
    try:
        traditional_skills = parser_traditional.extract_skills(sample_resume)
        print("✅ Traditional extraction successful!")
        print(f"Primary skills: {traditional_skills['primary_skills']}")
        print(f"Secondary skills: {traditional_skills['secondary_skills']}")
        
    except Exception as e:
        print(f"❌ Traditional extraction failed: {e}")
    
    # Test full parsing
    print("\n📄 Testing full resume parsing:")
    try:
        # Create a temporary file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_resume)
            temp_file = f.name
        
        # Parse the resume
        parsed_data = parser_ai.parse_resume(temp_file, 'txt')
        
        print("✅ Full parsing successful!")
        print(f"Name: {parsed_data['name']}")
        print(f"Email: {parsed_data['email']}")
        print(f"Experience: {parsed_data['total_experience']} years")
        print(f"AI extraction used: {parsed_data.get('ai_extraction_used', False)}")
        
        # Clean up
        import os
        os.unlink(temp_file)
        
    except Exception as e:
        print(f"❌ Full parsing failed: {e}")
    
    print("\n🎯 Test completed!")

if __name__ == "__main__":
    test_ai_parser()
