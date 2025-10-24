#!/usr/bin/env python3
"""
Test the skill extraction function locally
"""

from resume_parser import extract_skills_from_text, extract_experience_from_text

def test_skill_extraction():
    """Test skill extraction with the Python Developer job description"""
    
    python_dev_jd = """About the job
Purpose of the role: 

As Python Developer, the person will be responsible for designing, developing, and maintaining high-performance, scalable applications and services using Python programming language. The role requires expertise in Python and its ecosystem to deliver efficient, reliable, and maintainable software solutions. 



KEY RESPONSIBILITIES: In this role, you will be responsible for: 

Designing and implementing backend services and applications using Python
Building high-performance, concurrent, and scalable systems.
Building the server-side logic of web applications, including APIs and database interactions.
Connecting applications with other services, APIs, and databases. 
Ensuring code quality through testing, debugging, and troubleshooting. 


Experience: 

Graduate or postgraduate in Computer Science Engineering Specialization. 
4-7 years hands-on experience in software development with significant focus on Python


Skills & Competencies: 

Must Have: 

Strong proficiency in Python programming language and its standard library
Experience with web frameworks like Django or Flask.
Knowledge of databases (SQL and NoSQL).
Understanding of software development methodologies (e.g., Agile).
Experience with version control systems (e.g., Git).
Familiarity with cloud platforms (e.g., AWS, Azure, GCP) is often preferred.
Excellent problem-solving and communication skills."""

    print("Testing skill extraction locally...")
    print("=" * 60)
    
    # Test skill extraction
    extracted_skills = extract_skills_from_text(python_dev_jd)
    print(f"Extracted Skills: {extracted_skills}")
    print(f"Number of skills: {len(extracted_skills)}")
    
    # Test experience extraction
    extracted_experience = extract_experience_from_text(python_dev_jd)
    print(f"Extracted Experience: {extracted_experience} years")
    
    print("\n" + "=" * 60)
    print("Testing with minimal JD...")
    
    minimal_jd = "We are looking for a Python developer with 3-5 years experience in Django and Flask."
    
    minimal_skills = extract_skills_from_text(minimal_jd)
    print(f"Minimal JD Skills: {minimal_skills}")
    
    minimal_exp = extract_experience_from_text(minimal_jd)
    print(f"Minimal JD Experience: {minimal_exp} years")

if __name__ == "__main__":
    test_skill_extraction()
