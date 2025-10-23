"""
Resume Parser for ATS System.
Extracts structured information from PDF and DOCX resumes.
"""

import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

# PDF/DOCX parsing libraries
try:
    import PyPDF2
    from docx import Document
except ImportError:
    pass

# NLP libraries
try:
    import spacy
    from spacy.matcher import Matcher
except ImportError:
    pass

logger = logging.getLogger(__name__)


class ResumeParser:
    """Intelligent resume parser with NLP capabilities."""
    
    # Common skills database (expandable)
    TECHNICAL_SKILLS = {
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'go', 'rust',
        'sql', 'nosql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
        'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'fastapi', 'spring',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'ci/cd',
        'machine learning', 'deep learning', 'nlp', 'computer vision', 'tensorflow', 'pytorch',
        'data science', 'data analysis', 'pandas', 'numpy', 'scikit-learn',
        'rest api', 'graphql', 'microservices', 'kafka', 'rabbitmq',
        'agile', 'scrum', 'jira', 'linux', 'bash', 'powershell'
    }
    
    DOMAINS = {
        'finance', 'banking', 'fintech', 'healthcare', 'insurance', 'retail', 'e-commerce',
        'telecom', 'manufacturing', 'logistics', 'education', 'real estate', 'travel',
        'energy', 'automotive', 'media', 'entertainment', 'consulting', 'saas', 'b2b', 'b2c'
    }
    
    EDUCATION_KEYWORDS = {
        'b.tech', 'b.e.', 'bachelor', 'btech', 'bca', 'bsc', 'ba',
        'm.tech', 'm.e.', 'master', 'mtech', 'mca', 'msc', 'mba', 'ma',
        'phd', 'doctorate', 'diploma', 'associate'
    }
    
    def __init__(self, nlp_model: str = 'en_core_web_sm'):
        """Initialize parser with NLP model."""
        self.nlp = None
        self.matcher = None
        
        try:
            self.nlp = spacy.load(nlp_model)
            self.matcher = Matcher(self.nlp.vocab)
            self._setup_patterns()
            logger.info(f"Loaded spaCy model: {nlp_model}")
        except Exception as e:
            logger.warning(f"Could not load spaCy model: {e}. Using regex-based parsing.")
    
    def _setup_patterns(self):
        """Setup spaCy patterns for entity extraction."""
        if not self.matcher:
            return
        
        # Email pattern
        email_pattern = [{"LIKE_EMAIL": True}]
        self.matcher.add("EMAIL", [email_pattern])
        
        # Phone pattern
        phone_pattern = [{"SHAPE": "ddd-ddd-dddd"}]
        self.matcher.add("PHONE", [phone_pattern])
    
    def parse_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            raise ValueError(f"Failed to parse PDF: {str(e)}")
    
    def parse_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except Exception as e:
            logger.error(f"Error parsing DOCX: {e}")
            raise ValueError(f"Failed to parse DOCX: {str(e)}")
    
    def extract_text_from_file(self, file_path: str, file_type: str) -> str:
        """Extract text based on file type."""
        file_type = file_type.lower()
        
        if file_type == 'pdf':
            return self.parse_pdf(file_path)
        elif file_type in ['docx', 'doc']:
            return self.parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def extract_name(self, text: str) -> Optional[str]:
        """Extract candidate name from resume text."""
        # First few lines usually contain name
        lines = text.split('\n')
        for line in lines[:5]:
            line = line.strip()
            # Name is typically 2-4 words, mostly alphabetic
            if line and len(line.split()) <= 4 and len(line) < 50:
                words = line.split()
                if all(word.replace('.', '').replace(',', '').isalpha() for word in words):
                    return line
        
        # Fallback: use NLP if available
        if self.nlp:
            doc = self.nlp(text[:500])
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    return ent.text
        
        return "Unknown"
    
    def extract_email(self, text: str) -> Optional[str]:
        """Extract email address."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(email_pattern, text)
        return matches[0] if matches else None
    
    def extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number."""
        # Various phone formats
        patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\b\d{10}\b',
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]
        return None
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract technical and soft skills."""
        text_lower = text.lower()
        
        found_skills = set()
        
        # Extract technical skills
        for skill in self.TECHNICAL_SKILLS:
            # Look for whole word matches
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower, re.IGNORECASE):
                found_skills.add(skill)
        
        # Extract from Skills section if present
        skills_section_pattern = r'(?i)(?:skills?|technical skills?|core competencies?)[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)'
        skills_match = re.search(skills_section_pattern, text, re.DOTALL)
        
        if skills_match:
            skills_text = skills_match.group(1)
            # Split by common delimiters
            potential_skills = re.split(r'[,;•\n]', skills_text)
            for skill in potential_skills:
                skill = skill.strip()
                if skill and len(skill) < 50:
                    found_skills.add(skill.lower())
        
        # Categorize as primary/secondary (simple heuristic)
        all_skills = list(found_skills)
        primary_count = min(10, len(all_skills) // 2)
        
        return {
            'primary_skills': all_skills[:primary_count] if all_skills else [],
            'secondary_skills': all_skills[primary_count:] if len(all_skills) > primary_count else [],
            'all_skills': all_skills
        }
    
    def extract_experience(self, text: str) -> float:
        """Extract total years of experience."""
        # Look for experience statements
        patterns = [
            r'(?i)(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)',
            r'(?i)experience[:\s]+(\d+)\+?\s*(?:years?|yrs?)',
            r'(?i)total\s+experience[:\s]+(\d+)\+?\s*(?:years?|yrs?)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                try:
                    return float(matches[0])
                except ValueError:
                    pass
        
        # Alternative: Calculate from work history
        experience_years = self._calculate_experience_from_dates(text)
        return experience_years
    
    def _calculate_experience_from_dates(self, text: str) -> float:
        """Calculate experience from date ranges in work history."""
        # Look for date patterns like "Jan 2020 - Present" or "2018 - 2020"
        date_pattern = r'(?i)(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}|(?:\d{4})'
        dates = re.findall(date_pattern, text)
        
        if len(dates) >= 2:
            try:
                # Extract years
                years = []
                for date_str in dates:
                    year_match = re.search(r'\d{4}', date_str)
                    if year_match:
                        years.append(int(year_match.group()))
                
                if years:
                    # Simple estimation: max year - min year
                    current_year = datetime.now().year
                    max_year = min(max(years), current_year)
                    min_year = min(years)
                    return max(0, max_year - min_year)
            except Exception as e:
                logger.warning(f"Error calculating experience from dates: {e}")
        
        return 0.0
    
    def extract_domain(self, text: str) -> Optional[str]:
        """Extract domain/industry."""
        text_lower = text.lower()
        
        found_domains = []
        for domain in self.DOMAINS:
            if domain in text_lower:
                found_domains.append(domain)
        
        # Return most frequent or first found
        if found_domains:
            return found_domains[0]
        
        return None
    
    def extract_education(self, text: str) -> Dict[str, Any]:
        """Extract education information."""
        education_info = {
            'highest_degree': None,
            'education_details': []
        }
        
        text_lower = text.lower()
        
        # Find education section
        edu_section_pattern = r'(?i)(?:education|academic|qualification)[:\s]+(.*?)(?=\n\n[A-Z]|experience|skills|$)'
        edu_match = re.search(edu_section_pattern, text, re.DOTALL)
        
        if edu_match:
            edu_text = edu_match.group(1)
            education_info['education_details'].append(edu_text.strip())
        
        # Extract degree keywords
        degrees_found = []
        for keyword in self.EDUCATION_KEYWORDS:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                degrees_found.append(keyword)
        
        # Determine highest degree (simple heuristic)
        if any(deg in degrees_found for deg in ['phd', 'doctorate']):
            education_info['highest_degree'] = 'PhD'
        elif any(deg in degrees_found for deg in ['m.tech', 'm.e.', 'master', 'mtech', 'mca', 'msc', 'mba', 'ma']):
            education_info['highest_degree'] = 'Masters'
        elif any(deg in degrees_found for deg in ['b.tech', 'b.e.', 'bachelor', 'btech', 'bca', 'bsc', 'ba']):
            education_info['highest_degree'] = 'Bachelors'
        elif 'diploma' in degrees_found:
            education_info['highest_degree'] = 'Diploma'
        
        return education_info
    
    def extract_location(self, text: str) -> Optional[str]:
        """Extract current location."""
        # Look for location patterns
        location_pattern = r'(?i)(?:location|based in|residing in|current location)[:\s]+([A-Za-z\s,]+?)(?:\n|$)'
        matches = re.findall(location_pattern, text)
        
        if matches:
            return matches[0].strip()
        
        # Use NLP to find GPE (Geopolitical Entity)
        if self.nlp:
            doc = self.nlp(text[:1000])
            locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
            if locations:
                return locations[0]
        
        return None
    
    def parse_resume(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """
        Main parsing method that extracts all information from resume.
        
        Args:
            file_path: Path to resume file
            file_type: File extension (pdf, docx)
        
        Returns:
            Dictionary with extracted resume data
        """
        try:
            # Extract text
            resume_text = self.extract_text_from_file(file_path, file_type)
            
            if not resume_text or len(resume_text) < 100:
                raise ValueError("Resume text is too short or empty")
            
            # Extract all information
            name = self.extract_name(resume_text)
            email = self.extract_email(resume_text)
            phone = self.extract_phone(resume_text)
            skills = self.extract_skills(resume_text)
            experience = self.extract_experience(resume_text)
            domain = self.extract_domain(resume_text)
            education_info = self.extract_education(resume_text)
            location = self.extract_location(resume_text)
            
            # Get file info
            file_size_kb = os.path.getsize(file_path) / 1024 if os.path.exists(file_path) else 0
            
            parsed_data = {
                'name': name,
                'email': email,
                'phone': phone,
                'total_experience': experience,
                'primary_skills': ', '.join(skills['primary_skills']),
                'secondary_skills': ', '.join(skills['secondary_skills']),
                'all_skills': ', '.join(skills['all_skills']),
                'domain': domain,
                'education': education_info['highest_degree'],
                'education_details': '\n'.join(education_info['education_details']),
                'current_location': location,
                'resume_text': resume_text,
                'file_name': os.path.basename(file_path),
                'file_type': file_type,
                'file_size_kb': int(file_size_kb)
            }
            
            logger.info(f"Successfully parsed resume for: {name}")
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing resume: {e}")
            raise


def extract_skills_from_text(text: str) -> List[str]:
    """Standalone function to extract skills from any text."""
    parser = ResumeParser()
    skills = parser.extract_skills(text)
    return skills['all_skills']


def extract_experience_from_text(text: str) -> float:
    """Standalone function to extract experience from any text."""
    parser = ResumeParser()
    return parser.extract_experience(text)

