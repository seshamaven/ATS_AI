"""
Resume Parser for ATS System.
Extracts structured information from PDF and DOCX resumes with AI-powered skill analysis.
"""

import re
import json
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

# OpenAI/Azure OpenAI for AI-powered extraction
try:
    from openai import OpenAI, AzureOpenAI
except ImportError:
    pass

logger = logging.getLogger(__name__)


class ResumeParser:
    """Intelligent resume parser with NLP and AI capabilities."""
    
    # AI-powered comprehensive extraction prompt
    AI_COMPREHENSIVE_EXTRACTION_PROMPT = """
You are an expert resume parser trained to extract complete and accurate professional metadata from resumes.
Analyze the provided resume text carefully and return a structured JSON with well-validated information.

EXTRACTION GUIDELINES:

1. full_name – Identify the candidate's ACTUAL PERSONAL NAME (e.g., "Daniel Mindlin", "John Smith"). 
   CRITICAL: Do NOT confuse section headers or labels (like "Education", "Experience", "Skills", "Contact Information") with the person's name.
   The candidate's name is typically at the top of the resume, often centered or on the left.
   It is NEVER a section header. If the first prominent text is a section header, look for the actual name below or above it.

2. email – Extract the correct and primary email ID. Ensure this field is NEVER missing if present in resume.

3. phone_number – Include complete phone number if found.

4. total_experience – Calculate accurately based on career timeline and roles. Cross-check start and end dates to avoid inflated values (e.g., 32 should be 23 if that's the correct calculation).

5. current_company – Capture the current/most recent employer's name.

6. current_designation – Extract the most recent role or job title.

7. technical_skills – Identify ALL technical skills (programming languages, tools, frameworks, cloud platforms, databases, etc.) listed anywhere in the resume. Include EVERY skill mentioned.

8. secondary_skills – Capture complementary or soft skills (leadership, management, communication, mentoring, etc.).

9. all_skills – Combine technical and secondary skills to form complete skill set.

10. domain – Determine ALL relevant domains based on candidate's experience. List multiple domains if applicable (e.g., ["Banking", "Finance", "Insurance"]).

11. education_details – Include all degrees with full names and specializations (e.g., "MCA - Master of Computer Applications", "B.Tech in Computer Science").

12. certifications – Capture all professional or vendor-specific certifications if mentioned.

13. summary – Provide a concise 2-3 line professional summary describing overall experience, domain focus, and key strengths.

QUALITY & VALIDATION RULES:
- Ensure name reflects actual candidate, not organization names, job titles, OR section headers
- The name field should NEVER contain words like "Education", "Experience", "Skills", "Contact", "Objective", "Summary"
- Email must always be fetched if present
- Experience must be logically derived from career history
- Skills extraction must be exhaustive - no key technology should be missed
- Domain classification should be comprehensive (multi-domain where applicable)
- Education details must not be omitted
- If data is not available, return field as null (do not guess)
- Output strictly valid JSON ready for database insertion

OUTPUT FORMAT:
{
  "full_name": "Candidate's Actual Name",
  "email": "email@example.com",
  "phone_number": "phone_number_or_null",
  "total_experience": <numeric_value>,
  "current_company": "Company Name or null",
  "current_designation": "Job Title or null",
  "technical_skills": ["skill1", "skill2", ...],
  "secondary_skills": ["skill1", "skill2", ...],
  "all_skills": ["skill1", "skill2", ...],
  "domain": ["domain1", "domain2", ...],
  "education_details": ["Degree details"],
  "certifications": ["cert1", "cert2"] or null,
  "summary": "Professional summary here"
}

Resume Text:
{resume_text}
"""
    
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
    
    def __init__(self, nlp_model: str = 'en_core_web_sm', use_ai_extraction: bool = True):
        """Initialize parser with NLP model and AI capabilities."""
        self.nlp = None
        self.matcher = None
        self.use_ai_extraction = use_ai_extraction
        self.ai_client = None
        
        try:
            self.nlp = spacy.load(nlp_model)
            self.matcher = Matcher(self.nlp.vocab)
            self._setup_patterns()
            logger.info(f"Loaded spaCy model: {nlp_model}")
        except Exception as e:
            logger.warning(f"Could not load spaCy model: {e}. Using regex-based parsing.")
        
        # Initialize AI client if AI extraction is enabled
        if self.use_ai_extraction:
            self._initialize_ai_client()
    
    def _initialize_ai_client(self):
        """Initialize OpenAI or Azure OpenAI client."""
        try:
            from ats_config import ATSConfig
            
            # Try Azure OpenAI first
            if ATSConfig.AZURE_OPENAI_ENDPOINT and ATSConfig.AZURE_OPENAI_API_KEY:
                self.ai_client = AzureOpenAI(
                    api_key=ATSConfig.AZURE_OPENAI_API_KEY,
                    api_version=ATSConfig.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=ATSConfig.AZURE_OPENAI_ENDPOINT
                )
                self.ai_model = ATSConfig.AZURE_OPENAI_DEPLOYMENT_NAME
                logger.info("Initialized Azure OpenAI client for skill extraction")
            # Fallback to OpenAI
            elif ATSConfig.OPENAI_API_KEY:
                self.ai_client = OpenAI(api_key=ATSConfig.OPENAI_API_KEY)
                self.ai_model = ATSConfig.OPENAI_MODEL
                logger.info("Initialized OpenAI client for skill extraction")
            else:
                logger.warning("No OpenAI API key found. AI extraction disabled.")
                self.use_ai_extraction = False
                
        except Exception as e:
            logger.error(f"Failed to initialize AI client: {e}")
            self.use_ai_extraction = False
    
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
        # Common section headers to exclude
        invalid_names = {'education', 'experience', 'skills', 'contact', 'objective', 
                        'summary', 'qualifications', 'work history', 'professional summary',
                        'references', 'certifications', 'projects', 'achievements'}
        
        # First few lines usually contain name
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines (was 5)
            line = line.strip()
            # Skip empty lines and section headers
            if not line or line.lower() in invalid_names:
                continue
                
            # Name is typically 2-4 words, mostly alphabetic, not too long
            if line and len(line.split()) <= 4 and len(line) < 50:
                words = line.split()
                if all(word.replace('.', '').replace(',', '').replace("'", '').isalpha() for word in words):
                    # Additional check: name should not be in ALL CAPS (likely a section header)
                    if not line.isupper():
                        return line
        
        # Fallback: use NLP if available
        if self.nlp:
            doc = self.nlp(text[:500])
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    # Validate NLP result too
                    if ent.text.lower() not in invalid_names:
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
    
    def extract_comprehensive_data_with_ai(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive resume data using AI-powered analysis."""
        if not self.use_ai_extraction or not self.ai_client:
            logger.warning("AI extraction not available, falling back to regex-based extraction")
            return None
        
        try:
            # Prepare the prompt with resume text (increase limit for comprehensive extraction)
            prompt = self.AI_COMPREHENSIVE_EXTRACTION_PROMPT.format(resume_text=text[:16000])
            
            # Call AI API
            response = self.ai_client.chat.completions.create(
                model=self.ai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=3000,   # Increased for comprehensive response
                response_format={"type": "json_object"}  # Ensure JSON output
            )
            
            # Parse JSON response
            ai_result = json.loads(response.choices[0].message.content)
            
            # Validate extracted name - reject section headers
            full_name = ai_result.get('full_name', '')
            if full_name:
                # Common section headers that should NEVER be names
                invalid_names = ['education', 'experience', 'skills', 'contact', 'objective', 
                               'summary', 'qualifications', 'work history', 'professional summary',
                               'references', 'certifications', 'projects', 'achievements']
                
                if full_name.lower() in invalid_names:
                    logger.warning(f"AI extracted invalid name '{full_name}', likely a section header. Trying regex fallback...")
                    # Use regex-based extraction as fallback
                    ai_result['full_name'] = self.extract_name(text) or 'Unknown'
                    logger.info(f"Replaced with: {ai_result['full_name']}")
            
            logger.info(f"AI extraction completed for {ai_result.get('full_name', 'Unknown')}")
            return ai_result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.debug(f"AI response: {response.choices[0].message.content}")
            return None
            
        except Exception as e:
            logger.error(f"AI comprehensive extraction failed: {e}")
            return None
    
    def extract_skills_with_ai(self, text: str) -> Dict[str, Any]:
        """Legacy method for AI skill extraction - now calls comprehensive extraction."""
        ai_data = self.extract_comprehensive_data_with_ai(text)
        
        if ai_data:
            technical_skills = ai_data.get('technical_skills', [])
            secondary_skills = ai_data.get('secondary_skills', [])
            all_skills_list = ai_data.get('all_skills', [])
            
            return {
                'primary_skills': technical_skills[:15],
                'secondary_skills': secondary_skills + technical_skills[15:],
                'all_skills': all_skills_list,
                'ai_analysis': {
                    'total_experience': ai_data.get('total_experience', 0),
                    'candidate_name': ai_data.get('full_name', ''),
                    'email': ai_data.get('email', ''),
                    'phone': ai_data.get('phone_number', '')
                }
            }
        
        return self.extract_skills(text)
    
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
            
            # Try comprehensive AI extraction first
            ai_data = None
            if self.use_ai_extraction:
                ai_data = self.extract_comprehensive_data_with_ai(resume_text)
            
            # Use AI data if available, otherwise fallback to regex-based extraction
            if ai_data:
                # Use AI-extracted comprehensive data
                name = ai_data.get('full_name') or self.extract_name(resume_text)
                email = ai_data.get('email') or self.extract_email(resume_text)
                phone = ai_data.get('phone_number') or self.extract_phone(resume_text)
                experience = float(ai_data.get('total_experience', 0)) if ai_data.get('total_experience') else self.extract_experience(resume_text)
                
                # Get skills
                technical_skills = ai_data.get('technical_skills', [])
                secondary_skills = ai_data.get('secondary_skills', [])
                all_skills_list = ai_data.get('all_skills', [])
                
                # Format skills
                primary_skills = ', '.join(technical_skills[:15])  # Top 15 technical skills
                secondary_skills_str = ', '.join(secondary_skills) + ', ' + ', '.join(technical_skills[15:]) if len(technical_skills) > 15 else ', '.join(secondary_skills)
                all_skills_str = ', '.join(all_skills_list)
                
                # Get domains (handle both single and multiple)
                domain_list = ai_data.get('domain', [])
                if isinstance(domain_list, list):
                    domain = ', '.join(domain_list)
                else:
                    domain = domain_list or self.extract_domain(resume_text)
                
                # Get education
                education_list = ai_data.get('education_details', [])
                if isinstance(education_list, list):
                    education_details = '\n'.join(education_list)
                    # Extract highest degree
                    highest_degree = education_list[0] if education_list else None
                else:
                    education_details = education_list or ''
                    highest_degree = education_details.split('\n')[0] if education_details else None
                
                # Get certifications
                certifications = ai_data.get('certifications', [])
                certifications_str = ', '.join(certifications) if isinstance(certifications, list) else certifications or ''
                
                # Get current company and designation
                current_company = ai_data.get('current_company') or ''
                current_designation = ai_data.get('current_designation') or ''
                
                # Summary
                summary = ai_data.get('summary') or ''
                
                # Get additional data
                location = self.extract_location(resume_text)
                
            else:
                # Fallback to regex-based extraction
                name = self.extract_name(resume_text)
                email = self.extract_email(resume_text)
                phone = self.extract_phone(resume_text)
                experience = self.extract_experience(resume_text)
                
                skills = self.extract_skills(resume_text)
                primary_skills = ', '.join(skills['primary_skills'])
                secondary_skills_str = ', '.join(skills['secondary_skills'])
                all_skills_str = ', '.join(skills['all_skills'])
                
                domain = self.extract_domain(resume_text)
                education_info = self.extract_education(resume_text)
                highest_degree = education_info['highest_degree']
                education_details = '\n'.join(education_info['education_details'])
                
                # Extract current company and designation using regex
                current_company = self._extract_current_company(resume_text)
                current_designation = self._extract_current_designation(resume_text)
                certifications_str = ''
                summary = ''
                location = self.extract_location(resume_text)
            
            # Get file info
            file_size_kb = os.path.getsize(file_path) / 1024 if os.path.exists(file_path) else 0
            
            # Prepare parsed data
            parsed_data = {
                'name': name,
                'email': email,
                'phone': phone,
                'total_experience': experience,
                'primary_skills': primary_skills,
                'secondary_skills': secondary_skills_str,
                'all_skills': all_skills_str,
                'domain': domain,
                'education': highest_degree,
                'education_details': education_details,
                'current_location': location,
                'current_company': current_company,
                'current_designation': current_designation,
                'certifications': certifications_str,
                'resume_summary': summary,
                'resume_text': resume_text,
                'file_name': os.path.basename(file_path),
                'file_type': file_type,
                'file_size_kb': int(file_size_kb),
                'ai_extraction_used': ai_data is not None
            }
            
            logger.info(f"Successfully parsed resume for: {name}")
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing resume: {e}")
            raise
    
    def _extract_current_company(self, text: str) -> Optional[str]:
        """Extract current/most recent company name."""
        # Look for company in work history (first/last company mentioned)
        patterns = [
            r'(?i)(?:company|employer|organization)[:\s]+([A-Za-z0-9\s&.,]+)',
            r'(?i)(?:worked at|employed at|currently at)[:\s]+([A-Za-z0-9\s&.,]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0].strip()
        
        # Try to find first company in experience section
        exp_section_pattern = r'(?i)(?:experience|work history|employment)(.*?)(?=\n\n[A-Z]|education|skills|$)'
        exp_match = re.search(exp_section_pattern, text, re.DOTALL)
        if exp_match:
            exp_text = exp_match.group(1)
            # Extract first company name
            company_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|LLC|Ltd|Corp|Pvt))?)'
            companies = re.findall(company_pattern, exp_text[:500])
            if companies:
                return companies[0].strip()
        
        return None
    
    def _extract_current_designation(self, text: str) -> Optional[str]:
        """Extract current/most recent job designation."""
        # Look for designation patterns
        patterns = [
            r'(?i)(?:position|role|title|designation)[:\s]+([A-Za-z\s]+)',
            r'(?i)(?:currently|presently).*?(?:as|working as|role of)[:\s]+([A-Za-z\s]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0].strip()
        
        # Extract first role from experience section
        exp_section_pattern = r'(?i)(?:experience|work history)(.*?)(?=\n\n[A-Z]|education|skills|$)'
        exp_match = re.search(exp_section_pattern, text, re.DOTALL)
        if exp_match:
            exp_text = exp_match.group(1)
            # Common role patterns
            role_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Engineer|Manager|Director|Lead|Developer|Architect|Analyst|Consultant|Specialist)))'
            roles = re.findall(role_pattern, exp_text[:300])
            if roles:
                return roles[0].strip()
        
        return None


def extract_skills_from_text(text: str) -> List[str]:
    """Standalone function to extract skills from any text."""
    parser = ResumeParser()
    skills = parser.extract_skills(text)
    return skills['all_skills']


def extract_experience_from_text(text: str) -> float:
    """Standalone function to extract experience from any text."""
    parser = ResumeParser()
    return parser.extract_experience(text)

