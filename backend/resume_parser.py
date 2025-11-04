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

1. full_name – Identify the candidate's ACTUAL PERSONAL NAME (e.g., "Daniel Mindlin", "John M. Smith", "VARRE DHANA LAKSHMI DURGA").

   RULES FOR EXTRACTION:
   - The name is almost always at the very top of the resume — usually line 1 or line 2
   - The name line typically contains only 2–3 words in Title Case (e.g., Daniel Mindlin, John M. Smith, VARRE DHANA LAKSHMI DURGA)
   - The name appears before any contact information (address, phone, email) or section headers (Education, Experience, etc.)
   - The name should not contain punctuation like ., ,, /, or –
   - The name should not include job titles, degrees, or descriptive text

   STEP-BY-STEP STRATEGY:
   - Check Line 1: If it contains 2–4 words, all alphabetic, and written in Title Case → this is likely the name
   - If Line 1 fails, check Line 2 or Line 3 using the same rule
   - Stop once a valid name is found — do not scan further down the document

   WHAT TO REJECT (Never Treat as a Name):
   - Section headers: Education, Experience, Skills, Contact, Objective, Summary
   - Degrees: B.S. in Computer Science, B.Tech in EEE, M.S., Ph.D.
   - Job titles or roles: Software Engineer, Project Manager, Data Analyst
   - Organization names: Google LLC, Infosys Limited, University of California
   - Lines with punctuation (commas, periods, dashes)
   - Lines that include numeric characters, email addresses, or locations

   EXAMPLES:
   ✅ Correct Extraction:
   Line 1: Daniel Mindlin → ✔ Full Name

   ❌ Incorrect Extractions:
   Line 2: 6056 Sunnycrest Drive – Oak Park, California → ✗ Address
   Line 3: 1-(818)-665-8871 → ✗ Contact Info
   Line 4: Education → ✗ Section Header
   Line 5: B.S. in Statistical Data Science → ✗ Degree

   EXTRACTION RULES:
   - Check first line first (most common location for name)
   - Look for 2–4 words in Title Case or ALL CAPS
   - Must be alphabetic only (hyphens and periods allowed for middle initials)
   - Stop searching after first 3 lines
   - Reject anything with punctuation, numbers, or non-name patterns

2. email – Extract the correct and primary email ID. Ensure this field is NEVER missing if present in resume.

3. phone_number – Include complete phone number if found.

4. total_experience – Identify total professional experience in full years from any available source: prioritize explicit mentions (e.g., "3+ Years of experience", "Over 2 years in IT", "Nearly 5 years"), otherwise calculate from job timelines (start–end dates) in the Experience section, merging overlapping periods, treating "Present/Till Date" as the current date, and ignoring months or duplicate counts.

5. current_company – Capture the current/most recent employer's name.

6. current_designation – Extract the most recent role or job title.

7. technical_skills – Identify ALL technical skills (programming languages, tools, frameworks, cloud platforms, databases, etc.) listed ANYWHERE in the resume. Include EVERY skill mentioned.

8. secondary_skills – Capture complementary or soft skills (leadership, management, communication, mentoring, etc.).

9. all_skills – Combine technical and secondary skills to form complete skill set.

10. domain – Extract only the professional or business domains relevant to the candidate's actual work experience, client industries, or projects.

Return a JSON array of standardized domain names (e.g., ["Banking", "Finance", "Healthcare"]).

Use only from this list:

["Information Technology", "Software Development", "Banking", "Finance", "Insurance", "Healthcare", "Manufacturing", "Retail", "Telecommunications", "Education Technology", "E-commerce", "Logistics", "Real Estate", "Automotive", "Energy", "Construction", "Public Sector"]

Rules:

1. If the resume mentions any technical or programming skills such as Python, Java, SQL, C++, .NET, JavaScript, HTML, CSS, or cloud platforms (AWS, Azure, GCP), automatically include "Information Technology" in the output.

2. If multiple business domains are relevant (e.g., Banking + Finance), include all in the array.

3. CRITICAL - Ignore all mentions of education, study, courses, degrees (like B.Tech, MCA, MBA, B.S., M.S., Ph.D.), and do NOT return "Education Technology" or "Education" as a domain unless it specifically refers to EdTech work (software for learning platforms, LMS, student information systems, educational software development). Academic degrees and university education are NOT business domains - they are qualifications, not industry work.

4. Ignore generic terms like "project", "training", "learning" (unless part of EdTech work).

5. Output only the JSON array. Do not add explanations or extra text.

11. education_details – Include all degrees with full names and specializations (e.g., "MCA - Master of Computer Applications", "B.Tech in Computer Science").

12. certifications – Capture all professional or vendor-specific certifications if mentioned.

13. summary – Provide a concise 2-3 line professional summary describing overall experience, domain focus, and key strengths.

QUALITY & VALIDATION RULES FOR NAME EXTRACTION:
- The name MUST reflect the actual candidate's personal name
- The name field should NEVER contain:
  * Section headers: "Education", "Experience", "Skills", "Contact", "Objective", "Summary"
  * Academic degrees: "B.S. in Computer Science", "B.Tech in EEE", "M.S.", "Ph.D."
  * Job titles or roles: "Software Engineer", "Project Manager", "Data Analyst"
  * Organization names: "Google LLC", "Infosys Limited", "University of California"
  * Lines with punctuation (commas, periods, dashes)
  * Lines that include numeric characters, email addresses, or locations
- The name is almost always at the very top of the resume — usually line 1 or line 2
- The name line typically contains only 2–3 words in Title Case (e.g., Daniel Mindlin, John M. Smith, VARRE DHANA LAKSHMI DURGA)
- Names can be in Title Case (e.g., "John Smith") or ALL CAPS (e.g., "VARRE DHANA LAKSHMI DURGA")
- Names contain only alphabetic characters, hyphens, or periods (for middle initials like "John M. Smith")
- Check Line 1 first: If it contains 2–4 words, all alphabetic, and written in Title Case → this is likely the name
- If Line 1 fails, check Line 2 or Line 3 using the same rule
- Stop once a valid name is found — do not scan further down the document

QUALITY & VALIDATION RULES FOR ALL FIELDS:
- Email must always be fetched if present in resume
- Experience must be logically derived from career history
- Skills extraction must be from the Skills/Tech Skills section only - DO NOT extract from responsibilities
- Domain classification should be comprehensive (multi-domain where applicable)
- Education details must not be omitted
- If data is not available, return field as null (do not guess)
- Output strictly valid JSON ready for database insertion

CRITICAL: To extract the name correctly:
1. Check Line 1 FIRST - if it contains 2-4 words in Title Case or ALL CAPS (alphabetic only, no punctuation or numbers), that IS the name
2. If Line 1 doesn't match, check Line 2 using the same rule
3. If Line 2 doesn't match, check Line 3 using the same rule
4. STOP after checking first 3 lines - do not scan further
5. REJECT section headers, degrees, job titles, organization names, and lines with punctuation/numbers

OUTPUT FORMAT (return valid JSON only):
{
  "full_name": "Candidate's Actual Name from first line",
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

IMPORTANT: 
- Return ONLY valid JSON (no markdown code blocks, no explanatory text)
- Check Line 1 FIRST - if it contains 2-4 words in Title Case or ALL CAPS, that IS the name
- Check only first 3 lines for the name - STOP after that
- Reject section headers, degrees, job titles, organization names, and lines with punctuation/numbers
- Ensure the JSON is complete and valid

Resume Text (look for name in FIRST FEW LINES):
{resume_text}
"""
    
    # Comprehensive technical skills database - ONLY these should appear in primary_skills
    TECHNICAL_SKILLS = {
        # === Programming Languages ===
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
        'swift', 'kotlin', 'scala', 'r', 'perl', 'bash', 'shell scripting', 'objective-c', 'dart',
        'lua', 'matlab', 'assembly', 'fortran', 'sas', 'haskell', 'clojure', 'visual basic', 'vb.net', 'abap',
        
        # === Frameworks / Libraries ===
        'django', 'flask', 'spring', 'react', 'angular', 'vue', 'node.js', 'fastapi', 'express', 'express.js',
        'next.js', 'nestjs', 'laravel', 'symfony', 'flutter', 'react native', 'svelte', 'pytorch', 'tensorflow',
        'struts', 'play framework', 'koa', 'meteor', 'ember.js', 'backbone.js', 'codeigniter', 'cakephp', 'yii',
        'nuxt.js', 'gatsby', 'blazor', 'qt',
        
        # === .NET Framework ===
        '.net', '.net core', '.net framework', 'asp.net', 'asp.net mvc', 'asp.net core', 'ado.net', 'entity framework', 'linq',
        
        # === Databases / Data Tools ===
        'sql', 'sql server', 'mysql', 'postgresql', 'mongodb', 'redis', 'nosql', 'oracle', 'sqlite', 'elasticsearch', 'snowflake',
        'firebase', 'dynamodb', 'cassandra', 'neo4j', 'bigquery', 'redshift', 'clickhouse', 'couchdb', 'hbase',
        'influxdb', 'memcached', 'realm', 'timescaledb', 'duckdb', 'cosmos db',
        
        # === Cloud / DevOps ===
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible', 'prometheus', 'grafana',
        'circleci', 'github actions', 'bitbucket pipelines', 'openstack', 'cloudformation', 'helm', 'istio',
        'argo cd', 'vault', 'consul', 'packer', 'airflow', 'data pipeline', 'mlops', 'cloud run', 'lambda',
        'ecs', 'eks', 'cloudwatch',
        
        # === Security / Authentication ===
        'oauth', 'oauth2', 'jwt', 'ssl', 'tls', 'saml', 'openid connect', 'mfa', 'iam', 'cybersecurity',
        'network security', 'firewall', 'penetration testing', 'encryption', 'hashing',
        
        # === AI / ML / Data Science ===
        'machine learning', 'ai', 'data science', 'analytics', 'nlp', 'computer vision', 'deep learning', 'pandas',
        'numpy', 'scikit-learn', 'matplotlib', 'huggingface', 'openai api', 'llm', 'generative ai', 'langchain',
        'autogen', 'rasa', 'spacy', 'transformers', 'text classification', 'sentiment analysis', 'data visualization',
        'tableau', 'power bi', 'big data', 'hadoop', 'spark', 'pyspark', 'databricks',
        
        # === APIs, Architecture, Monitoring ===
        'rest api', 'graphql', 'graphql api', 'restful api', 'restful services', 'soap', 'rpc', 'grpc', 'openapi',
        'swagger', 'swagger ui', 'api testing', 'load testing', 'jmeter', 'new relic', 'datadog', 'sentry',
        'application monitoring', 'performance tuning', 'microservices', 'websockets', 'api gateway',
        'message queues', 'rabbitmq', 'kafka', 'celery', 'redis streams', 'event-driven architecture',
        'service mesh', 'load balancer',
        
        # === CI/CD & Testing ===
        'git', 'github', 'gitlab', 'agile', 'scrum', 'devops', 'pytest', 'jest',
        'mocha', 'cypress', 'postman', 'swagger', 'jira', 'confluence', 'maven', 'gradle', 'ant', 'sonarqube',
        'selenium', 'playwright', 'testng', 'junit', 'mockito', 'karma', 'chai', 'enzyme',
        
        # === Frontend / UI / UX ===
        'html', 'html5', 'css', 'css3', 'bootstrap', 'jquery', 'tailwind', 'chakra ui', 'material ui', 'redux', 'zustand',
        'framer motion', 'figma', 'ux design', 'responsive design', 'pwa', 'webpack', 'vite', 'babel',
        
        # === Mobile / Cross-Platform ===
        'android', 'ios', 'xcode', 'swiftui', 'jetpack compose', 'ionic', 'capacitor', 'cordova',
        'unity', 'unreal engine',
        
        # === ERP / CRM / Low-Code ===
        'sap', 'sap abap', 'sap hana', 'salesforce', 'salesforce apex', 'salesforce lightning',
        'power apps', 'power automate', 'microsoft dynamics 365', 'business central', 'navision',
        
        # === Other / Emerging Tech ===
        'blockchain', 'solidity', 'smart contracts', 'web3', 'nft', 'metaverse', 'edge computing',
        'quantum computing', 'robotics', 'iot', 'raspberry pi', 'arduino', 'automation',
        
        # === IDE / Development Tools ===
        'visual studio', 'visual studio code', 'eclipse', 'intellij idea', 'netbeans', 'xcode', 'android studio'
    }
    
    DOMAINS = {
        'finance', 'banking', 'fintech', 'healthcare', 'insurance', 'retail', 'e-commerce',
        'telecom', 'manufacturing', 'logistics', 'real estate', 'travel',
        'energy', 'automotive', 'media', 'entertainment', 'consulting', 'saas', 'b2b', 'b2c',
        # Note: 'education' removed - only 'education technology' or 'edtech' should match (via AI prompt)
        # Generic education/degrees are qualifications, not business domains
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
        """Extract candidate name from resume text with PDF header handling."""
        # Common section headers to exclude (but NOT in first 3 lines - those might be the actual name!)
        invalid_names = {'education', 'experience', 'skills', 'contact', 'objective', 
                        'summary', 'qualifications', 'work history', 'professional summary',
                        'references', 'certifications', 'projects', 'achievements'}
        
        # Academic degree patterns
        degree_keywords = ['b.a.', 'm.a.', 'b.s.', 'm.s.', 'phd', 'mba', 'b.tech', 'm.tech', 'degree', 
                          'in ', 'major', 'minor', 'diploma', 'certificate']
        
        # CRITICAL: PDF header area is usually the first 5 lines
        # Prioritize the TOP 2-5 lines as per the prompt guidelines
        lines = text.split('\n')
        
        logger.info("Checking PDF header area (top 5 lines) for candidate name...")
        
        # First, check top 5 lines thoroughly for the name (PDF header area)
        for idx, line in enumerate(lines[:5]):
            # Strip and normalize whitespace
            line = ' '.join(line.split())  # Normalize multiple spaces to single space
            line = line.strip()
            
            # Remove trailing separators like | or • that might appear after names
            line = line.rstrip('|•').strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip section headers (but check if NOT in first 3 lines where actual name might be)
            if idx > 2 and line.lower() in invalid_names:
                continue
            
            # Skip lines with academic degree patterns
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in degree_keywords):
                continue
            
            # Skip lines with degree abbreviations (B.A., M.S., etc.)
            if re.search(r'\b([BM]\.?[AS]\.?|MBA|PhD|MD|JD|B\.?Tech|M\.?Tech)\b', line, re.IGNORECASE):
                continue
            
            # Skip email addresses (they shouldn't be names)
            if '@' in line:
                continue
            
            # Skip phone numbers
            if re.search(r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', line):
                continue
            
            # Skip addresses (they often contain numbers and "Drive", "Street", etc.)
            if any(addr_word in line_lower for addr_word in ['drive', 'street', 'avenue', 'road', 'blvd', 'city']):
                continue
            
            # For first 3 lines only: be more lenient - might be the actual name
            if idx < 3:
                # Reject sentence fragments (ending with comma, period, or containing common phrases)
                if line.endswith(',') or line.endswith('.'):
                    continue
                # Reject common bullet point phrases
                if any(phrase in line_lower for phrase in ['while maintaining', 'while working', 'while attending', 
                                                           'while completing', 'full course', 'as part of', 
                                                           'in order to', 'for the', 'that']):
                    continue
                
                # Accept if it looks like a name (2-4 words, Title Case or ALL CAPS)
                # CRITICAL: Allow ALL CAPS names in PDF headers (e.g., "VARRE DHANA LAKSHMI DURGA")
                if line and 2 <= len(line.split()) <= 4 and len(line) < 70:
                    words = line.split()
                    # Check if mostly alphabetic and NOT a sentence fragment
                    if all(word.replace('.', '').replace(',', '').replace("'", '').replace('-', '').isalpha() for word in words):
                        # Accept even if ALL CAPS (common in PDF headers)
                        logger.info(f"Found candidate name in PDF header area: {line}")
                        return line
            
            # For lines beyond first 3: more strict validation
            # Name is typically 2-4 words, mostly alphabetic, not too long
            if line and 2 <= len(line.split()) <= 4 and len(line) < 50:
                words = line.split()
                # Allow hyphenated names (e.g., "Mary-Jane"), apostrophes, and periods
                if all(word.replace('.', '').replace(',', '').replace("'", '').replace('-', '').isalpha() for word in words):
                    # Additional checks: reject sentence fragments and bullet point content
                    # Reject if ends with comma or period
                    if line.endswith(',') or line.endswith('.'):
                        continue
                    # Reject common bullet point patterns
                    if any(phrase in line_lower for phrase in ['while maintaining', 'while working', 
                                                                'full course', 'as part of', 
                                                                'in order to', 'that', 'the']):
                        continue
                    # Additional check: name should not be in ALL CAPS (likely a section header)
                    # But allow Title Case
                    if not line.isupper():
                        return line
        
        # Fallback: use NLP if available
        if self.nlp:
            doc = self.nlp(text[:1000])  # Increased from 500 to 1000 for better coverage
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    # Validate NLP result - check for degrees and invalid names
                    ent_text_lower = ent.text.lower()
                    if ent_text_lower not in invalid_names:
                        # Check for degree patterns
                        if not any(keyword in ent_text_lower for keyword in degree_keywords):
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
                max_tokens=4000,   # Increased to ensure complete JSON response
                response_format={"type": "json_object"}  # Ensure JSON output
            )
            
            logger.info(f"AI response received, length: {len(response.choices[0].message.content)} chars")
            
            # Parse JSON response with better error handling
            response_content = response.choices[0].message.content.strip()
            
            # Try to fix common JSON issues
            if not response_content.startswith('{') and not response_content.startswith('['):
                # Try to find JSON object start
                start_idx = response_content.find('{')
                if start_idx != -1:
                    response_content = response_content[start_idx:]
                else:
                    logger.error(f"No JSON object found in AI response. First 200 chars: {response_content[:200]}")
                    return None
            
            # Try to parse JSON
            try:
                ai_result = json.loads(response_content)
            except json.JSONDecodeError as je:
                logger.error(f"JSON decode error at position {je.pos}: {je.msg}")
                logger.error(f"Problematic JSON around error: {response_content[max(0, je.pos-100):je.pos+100]}")
                logger.error(f"Full response (first 500 chars): {response_content[:500]}")
                return None
            
            # Validate extracted name - reject section headers and academic degrees
            full_name = ai_result.get('full_name', '')
            if full_name:
                # Common section headers and academic degrees that should NEVER be names
                invalid_names = ['education', 'experience', 'skills', 'contact', 'objective', 
                               'summary', 'qualifications', 'work history', 'professional summary',
                               'references', 'certifications', 'projects', 'achievements']
                
                # Check for academic degree patterns (B.A., M.S., PhD, etc.)
                degree_patterns = [
                    r'\b([BM]\.?[AS]\.?|MBA|PhD|MD|JD|B\.?Tech|M\.?Tech)\b',
                    r'\bin\s+[A-Z][a-z]+',
                    r'degree|diploma|certificate'
                ]
                
                is_invalid = False
                
                # Check if it's in invalid names list
                if full_name.lower() in invalid_names:
                    is_invalid = True
                
                # Check if it contains academic degree patterns
                if not is_invalid:
                    for pattern in degree_patterns:
                        if re.search(pattern, full_name, re.IGNORECASE):
                            is_invalid = True
                            break
                
                # Check if it looks like an academic degree format (e.g., "B.A. in History")
                if not is_invalid and any(keyword in full_name.lower() for keyword in ['in ', 'degree', 'major', 'minor']):
                    is_invalid = True
                
                # Check if it's a sentence fragment from bullet points (e.g., "full course load.")
                if not is_invalid:
                    sentence_fragment_keywords = ['while maintaining', 'while working', 'while attending', 
                                                  'while completing', 'full course', 'as part of', 
                                                  'in order to', 'for the', 'that', 'load.', 'completed in']
                    if any(phrase in full_name.lower() for phrase in sentence_fragment_keywords):
                        is_invalid = True
                
                # Check if it ends with a period or comma (likely a sentence fragment)
                if not is_invalid and (full_name.endswith('.') or full_name.endswith(',')):
                    is_invalid = True
                
                if is_invalid:
                    logger.warning(f"AI extracted invalid name '{full_name}' (sentence fragment/section header/academic degree). Trying regex fallback...")
                    # Use regex-based extraction as fallback
                    fallback_name = self.extract_name(text)
                    if fallback_name and fallback_name != 'Unknown':
                        ai_result['full_name'] = fallback_name
                        logger.info(f"Replaced with: {fallback_name}")
                    else:
                        # Last resort: try to extract from first PERSON entity in first 500 chars
                        logger.warning(f"Regex fallback also failed. Trying NLP...")
                        if self.nlp:
                            doc = self.nlp(text[:500])
                            for ent in doc.ents:
                                if ent.label_ == "PERSON" and len(ent.text.split()) <= 4:
                                    logger.info(f"Found name via NLP: {ent.text}")
                                    ai_result['full_name'] = ent.text
                                    break
                        if ai_result.get('full_name', '').lower() in ['education', 'experience', 'skills']:
                            ai_result['full_name'] = 'Unknown'
                            logger.warning(f"Final fallback resulted in invalid name, setting to Unknown")
            
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
    
    def extract_skills_section(self, text: str) -> Optional[str]:
        """Extract the Skills section content."""
        # Look for Skills section with various patterns (Primary Search)
        patterns = [
            r'(?i)(?:skill profile|technical skills|skills|core competencies?)[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'(?i)(?:proficiencies?|competencies?)[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'(?i)(?:technical skills|skill set|technical summary|technical expertise|core competencies?|proficiencies|tools & technologies|tools and technologies)[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'(?i)(?:skill set)[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'(?i)(?:technical summary)[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'(?i)(?:technical expertise)[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'(?i)(?:tools & technologies|tools and technologies)[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                skills_text = match.group(1).strip()
                if len(skills_text) > 10:  # Make sure it's substantial
                    logger.info("Found skills section using primary search")
                    return skills_text
        
        # Fallback Search: Look for skill-like lists (comma-separated technology names)
        # Pattern: Short words (2-15 chars) separated by commas, often without headers
        fallback_pattern = r'(?:^|\n)([A-Za-z0-9+#\.\-]{1,25}(?:,?\s+[A-Za-z0-9+#\.\-]{1,25}){3,20})'
        match = re.search(fallback_pattern, text, re.MULTILINE)
        if match:
            skills_text = match.group(1).strip()
            if len(skills_text) > 10:
                # Verify it looks like a skills list (not sentences)
                words = re.split(r'[,;•\n]', skills_text)
                avg_word_length = sum(len(w.strip()) for w in words) / max(len(words), 1)
                if avg_word_length < 15:  # Skills are usually short words
                    logger.info("Found skills section using fallback search")
                    return skills_text
        
        return None
    
    def _extract_skills_from_text_with_word_boundaries(self, resume_text: str, existing_skills: List[str], existing_skills_set: set, max_skills: int = 15) -> List[str]:
        """Extract skills from resume text using word-boundary matching with TECHNICAL_SKILLS."""
        logger.info(f"Running word-boundary matching on entire resume... (currently have {len(existing_skills)} skills)")
        resume_text_lower = resume_text.lower()
        
        # Use case-insensitive whole-word matching for each skill in TECHNICAL_SKILLS
        for skill in sorted(self.TECHNICAL_SKILLS, key=len, reverse=True):  # Check longer skills first
            skill_lower = skill.lower()
            # Match whole words only (case-insensitive) using word boundaries
            pattern = r'\b' + re.escape(skill_lower) + r'\b'
            
            # Also handle compound words (e.g., "MicrosoftSqlServer" should match "sql server")
            # Create a pattern without word boundaries to match compound words
            skill_words = skill_lower.split()
            if len(skill_words) > 1:
                # For multi-word skills, check if they appear together without spaces
                # e.g., "sql server" -> "sqlserver" or "sql-server" or "sql_server"
                pattern_compound = r'\b' + r'[\s\-_]?'.join(re.escape(w) for w in skill_words) + r'\b'
            else:
                pattern_compound = pattern
            
            if re.search(pattern, resume_text_lower) or re.search(pattern_compound, resume_text_lower):
                if skill_lower not in existing_skills_set:
                    existing_skills.append(skill)
                    existing_skills_set.add(skill_lower)
                    logger.info(f"Added skill via word-boundary matching: {skill}")
                    if len(existing_skills) >= max_skills:  # Stop after finding max skills
                        break
        
        return existing_skills
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract technical and soft skills."""
        text_lower = text.lower()
        
        found_skills = set()
        
        # CRITICAL: Only look for skills in the Skills section
        skills_section = self.extract_skills_section(text)
        
        if skills_section:
            # Only extract skills that appear in the Skills section
            skills_section_lower = skills_section.lower()
            
            # Extract technical skills that are in TECHNICAL_SKILLS AND in the Skills section
            for skill in self.TECHNICAL_SKILLS:
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, skills_section_lower, re.IGNORECASE):
                    found_skills.add(skill)
            
            # Extract all potential skills from the Skills section
            potential_skills = re.split(r'[,;•\n•]', skills_section)
            for skill in potential_skills:
                skill = skill.strip()
                if skill and len(skill) < 50:
                    skill_lower = skill.lower()
                    # Only keep if it's in TECHNICAL_SKILLS
                    if skill_lower in self.TECHNICAL_SKILLS:
                        found_skills.add(skill_lower)
        else:
            # Fallback: extract from entire text if no Skills section found
            for skill in self.TECHNICAL_SKILLS:
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower, re.IGNORECASE):
                    found_skills.add(skill)
        
        # Categorize as primary/secondary (simple heuristic)
        all_skills = list(found_skills)
        primary_count = min(10, len(all_skills) // 2)
        
        return {
            'primary_skills': all_skills[:primary_count] if all_skills else [],
            'secondary_skills': all_skills[primary_count:] if len(all_skills) > primary_count else [],
            'all_skills': all_skills
        }
    
    def extract_experience(self, text: str) -> float:
        """Calculate total professional experience (full years): prioritize explicit mentions,
        otherwise compute from job timelines.

        - First tries explicit mentions like "3+ Years of experience", "Over 2 years", "Nearly 5 years"
        - Falls back to parsing start-end dates, merging overlaps, treating Present/Till Date as today
        - Floors to integer years
        """
        # Priority 1: Look for explicit experience mentions
        explicit_patterns = [
            r'(?i)(\d+)\s*[+]\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)',
            r'(?i)(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)',
            r'(?i)(?:around|over|nearly|almost|about)\s+(\d+)\+?\s*(?:years?|yrs?)',
            r'(?i)experience[:\s]+(\d+)\+?\s*(?:years?|yrs?)',
            r'(?i)total\s+experience[:\s]+(\d+)\+?\s*(?:years?|yrs?)',
            r'(?i)(?:over|nearly|almost|about)\s+(\d+)\+?\s*(?:years?|yrs?)',
            r'(?i)(\d+)\+?\s*(?:years?|yrs?)(?:\s+in)?\s+(?:the|it|software|technology|industry)'
        ]
        
        for pattern in explicit_patterns:
            matches = re.findall(pattern, text)
            if matches:
                try:
                    value = float(matches[0])
                    if 0 <= value <= 50:  # Reasonable bounds
                        logger.info(f"Found explicit experience mention: {value} years")
                        return value
                except (ValueError, TypeError):
                    pass
        
        # Priority 2: Calculate from timeline-based job dates
        logger.info("No explicit mention found, calculating from job timelines...")
        return self._calculate_experience_from_dates(text)
    
    def _calculate_experience_from_dates(self, text: str) -> float:
        """Calculate experience from date ranges in work history.
        Handles "Present/Current/Till Date" as today and merges overlapping periods.
        """
        # Find Experience section if possible
        experience_text = text
        try:
            section_match = re.search(r'(?is)(experience|work experience|professional experience)[\s\n\r:.-]+(.+?)(?=\n\s*[A-Z][A-Za-z ]{2,}:|\n\s*(education|skills|projects)\b|\Z)', text)
            if section_match and section_match.group(2):
                experience_text = section_match.group(2)
        except Exception:
            pass
        
        # Look for date patterns like "Jan 2020 - Present" or "2018 - 2020"
        month_regex = r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*'
        year_regex = r'(?:19|20)\d{2}'
        date_pattern = rf'(?i){month_regex}\s+{year_regex}|{year_regex}'
        dates = re.findall(date_pattern, experience_text)
        
        # Check for Present/Current/Till Date and add current date if found
        has_present = bool(re.search(r'(?i)(present|current|till date)', experience_text))
        if has_present:
            current_date_str = datetime.now().strftime("%b %Y")
            dates.append(current_date_str.lower())
        
        if len(dates) >= 1:
            try:
                # Extract years
                years = []
                for date_str in dates:
                    year_match = re.search(r'\d{4}', date_str)
                    if year_match:
                        years.append(int(year_match.group()))
                
                if years and len(years) >= 2:
                    # Calculate span from earliest start to latest end
                    current_year = datetime.now().year
                    max_year = min(max(years), current_year)
                    min_year = min(years)
                    return max(0, max_year - min_year)
                elif years and len(years) == 1 and has_present:
                    # Single year + Present means from that year to now
                    return max(0, current_year - years[0])
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
            print(resume_text)
            logger.info(f"----------=======: {resume_text}")
            
            if not resume_text or len(resume_text) < 100:
                raise ValueError("Resume text is too short or empty")
            
            # Try comprehensive AI extraction first
            ai_data = None
            if self.use_ai_extraction:
                ai_data = self.extract_comprehensive_data_with_ai(resume_text)
            
            # Use AI data if available, otherwise fallback to regex-based extraction
            if ai_data:
                # Use AI-extracted comprehensive data
                # First try AI extracted name
                name = ai_data.get('full_name') or ''
                
                # If name is invalid or missing, extract from text
                invalid_names = {'education', 'experience', 'skills', 'contact', 'objective'}
                degree_keywords = ['b.a.', 'm.a.', 'b.s.', 'm.s.', 'phd', 'mba', 'degree']
                
                if not name or name.lower() in ['unknown', 'education', 'experience']:
                    logger.warning(f"AI name extraction failed or returned invalid: '{name}', trying regex fallback...")
                    # Try regex extraction
                    name = self.extract_name(resume_text)
                    logger.info(f"Regex fallback returned: {name}")
                    
                    # If still not found, try a simple heuristic: first line that looks like a name
                    if not name or name == 'Unknown':
                        logger.warning(f"Regex also failed, trying heuristic approach...")
                        lines = resume_text.split('\n')
                        # Check first 10 lines thoroughly
                        for idx, line in enumerate(lines[:10]):
                            line = line.strip()
                            
                            # Remove trailing separators like | or •
                            line = line.rstrip('|•').strip()
                            
                            # Look for lines that are Title Case, 2-4 words, no special chars
                            # Allow up to 4 words for names like "VARRE DHANA LAKSHMI DURGA"
                            if line and 2 <= len(line.split()) <= 4 and len(line) < 70 and len(line) > 3:
                                # Check if it's all alphabetic (plus spaces, hyphens, periods)
                                words = line.split()
                                if all(word.replace('-', '').replace("'", '').replace('.', '').isalpha() for word in words):
                                    # Skip if it's a section header
                                    line_lower = line.lower()
                                    if line_lower not in invalid_names and not any(keyword in line_lower for keyword in degree_keywords):
                                        # Skip if it's an email
                                        if '@' not in line and '://' not in line:
                                            # Skip if it contains phone number patterns
                                            if not re.search(r'\+\d|\d{3}[-.]?\d{3}', line):
                                                name = line
                                                logger.info(f"Found name via heuristic (line {idx+1}): {name}")
                                                break
                email = ai_data.get('email') or self.extract_email(resume_text)
                phone = ai_data.get('phone_number') or self.extract_phone(resume_text)
                # Always use custom extraction to prioritize explicit mentions and properly merge timelines
                experience = self.extract_experience(resume_text)
                
                # Get skills from AI
                ai_technical_skills = ai_data.get('technical_skills', [])
                ai_secondary_skills = ai_data.get('secondary_skills', [])
                all_skills_list = ai_data.get('all_skills', [])
                
                logger.info(f"AI extracted {len(ai_technical_skills)} technical skills")
                
                # Ensure we have lists
                if isinstance(ai_technical_skills, str):
                    ai_technical_skills = [s.strip() for s in ai_technical_skills.split(',') if s.strip()] if ai_technical_skills else []
                if isinstance(ai_secondary_skills, str):
                    ai_secondary_skills = [s.strip() for s in ai_secondary_skills.split(',') if s.strip()] if ai_secondary_skills else []
                
                # Collect all valid technical skills
                technical_skills = []
                technical_skills_lower = set()
                
                # First, process AI-extracted skills
                for skill in ai_technical_skills:
                    if not skill or not isinstance(skill, str):
                        continue
                    skill_lower = skill.lower().strip()
                    
                    # Check if exact match in TECHNICAL_SKILLS
                    if skill_lower in self.TECHNICAL_SKILLS:
                        if skill_lower not in technical_skills_lower:
                            technical_skills.append(skill.strip())
                            technical_skills_lower.add(skill_lower)
                            logger.info(f"✓ Added AI skill: {skill}")
                    else:
                        # Try fuzzy/partial matching
                        matched = False
                        for tech_skill in self.TECHNICAL_SKILLS:
                            if skill_lower in tech_skill or tech_skill in skill_lower:
                                if tech_skill not in technical_skills_lower:
                                    technical_skills.append(tech_skill)
                                    technical_skills_lower.add(tech_skill)
                                    logger.info(f"✓ Added AI skill (fuzzy): {tech_skill} (matched {skill})")
                                    matched = True
                                break
                        if not matched:
                            logger.debug(f"AI skill not matched: {skill}")
                
                # Then, try regex fallback for additional skills
                logger.info(f"Trying regex fallback for additional skills...")
                regex_skills = self.extract_skills(resume_text)
                all_extracted_skills = regex_skills.get('all_skills', [])
                logger.info(f"Regex extracted {len(all_extracted_skills)} potential skills")
                
                for skill in all_extracted_skills:
                    if not skill or not isinstance(skill, str):
                        continue
                    skill_lower = skill.lower().strip()
                    
                    # Only add if not already found
                    if skill_lower in self.TECHNICAL_SKILLS and skill_lower not in technical_skills_lower:
                        technical_skills.append(skill.strip())
                        technical_skills_lower.add(skill_lower)
                        logger.info(f"✓ Added regex skill: {skill}")
                
                # Secondary skills: everything that's NOT in TECHNICAL_SKILLS
                secondary_skills = []
                
                # Process AI secondary skills
                for skill in ai_secondary_skills:
                    if not skill or not isinstance(skill, str):
                        continue
                    skill_lower = skill.lower().strip()
                    if skill_lower not in self.TECHNICAL_SKILLS and skill_lower not in [s.lower() for s in secondary_skills]:
                        secondary_skills.append(skill.strip())
                        logger.info(f"✓ Added secondary skill: {skill}")
                
                # ALWAYS supplement with word-boundary matching to catch any missed skills
                logger.info(f"Supplementing with word-boundary matching from entire resume...")
                technical_skills = self._extract_skills_from_text_with_word_boundaries(
                    resume_text, technical_skills, technical_skills_lower, max_skills=15
                )
                
                # Format skills - primary_skills should ONLY contain TECHNICAL_SKILLS
                primary_skills = ', '.join(technical_skills[:15]) if technical_skills else ''  # Top 15 technical skills
                secondary_skills_str = ', '.join(secondary_skills) if secondary_skills else ''  # Non-technical skills
                
                # all_skills = primary_skills + secondary_skills
                all_skills_combined = technical_skills + secondary_skills
                all_skills_str = ', '.join(all_skills_combined) if all_skills_combined else ''
                
                logger.info(f"✓ Primary skills ({len(technical_skills)}): {primary_skills[:80]}...")
                logger.info(f"✓ Secondary skills ({len(secondary_skills)}): {secondary_skills_str[:80]}...")
                logger.info(f"✓ All skills ({len(all_skills_combined)}): {all_skills_str[:80]}...")
                
                logger.info(f"✓ AI extraction completed: {len(technical_skills)} technical skills")
                
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
                
                # CRITICAL: Filter to ONLY include skills from TECHNICAL_SKILLS list
                all_extracted_skills = skills['all_skills']
                
                # Common responsibility phrases that should be rejected
                responsibility_phrases = [
                    'unit testing', 'integration testing', 'system testing', 'end to end testing',
                    'test driven development', 'tdd', 'bdd', 'behavior driven development',
                    'agile methodology', 'scrum methodology', 'waterfall methodology',
                    'performed unit testing', 'implemented unit testing', 'wrote unit tests'
                ]
                
                # Separate into technical (in our list) and non-technical
                technical_skills_list = []
                secondary_skills_list = []
                technical_skills_set = set()  # For deduplication
                
                for skill in all_extracted_skills:
                    skill_lower = skill.lower().strip()
                    
                    # Reject responsibility-like phrases unless they're explicitly in TECHNICAL_SKILLS
                    if skill_lower in responsibility_phrases and skill_lower not in self.TECHNICAL_SKILLS:
                        logger.warning(f"Rejected responsibility phrase as skill: '{skill}'")
                        continue
                    
                    # Check if this skill is in our TECHNICAL_SKILLS list
                    if skill_lower in self.TECHNICAL_SKILLS:
                        if skill_lower not in technical_skills_set:
                            technical_skills_list.append(skill)
                            technical_skills_set.add(skill_lower)
                    else:
                        # Try partial match
                        found_match = False
                        for tech_skill in self.TECHNICAL_SKILLS:
                            if skill_lower in tech_skill or tech_skill in skill_lower:
                                if tech_skill not in technical_skills_set:
                                    technical_skills_list.append(tech_skill)
                                    technical_skills_set.add(tech_skill)
                                    found_match = True
                                    break
                        if not found_match:
                            secondary_skills_list.append(skill)
                
                # ALWAYS supplement with word-boundary matching to catch any missed skills
                logger.info(f"Supplementing with word-boundary matching from entire resume...")
                technical_skills_list = self._extract_skills_from_text_with_word_boundaries(
                    resume_text, technical_skills_list, technical_skills_set, max_skills=15
                )
                
                # Format primary_skills after potential lenient extraction
                primary_skills = ', '.join(technical_skills_list[:15]) if technical_skills_list else ''
                secondary_skills_str = ', '.join(secondary_skills_list) if secondary_skills_list else ''
                
                # all_skills = primary_skills + secondary_skills (combine lists, then join)
                all_skills_combined = technical_skills_list + secondary_skills_list
                all_skills_str = ', '.join(all_skills_combined) if all_skills_combined else ''
                
                logger.info(f"✓ Primary skills ({len(technical_skills_list)}): {primary_skills[:80]}...")
                logger.info(f"✓ Secondary skills ({len(secondary_skills_list)}): {secondary_skills_str[:80]}...")
                logger.info(f"✓ All skills ({len(all_skills_combined)}): {all_skills_str[:80]}...")
                
                logger.info(f"✓ Regex extraction completed: {len(technical_skills_list)} technical skills")
                
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

