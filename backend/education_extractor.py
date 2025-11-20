import re
import json
import logging
from typing import List, Optional, Dict, Any

# AI client imports
try:
    from openai import OpenAI, AzureOpenAI
except ImportError:
    OpenAI = None
    AzureOpenAI = None

# Database imports
try:
    from ats_database import ATSDatabase
    from ats_config import ATSConfig
except ImportError:
    ATSDatabase = None
    ATSConfig = None

logger = logging.getLogger(__name__)


class EducationExtractor:
    """
    Standalone Education Extractor - Extracts only real degrees with full degree names and specializations.
    Ignores trainings, certifications, workshops.
    
    Usage:
        extractor = EducationExtractor(resume_text)
        education_list = extractor.extract()
    """
    
    # Valid education section headings
    EDUCATION_HEADINGS = [
        r'education', r'educational background', r'academics', r'qualification',
        r'qualifications', r'academic details', r'academic summary',
        r'scholastic profile', r'educational credentials', r'education summary',
        r'education & training'
    ]
    
    # Valid experience section headings (used to find section boundaries)
    VALID_EXPERIENCE_HEADINGS = [
        r'work experience', r'professional experience', r'experience', 
        r'employment history', r'job history', r'internships', 
        r'apprenticeship', r'freelance work', r'contract work'
    ]
    
    # Skills section headings (used to find section boundaries)
    SKILLS_HEADINGS = [
        r'skills', r'key skills', r'technical skills', r'technical skill', r'core skills', r'skill set',
        r'technical expertise', r'technologies', r'programming languages', r'competencies', r'proficiencies'
    ]
    
    def __init__(self, resume_text: str, use_ai_fallback: bool = False, store_in_db: bool = False, candidate_id: Optional[int] = None):
        """
        Initialize the education extractor with resume text.
        
        Args:
            resume_text: The resume text to parse
            use_ai_fallback: If True, use AI when Python extraction fails
            store_in_db: If True, store extracted education in database
            candidate_id: Candidate ID for database storage (required if store_in_db=True)
        """
        self.resume_text = resume_text
        self.lines = [line.strip() for line in resume_text.split('\n') if line.strip()]
        self.use_ai_fallback = use_ai_fallback
        self.store_in_db = store_in_db
        self.candidate_id = candidate_id
        self.ai_client = None
        self.ai_model = None
        
        if use_ai_fallback:
            self._initialize_ai_client()
    
    def _initialize_ai_client(self):
        """Initialize OpenAI or Azure OpenAI client for AI fallback."""
        try:
            if not ATSConfig:
                logger.warning("ATSConfig not available. AI fallback disabled.")
                self.use_ai_fallback = False
                return
            
            # Try Azure OpenAI first
            if ATSConfig.AZURE_OPENAI_ENDPOINT and ATSConfig.AZURE_OPENAI_API_KEY:
                self.ai_client = AzureOpenAI(
                    api_key=ATSConfig.AZURE_OPENAI_API_KEY,
                    api_version=ATSConfig.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=ATSConfig.AZURE_OPENAI_ENDPOINT
                )
                self.ai_model = ATSConfig.AZURE_OPENAI_DEPLOYMENT_NAME
                logger.info("Initialized Azure OpenAI client for education extraction")
            # Fallback to OpenAI
            elif ATSConfig.OPENAI_API_KEY:
                self.ai_client = OpenAI(api_key=ATSConfig.OPENAI_API_KEY)
                self.ai_model = ATSConfig.OPENAI_MODEL
                logger.info("Initialized OpenAI client for education extraction")
            else:
                logger.warning("No OpenAI API key found. AI fallback disabled.")
                self.use_ai_fallback = False
                
        except Exception as e:
            logger.error(f"Failed to initialize AI client: {e}")
            self.use_ai_fallback = False
    
    def _is_extraction_valid(self, education_list: List[str]) -> bool:
        """
        Validate if Python extraction is correct.
        
        Args:
            education_list: List of extracted education degrees
            
        Returns:
            True if extraction is valid, False otherwise
        """
        if not education_list:
            logger.warning("Python extraction returned empty list")
            return False
        
        # Check if any entry contains valid degree patterns
        degree_patterns = [
            r'\bB\.?\s*Tech\b', r'\bM\.?\s*Tech\b', r'\bB\.?E\.?\b', r'\bM\.?E\.?\b',
            r'\bBachelor\b', r'\bMaster\b', r'\bB\.?\s*Sc\.?\b', r'\bM\.?\s*Sc\.?\b',
            r'\bB\.?\s*Com\.?\b', r'\bM\.?\s*Com\.?\b', r'\bB\.?\s*A\.?\b', r'\bM\.?\s*A\.?\b',
            r'\bM\.?\s*C\.?\s*A\.?\b', r'\bB\.?\s*C\.?\s*A\.?\b', r'\bM\.?\s*B\.?\s*A\.?\b',
            r'\bB\.?\s*B\.?\s*A\.?\b', r'\bPh\.?\s*D\.?\b', r'\bDiploma\b', r'\bP\.?\s*G\.?\s*D\.?\s*C\.?\s*A\.?\b'
        ]
        
        for edu in education_list:
            edu_lower = edu.lower()
            # Check if it's too short (likely invalid)
            if len(edu.strip()) < 5:
                continue
            
            # Check if it contains a valid degree pattern
            for pattern in degree_patterns:
                if re.search(pattern, edu, re.IGNORECASE):
                    logger.info(f"Python extraction validated: Found valid degree in '{edu}'")
                    return True
        
        logger.warning(f"Python extraction invalid: No valid degree patterns found in {education_list}")
        return False
    
    def _extract_with_ai(self) -> Optional[List[Dict[str, Any]]]:
        """
        Extract education using AI as fallback with comprehensive structured extraction.
        
        Returns:
            List of education dictionaries with structured data, or None if extraction fails
        """
        if not self.ai_client or not self.use_ai_fallback:
            logger.warning("AI client not available for education extraction")
            return None
        
        try:
            prompt = f"""You are an expert Resume Education Extractor.

Your task:
Given ONLY the raw resume text, extract clean, structured education details.
Return results ONLY in JSON format exactly as specified.

================================================================================
SECTION 1 — EDUCATION SECTION DETECTION
================================================================================

Detect the education block using ANY of the following headings (case-insensitive):

"Education"
"Educational Background"
"Education Details"
"Education Qualification"
"Education Qualifications"
"Educational Qualifications"
"Educational Experience"
"Education Summary"
"Education & Training"
"Academic Details"
"Academics"
"Academic Background"
"Academic Summary"
"Scholastic Profile"
"Qualifications"
"Qualification"
"Profile – Academic"
"Education Highlights"

Stop extraction when a new unrelated section begins:
- Experience / Professional Experience / Employment / Work History
- Skills / Technical Skills
- Projects / Academic Projects
- Certifications / Achievements
- Summary / Objective / Personal Info
- Security Contests (Pattern 5 case)
- Extracurriculars

================================================================================
SECTION 2 — WHAT TO EXTRACT
================================================================================

Extract **each education entry** with:

- `degree` → Exact degree name (e.g., B.Tech, M.Tech, B.Sc, MS, MA, Diploma, 10+2, SSC)
- `specialization` → Branch/major (ECE, CSE, Geography, MPC, etc.)
- `institution` → College/University/School name
- `start_year` → YYYY (when present)
- `end_year` → YYYY (when present)
- `percentage_or_cgpa` → 75%, 9.3 CGPA, etc.
- `raw_text` → Full matched text for debugging

================================================================================
SECTION 3 — DEGREE IDENTIFICATION PATTERNS
================================================================================

You must detect degrees in **any format**, including:

------------ UNDERGRADUATE ------------
B.Tech / Bachelor of Technology  
B.E / Bachelor of Engineering  
B.Sc / Bachelor of Science  
B.Com / Bachelor of Commerce  
BBA / BCA / BSW  
Bachelor of Arts (BA)

------------ POSTGRADUATE ------------
M.Tech / Master of Technology  
M.E / Master of Engineering  
M.Sc / Master of Science  
Master of Computer Science (pattern 2)  
Master of Arts / MA  
Master of Business Administration / MBA  
MS (Computer Science)  
MCA

------------ SCHOOLING ------------
10+2  
Intermediate / Inter (MPC / BiPC / CEC)  
PUC / HSC  
SSC / SSLC / CBSE 10th  
SCC (Pattern 1)

------------ DIPLOMAS ------------
Diploma in <specialization>  
Polytechnic Diploma

================================================================================
SECTION 4 — SUPPORTED PATTERN STYLES
================================================================================

The extractor MUST correctly handle ALL of the following formats:

----------------------------- PATTERN 1 STYLE -----------------------------
• Bullet list with degree, year, institution, and percentage in same line  
Example:
● B.Tech in ECE (2017) from GITAM University with 77%

----------------------------- PATTERN 2 STYLE -----------------------------
• Multi-line educational entries with:
  - University name (single line)
  - Degree (separate line)
  - Duration (separate line)
Example:
Colorado Technical University
Master of Computer Science  
Aug 2019 – Current

----------------------------- PATTERN 3 STYLE -----------------------------
• Very short bullet-style entries without institution or year  
Example:
• Diploma in Computer Science

----------------------------- PATTERN 4 STYLE -----------------------------
• Degree on one line
• Specialization on next line
• University + year on next line

----------------------------- PATTERN 5 STYLE -----------------------------
• "Educational Experience" heading
• Degrees on separate lines
• Institutions repeated
• YEARS APPEAR BELOW SEPARATELY under SECURITY CONTESTS (IGNORE THESE)

----------------------------- PATTERN 6 STYLE -----------------------------
• Mixed bullet format + duration in "from … to …" form  
Example:
Bachelor of Technology in CSE from 2014 to 2018 in Anurag University

----------------------------- ADDITIONAL REAL-WORLD PATTERNS SUPPORTED -----------------------------

• Inline pattern:
B.Tech (CSE), 2016–2020, JNTU

• Table-like aligned entries:
Course        Institute         Year     CGPA  
B.Tech (CSE)  VIT Chennai       2020     8.7

• Reversed format:
University of Texas  
BSc Computer Science  
2015 – 2019

• Specialization in brackets:
B.Tech (Mechanical Engineering)

• Percentage/CGPA anywhere:
CGPA: 9.1/10  
Scored 85%

• School style:
SSC – 2012 – Narayana School – 9.3 CGPA

================================================================================
OUTPUT FORMAT
================================================================================

Return a JSON object with this exact structure:
{{
  "education_entries": [
    {{
      "degree": "B.Tech",
      "specialization": "Computer Science",
      "institution": "XYZ University",
      "start_year": 2018,
      "end_year": 2022,
      "percentage_or_cgpa": "8.5 CGPA",
      "raw_text": "B.Tech in Computer Science from XYZ University, 2018-2022, CGPA: 8.5/10"
    }},
    {{
      "degree": "SSC",
      "specialization": null,
      "institution": "ABC School",
      "start_year": 2016,
      "end_year": 2017,
      "percentage_or_cgpa": "85%",
      "raw_text": "SSC from ABC School, 2016-2017, 85%"
    }}
  ]
}}

IMPORTANT:
- Return ALL education entries found, not just the highest
- If a field is not found, use null (not empty string)
- Years should be integers (YYYY format)
- Return ONLY valid JSON, no markdown, no explanations
- If no education found, return: {{"education_entries": []}}

Resume text:
{self.resume_text[:12000]}

Return ONLY the JSON object:"""

            response = self.ai_client.chat.completions.create(
                model=self.ai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}  # Ensure JSON output
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # Try to parse JSON
            try:
                # Remove markdown code blocks if present
                if response_content.startswith('```'):
                    # Extract JSON from code block
                    import re
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_content, re.DOTALL)
                    if json_match:
                        response_content = json_match.group(1)
                
                ai_result = json.loads(response_content)
                education_entries = ai_result.get('education_entries', [])
                
                if not education_entries:
                    logger.warning("AI returned empty education entries")
                    return None
                
                logger.info(f"AI extracted {len(education_entries)} education entries")
                return education_entries
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI JSON response: {e}")
                logger.debug(f"Response content: {response_content[:500]}")
                return None
            
        except Exception as e:
            logger.error(f"AI education extraction failed: {e}")
            return None
    
    def _store_in_database(self, education: str, education_details: str):
        """
        Store extracted education in database.
        
        Args:
            education: Highest education degree
            education_details: Full education details
        """
        if not self.store_in_db:
            return
        
        if not ATSDatabase or not self.candidate_id:
            logger.warning("Database storage requested but ATSDatabase not available or candidate_id missing")
            return
        
        try:
            db = ATSDatabase()
            if not db.connect():
                logger.error("Failed to connect to database for education storage")
                return
            
            # Update education fields for the candidate
            query = """
                UPDATE resume_metadata 
                SET education = %s, education_details = %s
                WHERE candidate_id = %s
            """
            
            db.cursor.execute(query, (education, education_details, self.candidate_id))
            db.connection.commit()
            logger.info(f"Stored education in database for candidate_id: {self.candidate_id}")
            db.disconnect()
            
        except Exception as e:
            logger.error(f"Failed to store education in database: {e}")
    
    def find_section(self, headings: List[str]) -> Optional[int]:
        """
        Find the line index where a section starts.
        
        Args:
            headings: List of heading patterns to search for
            
        Returns:
            Line index where section starts, or None if not found
        """
        for i, line in enumerate(self.lines):
            line_lower = line.lower().strip()
            # Remove trailing colons, dashes, etc. for matching
            line_clean = re.sub(r'[:–\-•\s]+$', '', line_lower)
            for heading in headings:
                # Check if line starts with heading or contains it as a major section header
                # Section headers are usually standalone or at the start
                heading_clean = heading.strip()
                if re.search(rf'^{re.escape(heading_clean)}\b', line_clean, re.IGNORECASE) or \
                   (len(line_clean.split()) <= 3 and heading_clean in line_clean):
                    return i
        return None
    
    def extract(self) -> List[str]:
        """
        Extract only REAL degrees with full degree names and specializations.
        Ignores trainings, certifications, workshops.
        
        Uses Python extraction first, then AI fallback if extraction is invalid.
        Optionally stores result in database.
        
        Returns:
            List of extracted education degrees (cleaned and deduplicated)
        """
        # Step 1: Try Python extraction
        education_list = []
        section_start = self.find_section(self.EDUCATION_HEADINGS)
        
        if section_start is None:
            logger.warning("No education section found in resume")
        else:
            # Find where education section ends (next major section or end of resume)
            section_end = len(self.lines)
            for i in range(section_start + 1, len(self.lines)):
                line_lower = self.lines[i].lower()
                # Check if it's another major section
                if any(re.search(rf'^{re.escape(heading)}', line_lower, re.IGNORECASE) 
                       for heading in self.VALID_EXPERIENCE_HEADINGS + self.SKILLS_HEADINGS):
                    section_end = i
                    break
            
            # Extract education from this section
            education_lines = self.lines[section_start + 1:section_end]
            
            # Common degree patterns (including space after period)
            # Test pattern: should match "B. Tech", "B.Tech", "B Tech"
            degree_patterns = [
                r'\bB\.?\s*Tech\b',
                r'\bB\.?Tech\b',
                r'\bB\.?E\.?\b',
                r'\bBachelor of Technology\b',
                r'\bBachelor of Engineering\b',
                r'\b(M\.?\s*Tech|M\.?Tech|M\.?E\.?|Master of Technology|Master of Engineering)\b',
                r'\b(B\.?\s*Sc\.?|B\.?Sc\.?|Bachelor of Science)\b',
                r'\b(M\.?\s*Sc\.?|M\.?Sc\.?|Master of Science)\b',
                r'\b(B\.?\s*Com\.?|B\.?Com\.?|Bachelor of Commerce)\b',
                r'\b(M\.?\s*Com\.?|M\.?Com\.?|Master of Commerce)\b',
                r'\b(B\.?\s*A\.?|B\.?A\.?|Bachelor of Arts)\b',
                r'\b(M\.?\s*A\.?|M\.?A\.?|Master of Arts)\b',
                r'\b(M\.?\s*C\.?\s*A\.?|M\.?C\.?A\.?|Master of Computer Applications)\b',
                r'\b(B\.?\s*C\.?\s*A\.?|B\.?C\.?A\.?|Bachelor of Computer Applications)\b',
                r'\b(M\.?\s*B\.?\s*A\.?|M\.?B\.?A\.?|Master of Business Administration)\b',
                r'\b(B\.?\s*B\.?\s*A\.?|B\.?B\.?A\.?|Bachelor of Business Administration)\b',
                r'\b(Ph\.?\s*D\.?|Ph\.?D\.?|Doctor of Philosophy)\b',
                r'\b(Diploma)\b',
                r'\b(P\.?\s*G\.?\s*D\.?\s*C\.?\s*A\.?|P\.?G\.?D\.?C\.?A\.?|Post Graduate Diploma)\b'
            ]
            
            current_degree = []
            for line in education_lines:
                # Skip empty lines
                if not line:
                    if current_degree:
                        degree_text = ' '.join(current_degree)
                        if any(re.search(pattern, degree_text, re.IGNORECASE) for pattern in degree_patterns):
                            education_list.append(degree_text.strip())
                        current_degree = []
                    continue
                
                # Skip lines that are clearly section headers or non-degree content
                line_lower = line.lower()
                if any(word in line_lower for word in ['certification', 'training', 'workshop', 'course']) and not any(re.search(pattern, line, re.IGNORECASE) for pattern in degree_patterns):
                    if current_degree:
                        degree_text = ' '.join(current_degree)
                        if any(re.search(pattern, degree_text, re.IGNORECASE) for pattern in degree_patterns):
                            education_list.append(degree_text.strip())
                        current_degree = []
                    continue
                
                # Check if line contains a degree indicator
                line_lower = line.lower()
                if any(re.search(pattern, line, re.IGNORECASE) for pattern in degree_patterns):
                    if current_degree:
                        degree_text = ' '.join(current_degree)
                        if any(re.search(pattern, degree_text, re.IGNORECASE) for pattern in degree_patterns):
                            education_list.append(degree_text.strip())
                    # Start new degree entry - include the full line to capture specialization
                    # e.g., "BTech, Civil Engineering" should be captured as one entry
                    current_degree = [line]
                elif current_degree:
                    # Continue building current degree entry (but stop at "Career objectives" or similar)
                    # Check if this line looks like it's part of the degree (specialization, etc.)
                    # or if it's a new section (CGPA, university name, etc.)
                    line_lower = line.lower()
                    
                    # Stop if it's clearly a new section or metadata
                    if any(word in line_lower for word in ['career objectives', 'objective', 'certification', 'work history', 'experience']):
                        # Save current degree and stop
                        degree_text = ' '.join(current_degree)
                        if any(re.search(pattern, degree_text, re.IGNORECASE) for pattern in degree_patterns):
                            education_list.append(degree_text.strip())
                        current_degree = []
                    # Stop if line contains CGPA/GPA/percentage (metadata, not part of degree name)
                    elif re.search(r'\b(CGPA|GPA|percentage|%)\b', line_lower):
                        # Save current degree before this metadata line
                        degree_text = ' '.join(current_degree)
                        if any(re.search(pattern, degree_text, re.IGNORECASE) for pattern in degree_patterns):
                            education_list.append(degree_text.strip())
                        current_degree = []
                    # Stop if line looks like a university name (contains "University", "College", etc.)
                    elif re.search(r'\b(University|College|Institute|School)\b', line, re.IGNORECASE):
                        # Save current degree before university name
                        degree_text = ' '.join(current_degree)
                        if any(re.search(pattern, degree_text, re.IGNORECASE) for pattern in degree_patterns):
                            education_list.append(degree_text.strip())
                        current_degree = []
                    # Otherwise, continue building (might be specialization on next line)
                    else:
                        # Only add if it looks like it could be part of the degree (short line, no numbers)
                        # This handles cases like "Civil Engineering" on a separate line
                        if len(line.split()) <= 5 and not re.search(r'\d', line):
                            current_degree.append(line)
                        else:
                            # Save current degree and start fresh
                            degree_text = ' '.join(current_degree)
                            if any(re.search(pattern, degree_text, re.IGNORECASE) for pattern in degree_patterns):
                                education_list.append(degree_text.strip())
                            current_degree = []
                elif len(line.split()) <= 15 and 'from' in line_lower:  # Potential degree line with "from"
                    # Check if it contains a degree pattern
                    if any(re.search(pattern, line, re.IGNORECASE) for pattern in degree_patterns):
                        current_degree = [line]
            
            # Add last degree if exists
            if current_degree:
                degree_text = ' '.join(current_degree)
                if any(re.search(pattern, degree_text, re.IGNORECASE) for pattern in degree_patterns):
                    education_list.append(degree_text.strip())
        
        # Clean and deduplicate
        cleaned = []
        for edu in education_list:
            # Remove years, percentages, GPAs, university names (keep only degree info)
            # Remove year ranges
            edu = re.sub(r'\d{4}[-–]\d{4}', '', edu)
            # Remove standalone years
            edu = re.sub(r'\b\d{4}\b', '', edu)
            # Remove percentages
            edu = re.sub(r'\d+\.?\d*%', '', edu)
            # Remove GPA/CGPA with any trailing characters
            edu = re.sub(r'GPA[:\s]*\d+\.?\d*[/]?\d*[^\w\s]*', '', edu, flags=re.IGNORECASE)
            edu = re.sub(r'CGPA[:\s]*\d+\.?\d*[/]?\d*[^\w\s]*', '', edu, flags=re.IGNORECASE)
            # Remove "from" and everything after it (university name, location, etc.)
            # But keep the degree name before "from"
            if ' from ' in edu.lower():
                parts = re.split(r'\s+from\s+', edu, flags=re.IGNORECASE, maxsplit=1)
                if parts:
                    edu = parts[0].strip()
            # Remove common university indicators (but keep if it's part of degree name)
            # Match university names - look for institution names at the end
            # Pattern: word(s) followed by University/College/Institute/School
            # But be careful not to remove specialization fields (e.g., "Civil Engineering")
            # Only remove if it's clearly a university name (contains "University", "College", etc.)
            edu = re.sub(r'[,\s]+[A-Z][A-Za-z\s]*(?:University|College|Institute|School)[,\s]*.*$', '', edu, flags=re.IGNORECASE)
            edu = re.sub(r',\s*[A-Z][A-Za-z\s]*(?:University|College|Institute|School)[,\s]*.*', '', edu, flags=re.IGNORECASE)
            # Remove standalone university abbreviations like "JNTU" at the end
            # But only if it's not part of a specialization (e.g., don't remove "Engineering" from "Civil Engineering")
            # Only remove if it's a short acronym (2-6 letters) at the very end
            edu = re.sub(r'\s+[A-Z]{2,6}\s*$', '', edu)  # Remove 2-6 letter acronyms at end
            # Remove everything after "Career objectives" or similar sections
            edu = re.sub(r'\s+Career objectives.*$', '', edu, flags=re.IGNORECASE)
            edu = re.sub(r'\s+Objective.*$', '', edu, flags=re.IGNORECASE)
            # Clean up extra spaces
            edu = ' '.join(edu.split()).strip()
            # Fix missing spaces after commas (e.g., "BTech,CivilEngineering" -> "BTech, Civil Engineering")
            # Apply multiple times to handle all commas
            while re.search(r',([A-Za-z])', edu):
                edu = re.sub(r',([A-Za-z])', r', \1', edu)  # Add space after comma if missing
            # Remove trailing commas, dashes, and other punctuation (but keep periods that are part of abbreviations)
            # IMPORTANT: Don't remove commas in the middle (e.g., "BTech, Civil Engineering" should keep the comma)
            edu = re.sub(r'[,–\-]+$', '', edu).strip()  # Only remove trailing punctuation
            # Remove standalone single letters at the end (often artifacts from GPA removal)
            # But be careful not to remove letters that are part of abbreviations like "B.Tech"
            edu = re.sub(r'\s+[A-Z]\s*$', '', edu)  # Only remove if it's a standalone word at the end

            # Strip any lingering leading/trailing punctuation characters (., *, -, etc.)
            edu = re.sub(r'^[\W_]+', '', edu)
            edu = re.sub(r'[\W_]+$', '', edu)

            # Skip entries that are effectively empty or purely punctuation after cleaning
            alnum_payload = re.sub(r'[^A-Za-z0-9]+', '', edu)

            if edu and len(edu) > 3 and alnum_payload and edu not in cleaned:  # Filter out very short/punctuation artifacts
                cleaned.append(edu)
        
        # Step 2: Validate Python extraction
        is_valid = self._is_extraction_valid(cleaned)
        
        # Step 3: Use AI fallback if Python extraction is invalid
        if not is_valid and self.use_ai_fallback:
            logger.info("Python extraction invalid, trying AI fallback...")
            ai_education_entries = self._extract_with_ai()
            
            if ai_education_entries and isinstance(ai_education_entries, list) and len(ai_education_entries) > 0:
                # Convert AI structured entries to simple degree strings
                ai_cleaned = []
                for entry in ai_education_entries:
                    if isinstance(entry, dict):
                        degree = entry.get('degree', '')
                        specialization = entry.get('specialization', '')
                        
                        # Build degree string
                        if degree:
                            degree_str = degree
                            if specialization:
                                degree_str = f"{degree} in {specialization}"
                            elif 'in ' in entry.get('raw_text', ''):
                                # Try to extract specialization from raw_text if not provided
                                raw = entry.get('raw_text', '')
                                if ' in ' in raw.lower():
                                    parts = raw.split(' in ', 1)
                                    if len(parts) > 1:
                                        spec_part = parts[1].split(' from ')[0].split(',')[0].strip()
                                        if spec_part and len(spec_part) < 50:  # Reasonable specialization length
                                            degree_str = f"{degree} in {spec_part}"
                            
                            if degree_str not in ai_cleaned:
                                ai_cleaned.append(degree_str)
                
                if ai_cleaned:
                    # Replace cleaned list with AI results
                    cleaned = ai_cleaned
                    logger.info(f"Using AI-extracted education: {cleaned}")
                else:
                    logger.warning("AI extraction returned entries but couldn't parse degrees")
            else:
                logger.warning("AI extraction also failed or returned invalid result")
        
        # Step 4: Store in database if requested
        if self.store_in_db and cleaned:
            highest_degree = cleaned[0] if cleaned else None
            education_details = '\n'.join(cleaned) if cleaned else ''
            self._store_in_database(highest_degree, education_details)
        
        return cleaned


def extract_education(
    resume_text: str, 
    use_ai_fallback: bool = False, 
    store_in_db: bool = False, 
    candidate_id: Optional[int] = None
) -> List[str]:
    """
    Convenience function to extract education from resume text.
    
    Args:
        resume_text: The resume text to parse
        use_ai_fallback: If True, use AI when Python extraction fails
        store_in_db: If True, store extracted education in database
        candidate_id: Candidate ID for database storage (required if store_in_db=True)
        
    Returns:
        List of extracted education degrees
        
    Example:
        >>> resume = \"\"\"
        ... EDUCATION
        ... B.Tech in Computer Science
        ... XYZ University, 2018-2022
        ... \"\"\"
        >>> degrees = extract_education(resume)
        >>> print(degrees)
        ['B.Tech in Computer Science']
    """
    extractor = EducationExtractor(
        resume_text, 
        use_ai_fallback=use_ai_fallback,
        store_in_db=store_in_db,
        candidate_id=candidate_id
    )
    return extractor.extract()


if __name__ == "__main__":
    # Example usage
    sample_resume = """
    John Doe
    john.doe@email.com | +1-234-567-8900
    
    SUMMARY
    Experienced software developer with 5 years of experience...
    
    EDUCATION
    B.Tech 
    XYZ University, 2018-2022
    CGPA: 8.5/10
    
    M.Tech in Data Science
    ABC University, 2022-2024
    
    WORK EXPERIENCE
    Senior Python Developer
    ABC Company
    Jan 2020 - Present
    
    SKILLS
    Python, Django, React, SQL, AWS, Docker
    """
    
    # Using the class - Python extraction only (default, no AI)
    extractor = EducationExtractor(sample_resume)
    education = extractor.extract()
    print("Using class (Python extraction only - default):")
    print(education)
    
    # Using the class - with AI fallback (only if explicitly enabled)
    extractor_with_ai = EducationExtractor(sample_resume, use_ai_fallback=True)
    education_ai = extractor_with_ai.extract()
    print("\nUsing class (with AI fallback - explicitly enabled):")
    print(education_ai)
    
    # Using the convenience function (Python extraction only by default)
    education2 = extract_education(sample_resume)
    print("\nUsing function (Python extraction only - default):")
    print(education2)
    
    # Example with database storage (requires candidate_id)
    # education3 = extract_education(sample_resume, store_in_db=True, candidate_id=1)

