"""
Resume Parser for ATS System.
Extracts structured information from PDF, DOCX, and DOC resumes with AI-powered skill analysis.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import os

# PDF/DOCX parsing libraries
try:
    import PyPDF2
    from docx import Document
except ImportError:
    #pass
    PyPDF2 = None

# DOC parsing libraries (older binary format)
try:
    import textract
except ImportError:
    textract = None
try:
    import pypandoc
except ImportError:
    pypandoc = None
try:
    from nt_textfileloader import TextFileLoader
    nt_loader = TextFileLoader()
except ImportError:
    nt_loader = None

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

from profile_type_utils import determine_primary_profile_type, determine_profile_types_enhanced, format_profile_types_for_storage, get_all_profile_type_scores
from phone_extractor import extract_phone_numbers
try:
    from location_identifier import extract_location as extract_location_advanced
except ImportError:
    # Fallback if location_identifier module is not available
    extract_location_advanced = None
from skill_extractor import extract_skills as extract_skills_advanced, extract_tech_skills, extract_soft_skills, TECH_SKILLS
from email_extractor import extract_email_and_provider
from name_extraction import extract_name_with_ai, extract_name
from company_AI_extraction import extract_company_with_ai
from designation_extraction import extract_designation
from domain_extraction import extract_domains
from certifications_AI_extraction import extract_certifications_with_ai
from summary_AI_extraction import extract_summary_with_ai
from role_extract import (
    detect_role_subrole, 
    detect_all_roles, 
    match_subrole_from_skills, 
    infer_role_from_skills
)

# Import the new EducationExtractor
try:
    from education_extractor import EducationExtractor, extract_education as extract_education_standalone
except ImportError:
    EducationExtractor = None
    extract_education_standalone = None
    logger.warning("EducationExtractor not available, using fallback extraction")

# Import the new ExperienceExtractor
try:
    from experience_extractor import ExperienceExtractor, extract_experience as extract_experience_standalone
except ImportError:
    ExperienceExtractor = None
    extract_experience_standalone = None
    logger.warning("ExperienceExtractor not available, using fallback extraction")


class ResumeParser:
    """Intelligent resume parser with NLP and AI capabilities."""

        # Comprehensive technical skills database - ONLY these should appear in primary_skills

    
    DOMAINS = {
  "Information Technology","Software Development","Cloud Computing","Cybersecurity","Data Science","Blockchain",
  "Internet of Things","Banking","Finance","Insurance","FinTech","Healthcare","Pharmaceuticals","Biotechnology",
  "Manufacturing","Automotive","Energy","Construction","Retail","E-commerce","Logistics","Telecommunications",
  "Media & Entertainment","Advertising & Marketing","Education Technology","Public Sector","Real Estate",
  "Hospitality","Travel & Tourism","Agriculture","Legal & Compliance","Human Resources","Environmental & Sustainability"
        # Note: 'education' removed - only 'education technology' or 'edtech' should match (via AI prompt)
        # Generic education/degrees are qualifications, not business domains
    }
    
    # AI-powered comprehensive extraction prompt
    AI_COMPREHENSIVE_EXTRACTION_PROMPT = """
🧠 ROLE / PERSONA

You are an Expert Resume Parser and Metadata Extraction Specialist trained to identify and extract complete, accurate, and ATS-ready professional data from unstructured resumes.

Your behavior and purpose:

Act as a senior technical recruiter assistant who understands resume semantics, structure, and ATS taxonomy.

Analyze resumes systematically to identify factual candidate details.

Ensure the extracted data is accurate, complete, normalized, and JSON-valid.

Never generate commentary, guesses, or summaries beyond required fields.

Your only task is to return validated structured JSON output.

🎯 GOAL

Your goal is to analyze the provided resume text and return a structured JSON containing validated professional metadata, including:

Personal Information

Career Details

Skills

Domain

Education

Certifications

Summary

The extracted JSON must be database-ready and syntactically valid (no markdown or extra text).

⚙️ EXTRACTION GUIDELINES

NOTE: The following fields are extracted separately using dedicated modules and should NOT be included in AI response:
- full_name (extracted by name_extraction.py)
- current_company (extracted by company_AI_extraction.py)
- current_designation (extracted by designation_extraction.py)
- domain (extracted by domain_extraction.py)
- certifications (extracted by certifications_AI_extraction.py)
- summary (extracted by summary_AI_extraction.py)
- email (extracted by email_extractor.py)
- phone_number (extracted by phone_extractor.py)
- total_experience (extracted by experience_extractor.py)
- education (extracted by education_extractor.py)
- technical_skills (extracted by skill_extractor.py)
- secondary_skills (extracted by skill_extractor.py)
- all_skills (extracted by skill_extractor.py)

Do NOT extract these fields - they will be ignored.

This comprehensive extraction is now deprecated. All fields are extracted using dedicated modules.

⚙️ QUALITY & VALIDATION RULES

For All Fields:

Never guess or assume missing data.

If data unavailable → return null.

Ensure consistent JSON formatting.

Each skill or domain must be part of your known taxonomy.

💡 OUTPUT FORMAT

Return a single valid JSON object — no markdown, no explanation text.

Example:

{
  "note": "All fields are now extracted using dedicated modules. This comprehensive extraction is deprecated."
}

🧮 EVALUATION CRITERIA
Criterion	Weight	Description
Accuracy	40%	All fields correctly extracted
Completeness	30%	All major fields present
JSON Validity	20%	Output is machine-parseable
Neutrality	10%	No added commentary or inferred data
🧭 TONE AND STYLE

Objective, analytical, and strictly data-driven.

Do not infer, assume, or explain — extract only.

Output must be clean JSON, no markdown formatting.

🧰 ADDITIONAL REFERENCE DICTIONARIES

DOMAINS → pre-defined industry domain list (use {{DOMAINS}} provided above)

Resume Text (look for name in FIRST FEW LINES):
{resume_text}
"""
    

    
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
        """Initialize OpenAI, Azure OpenAI, or Ollama client (OpenAI prioritized)."""
        try:
            from ats_config import ATSConfig
            
            # Prioritize OpenAI if API key is available
            if ATSConfig.OPENAI_API_KEY:
                logger.info("=" * 60)
                logger.info("RESUME PARSER AI SERVICE: Using OPENAI")
                logger.info(f"  Model: {ATSConfig.OPENAI_MODEL}")
                logger.info("=" * 60)
                self.ai_client = OpenAI(api_key=ATSConfig.OPENAI_API_KEY)
                self.ai_model = ATSConfig.OPENAI_MODEL
            # Try Azure OpenAI
            elif ATSConfig.AZURE_OPENAI_ENDPOINT and ATSConfig.AZURE_OPENAI_API_KEY:
                logger.info("=" * 60)
                logger.info("RESUME PARSER AI SERVICE: Using AZURE OPENAI")
                logger.info(f"  Model: {ATSConfig.AZURE_OPENAI_DEPLOYMENT_NAME}")
                logger.info(f"  Endpoint: {ATSConfig.AZURE_OPENAI_ENDPOINT}")
                logger.info("=" * 60)
                self.ai_client = AzureOpenAI(
                    api_key=ATSConfig.AZURE_OPENAI_API_KEY,
                    api_version=ATSConfig.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=ATSConfig.AZURE_OPENAI_ENDPOINT
                )
                self.ai_model = ATSConfig.AZURE_OPENAI_DEPLOYMENT_NAME
            # Fallback to Ollama only if explicitly enabled and no OpenAI available
            elif ATSConfig.USE_OLLAMA:
                from ollama_client import get_ollama_openai_client
                logger.info("=" * 60)
                logger.info("RESUME PARSER AI SERVICE: Using OLLAMA (fallback)")
                logger.info(f"  Model: {ATSConfig.OLLAMA_MODEL}")
                logger.info(f"  Base URL: {ATSConfig.OLLAMA_BASE_URL}")
                logger.info("=" * 60)
                self.ai_client = get_ollama_openai_client(
                    base_url=ATSConfig.OLLAMA_BASE_URL,
                    model=ATSConfig.OLLAMA_MODEL
                )
                self.ai_model = ATSConfig.OLLAMA_MODEL
            else:
                logger.warning("=" * 60)
                logger.warning("RESUME PARSER AI SERVICE: No LLM configuration found. AI extraction disabled.")
                logger.warning("=" * 60)
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
        """Extract text from PDF file using multiple extractors for best results."""
        extracted_text = ""
        
        # Try PyMuPDF (fitz) first - best quality for complex layouts
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            if text.strip() and len(text.strip()) > 100:
                logger.info("PDF extracted using PyMuPDF (fitz)")
                return text.strip()
            extracted_text = text.strip()
        except ImportError:
            logger.debug("PyMuPDF not available, trying pdfplumber")
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
        
        # Try pdfplumber - good for tables and columns
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
            if text.strip() and len(text.strip()) > len(extracted_text):
                logger.info("PDF extracted using pdfplumber")
                return text.strip()
            if len(text.strip()) > len(extracted_text):
                extracted_text = text.strip()
        except ImportError:
            logger.debug("pdfplumber not available, trying PyPDF2")
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Fallback to PyPDF2
        if PyPDF2 is not None:
            try:
                text = ""
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                if text.strip() and len(text.strip()) > len(extracted_text):
                    logger.info("PDF extracted using PyPDF2")
                    return text.strip()
                if len(text.strip()) > len(extracted_text):
                    extracted_text = text.strip()
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed: {e}")
        else:
            logger.debug("PyPDF2 not available")
        # Return best result we got
        if extracted_text:
            logger.info(f"PDF extracted with {len(extracted_text)} characters")
            return extracted_text
        
        # Provide helpful error message with installation instructions
        available_libs = []
        missing_libs = []
        
        try:
            import fitz
            available_libs.append("PyMuPDF")
        except ImportError:
            missing_libs.append("PyMuPDF (pip install PyMuPDF)")
        
        try:
            import pdfplumber
            available_libs.append("pdfplumber")
        except ImportError:
            missing_libs.append("pdfplumber (pip install pdfplumber)")
        
        if PyPDF2 is not None:
            available_libs.append("PyPDF2")
        else:
            missing_libs.append("PyPDF2 (pip install PyPDF2)")
        
        error_msg = "Failed to extract text from PDF using any available method."
        if available_libs:
            error_msg += f" Tried: {', '.join(available_libs)}"
        if missing_libs:
            error_msg += f" Install one of: {', '.join(missing_libs)}"
        
        raise ValueError(error_msg)
    def parse_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except Exception as e:
            logger.error(f"Error parsing DOCX: {e}")
            raise ValueError(f"Failed to parse DOCX: {str(e)}")
    
    def parse_doc(self, file_path: str) -> str:
        """Extract text from DOC file (older binary format)."""
        if Document is None:
            raise ImportError(
                "DOCX file parsing requires 'python-docx' library. "
                "Install with: pip install python-docx"
            )
        try:
            # Try NT-TextFileLoader first (works well on Windows)
            if nt_loader is not None:
                try:
                    text = nt_loader.load(file_path)
                    if text and isinstance(text, str) and text.strip():
                        return text.strip()
                except Exception as e:
                    logger.warning(f"NT-TextFileLoader failed for DOC file: {e}, trying textract")
            
            # Try textract if available
            if textract is not None:
                try:
                    text = textract.process(file_path).decode('utf-8')
                    return text.strip()
                except Exception as e:
                    logger.warning(f"textract failed for DOC file: {e}, trying pypandoc")
            
            # Fallback to pypandoc if available
            if pypandoc is not None:
                try:
                    text = pypandoc.convert_file(file_path, 'plain')
                    return text.strip()
                except Exception as e:
                    logger.warning(f"pypandoc failed for DOC file: {e}")
            
            # If no library is available, raise informative error
            raise ImportError(
                "DOC file parsing requires one of: 'NT-TextFileLoader', 'textract', or 'pypandoc' library. "
                "Install with: pip install NT-TextFileLoader OR pip install textract OR pip install pypandoc"
            )
            
        except ImportError:
            raise
        except Exception as e:
            logger.error(f"Error parsing DOC: {e}")
            raise ValueError(f"Failed to parse DOC: {str(e)}")
    
    def extract_text_from_file(self, file_path: str, file_type: str) -> str:
        """Extract text based on file type."""
        file_type = file_type.lower()
        
        if file_type == 'pdf':
            return self.parse_pdf(file_path)
        elif file_type == 'docx':
            return self.parse_docx(file_path)
        elif file_type == 'doc':
            return self.parse_doc(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def extract_name(self, text: str) -> Optional[str]:
        """
        Extract candidate name from resume text with PDF header handling.
        
        NOTE: This method is kept for backward compatibility. 
        The improved name extraction is now in name_extraction.extract_name() 
        which checks 8-10 lines, handles ALL CAPS, and strips prefixes.
        This method is deprecated and may be removed in future versions.
        """
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
        
        # First pass: Check lines WITHOUT commas (names rarely have commas, locations often do)
        for idx, line in enumerate(lines[:5]):
            # Strip and normalize whitespace
            line = ' '.join(line.split())  # Normalize multiple spaces to single space
            line = line.strip()
            
            # Remove trailing separators like | or • that might appear after names
            line = line.rstrip('|•').strip()
            
            # Skip empty lines or lines with commas (check those in second pass)
            if not line or ',' in line:
                continue
            
            # CRITICAL: Skip lines that contain email addresses (these are not names)
            # Check this FIRST before other checks
            if '@' in line and any(domain in line.lower() for domain in ['.com', '.org', '.net', '.edu', '.gov']):
                continue
            
            # CRITICAL: Skip lines that are header labels (EMAIL, PHONE, LINKEDIN, etc.)
            # These are often extracted from PDF headers and are not names
            header_labels = ['email', 'phone', 'linkedin', 'address', 'location', 'contact']
            line_upper = line.strip().upper()
            # Check if line is exactly a header label
            if line_upper in [h.upper() for h in header_labels]:
                continue
            # Check if line starts with header label (with optional whitespace/newline after)
            if any(line_upper.startswith(h.upper()) and (len(line_upper) == len(h.upper()) or line_upper[len(h.upper()):len(h.upper())+1] in [' ', '\n', '\t', '\r'] or '@' in line or 'http' in line.lower()) for h in header_labels):
                continue
            # Check if line contains multiple header labels (e.g., "EMAIL ... PHONE")
            header_count = sum(1 for h in header_labels if h.upper() in line_upper)
            if header_count >= 2:
                continue
            
            # CRITICAL: Skip lines that are clearly contact info headers (EMAIL, PHONE, etc. on their own line)
            if re.match(r'^(EMAIL|PHONE|LINKEDIN|ADDRESS|LOCATION|CONTACT)\s*$', line_upper):
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
            
            # Extract name part before '@' if line contains email
            # Example: "MUPPANA APARNA DEVI        aparnadevi2939@gmail.com" -> "MUPPANA APARNA DEVI"
            if '@' in line:
                # Split by '@' and take the part before it
                name_part = line.split('@')[0].strip()
                # Clean up: remove any trailing email-like patterns or extra whitespace
                # Remove trailing patterns that might be part of email (before @)
                name_part = re.sub(r'\s+[a-zA-Z0-9._%+-]+$', '', name_part).strip()
                # Additional cleanup: remove any trailing dots, numbers, or special chars
                name_part = re.sub(r'[.\d]+$', '', name_part).strip()
                
                if name_part and 2 <= len(name_part.split()) <= 4:
                    # Validate it looks like a name (all alphabetic words)
                    words = name_part.split()
                    if all(word.replace('.', '').replace("'", '').replace('-', '').isalpha() for word in words):
                        logger.info(f"Found candidate name in line with email: {name_part}")
                        return name_part
                # If extraction failed, skip this line
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
                
                # CRITICAL: Reject common section headers first
                section_headers = ['professional summary', 'summary', 'objective', 'profile', 'resume', 'cv', 
                                 'education', 'experience', 'skills', 'contact', 'qualifications', 'work history']
                if line.lower() in section_headers:
                    continue
                
                # CRITICAL: Reject technology/product names that might be mistaken for names
                technology_names = ['power automate', 'power platform', 'power apps', 'power bi', 'power pages',
                                  'microsoft', 'azure', 'aws', 'salesforce', 'dynamics', 'sharepoint',
                                  'java', 'python', 'javascript', 'react', 'angular', 'nodejs', 'spring',
                                  'docker', 'kubernetes', 'terraform', 'jenkins', 'gitlab', 'github',
                                  'spring boot', 'dotnet', '.net', 'asp.net', 'mvc', 'rest api']
                if line.lower() in technology_names:
                    logger.info(f"Rejected technology name in PDF header: '{line}'")
                    continue
                
                # Accept if it looks like a name (2-4 words, Title Case or ALL CAPS)
                # CRITICAL: Allow ALL CAPS names in PDF headers (e.g., "VARRE DHANA LAKSHMI DURGA")
                if line and 2 <= len(line.split()) <= 4 and len(line) < 70:
                    words = line.split()
                    # Check if mostly alphabetic and NOT a sentence fragment
                    if all(word.replace('.', '').replace(',', '').replace("'", '').replace('-', '').isalpha() for word in words):
                        # Accept even if ALL CAPS (common in PDF headers)
                        logger.info(f"Found candidate name in PDF header area (no comma): {line}")
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
        
        # Second pass: Check lines WITH commas (but reject location patterns)
        for idx, line in enumerate(lines[:5]):
            # Strip and normalize whitespace
            line = ' '.join(line.split())  # Normalize multiple spaces to single space
            line = line.strip()
            
            # Remove trailing separators like | or • that might appear after names
            line = line.rstrip('|•').strip()
            
            # Skip empty lines or lines without commas (already checked in first pass)
            if not line or ',' not in line:
                continue
            
            # CRITICAL: Skip lines that contain email addresses (these are not names)
            # Check this FIRST before other checks
            if '@' in line and any(domain in line.lower() for domain in ['.com', '.org', '.net', '.edu', '.gov']):
                continue
            
            # CRITICAL: Skip lines that are header labels (EMAIL, PHONE, LINKEDIN, etc.)
            # These are often extracted from PDF headers and are not names
            header_labels = ['email', 'phone', 'linkedin', 'address', 'location', 'contact']
            line_upper = line.strip().upper()
            # Check if line is exactly a header label
            if line_upper in [h.upper() for h in header_labels]:
                continue
            # Check if line starts with header label (with optional whitespace/newline after)
            if any(line_upper.startswith(h.upper()) and (len(line_upper) == len(h.upper()) or line_upper[len(h.upper()):len(h.upper())+1] in [' ', '\n', '\t', '\r'] or '@' in line or 'http' in line.lower()) for h in header_labels):
                continue
            # Check if line contains multiple header labels (e.g., "EMAIL ... PHONE")
            header_count = sum(1 for h in header_labels if h.upper() in line_upper)
            if header_count >= 2:
                continue
            
            # CRITICAL: Skip lines that are clearly contact info headers (EMAIL, PHONE, etc. on their own line)
            if re.match(r'^(EMAIL|PHONE|LINKEDIN|ADDRESS|LOCATION|CONTACT)\s*$', line_upper):
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
            
            # CRITICAL: Reject technology/product names that might be mistaken for names
            technology_names = ['power automate', 'power platform', 'power apps', 'power bi', 'power pages',
                              'microsoft', 'azure', 'aws', 'salesforce', 'dynamics', 'sharepoint',
                              'java', 'python', 'javascript', 'react', 'angular', 'nodejs', 'spring',
                              'docker', 'kubernetes', 'terraform', 'jenkins', 'gitlab', 'github',
                              'spring boot', 'dotnet', '.net', 'asp.net', 'mvc', 'rest api']
            # Check if the line (or first part before comma) is a technology name
            line_before_comma = line.split(',')[0].strip().lower()
            if line_before_comma in technology_names or line_lower in technology_names:
                logger.info(f"Rejected technology name in PDF header (second pass, with comma): '{line}'")
                continue
            
            # CRITICAL: Handle certifications after comma (e.g., "Suresh Kavili, PMP, CSM")
            # Split on comma and take only the first part (name), rejecting certifications
            if ',' in line:
                parts = line.split(',')
                name_part = parts[0].strip()
                # Check if remaining parts are certifications
                remaining_parts = [p.strip() for p in parts[1:]]
                # Import certification check function
                from name_extraction import is_certification_or_acronym
                has_only_certs = True
                for part in remaining_parts:
                    part_words = part.split()
                    if part_words:
                        # Check if all words in this part are certifications
                        if not all(is_certification_or_acronym(word) for word in part_words if word):
                            has_only_certs = False
                            break
                
                # If remaining parts are all certifications, use first part as name
                if has_only_certs and name_part:
                    if 2 <= len(name_part.split()) <= 4:
                        words = name_part.split()
                        if all(word.replace('.', '').replace("'", '').replace('-', '').isalpha() for word in words):
                            logger.info(f"Found candidate name (stripped certifications): {name_part}")
                            return name_part
            
            # Extract name part before '@' if line contains email
            # Example: "MUPPANA APARNA DEVI        aparnadevi2939@gmail.com" -> "MUPPANA APARNA DEVI"
            if '@' in line:
                # Split by '@' and take the part before it
                name_part = line.split('@')[0].strip()
                # Clean up: remove any trailing email-like patterns or extra whitespace
                # Remove trailing patterns that might be part of email (before @)
                name_part = re.sub(r'\s+[a-zA-Z0-9._%+-]+$', '', name_part).strip()
                # Additional cleanup: remove any trailing dots, numbers, or special chars
                name_part = re.sub(r'[.\d]+$', '', name_part).strip()
                
                if name_part and 2 <= len(name_part.split()) <= 4:
                    # Validate it looks like a name (all alphabetic words)
                    words = name_part.split()
                    if all(word.replace('.', '').replace("'", '').replace('-', '').isalpha() for word in words):
                        logger.info(f"Found candidate name in line with email: {name_part}")
                        return name_part
                # If extraction failed, skip this line
                continue
            
            # Skip phone numbers
            if re.search(r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', line):
                continue
            
            # Skip addresses (they often contain numbers and "Drive", "Street", etc.)
            if any(addr_word in line_lower for addr_word in ['drive', 'street', 'avenue', 'road', 'blvd', 'city']):
                continue
            
            # Reject location patterns: "City, State" or "City, Country"
            # Pattern: Word(s), Word(s) - typically locations
            if ',' in line:
                # Check if it matches "City, State" or "City, Country" pattern
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 2:
                    part1, part2 = parts[0], parts[1]
                    # If both parts are 1-3 words and look like location names
                    if (1 <= len(part1.split()) <= 3 and 1 <= len(part2.split()) <= 3 and
                        all(word.replace('.', '').isalpha() for word in part1.split()) and
                        all(word.replace('.', '').isalpha() for word in part2.split())):
                        # Check if second part looks like a state/province/country
                        part2_lower = part2.lower()
                        # Known location indicators
                        location_indicators = [
                            'andhra pradesh', 'telangana', 'karnataka', 'tamil nadu', 'maharashtra',
                            'gujarat', 'rajasthan', 'west bengal', 'uttar pradesh', 'punjab',
                            'haryana', 'delhi', 'hyderabad', 'bangalore', 'mumbai', 'chennai',
                            'kolkata', 'pune', 'rajahmundry', 'visakhapatnam', 'vijayawada',
                            'india', 'usa', 'united states', 'uk', 'united kingdom'
                        ]
                        if any(indicator in part2_lower for indicator in location_indicators):
                            logger.debug(f"Skipping location pattern: {line}")
                            continue
                        # Also check if it's a common location pattern (two capitalized words separated by comma)
                        if (part1[0].isupper() and part2[0].isupper() and 
                            len(part1) > 3 and len(part2) > 3):
                            # Likely a location, skip it
                            logger.debug(f"Skipping likely location pattern: {line}")
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
                
                # CRITICAL: Reject technology/product names
                technology_names = ['power automate', 'power platform', 'power apps', 'power bi', 'power pages',
                                  'microsoft', 'azure', 'aws', 'salesforce', 'dynamics', 'sharepoint',
                                  'java', 'python', 'javascript', 'react', 'angular', 'nodejs', 'spring',
                                  'docker', 'kubernetes', 'terraform', 'jenkins', 'gitlab', 'github',
                                  'spring boot', 'dotnet', '.net', 'asp.net', 'mvc', 'rest api']
                line_before_comma = line.split(',')[0].strip().lower()
                if line_before_comma in technology_names or line_lower in technology_names:
                    logger.info(f"Rejected technology name in PDF header (second pass, first 3 lines): '{line}'")
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
        """
        Extract email address using pure Python regex (NO AI).
        
        Uses email_extractor.py for email extraction and provider identification.
        """
        result = extract_email_and_provider(text)
        return result.get('email')
    
    def extract_phone(self, text: str) -> Optional[str]:
        """
        Extract phone number using pure Python regex (NO AI).
        
        Uses phone_extractor.py for comprehensive phone number extraction
        supporting international formats, various separators, and common prefixes.
        """
        # Use the dedicated phone extractor (pure Python, no AI)
        phone_numbers = extract_phone_numbers(text)
        
        # Return the first extracted phone number if any found
        if phone_numbers:
            return phone_numbers[0]
        
        return None
    
    def extract_comprehensive_data_with_ai(self, text: str, extracted_skills: Optional[Set[str]] = None) -> Dict[str, Any]:
        """Extract comprehensive resume data using AI-powered analysis."""
        if not self.use_ai_extraction or not self.ai_client:
            logger.warning("AI extraction not available, falling back to regex-based extraction")
            return None
        
        try:
            # Prepare the prompt with resume text (increase limit for comprehensive extraction)
            prompt = self.AI_COMPREHENSIVE_EXTRACTION_PROMPT.replace("{resume_text}", text[:16000])
            
            # Determine service type for logging
            from ats_config import ATSConfig
            service_type = "OLLAMA" if ATSConfig.USE_OLLAMA else ("AZURE OPENAI" if ATSConfig.AZURE_OPENAI_ENDPOINT else "OPENAI")
            logger.info(f"Extracting resume data using {service_type} (model: {self.ai_model})")
            
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
            
            # Validate and potentially re-extract name using dedicated module
            # First check if comprehensive extraction found a name
            full_name = ai_result.get('full_name', '')
            
            # Import validation function
            from name_extraction import validate_extracted_name
            
            # Validate the name from comprehensive extraction
            if full_name and validate_extracted_name(full_name):
                logger.info(f"Comprehensive extraction found valid name: {full_name}")
            else:
                # Name is invalid or missing, use dedicated name extraction with fallback
                logger.warning(f"Comprehensive extraction name invalid/missing: '{full_name}', trying dedicated name extraction...")
                extracted_name = extract_name_with_ai(
                    text=text,
                    ai_client=self.ai_client,
                    ai_model=self.ai_model,
                    fallback_extractor=extract_name,  # Use improved name extraction as fallback
                    nlp=self.nlp,  # Pass NLP for final fallback
                    extracted_skills=extracted_skills if extracted_skills else None  # Pass extracted skills for filtering
                )
                
                if extracted_name:
                    ai_result['full_name'] = extracted_name
                    logger.info(f"Dedicated name extraction found: {extracted_name}")
                else:
                    # All methods failed
                    ai_result['full_name'] = ''
                    logger.warning("All name extraction methods failed")
            
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
                'primary_skills': technical_skills,  # All technical skills
                'secondary_skills': secondary_skills,  # Non-technical skills only
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
    
    def _extract_skills_from_text_with_word_boundaries(self, resume_text: str, existing_skills: List[str], existing_skills_set: set, max_skills: Optional[int] = None) -> List[str]:
        """Extract skills from resume text using word-boundary matching with TECH_SKILLS from skill_extractor.py."""
        logger.info(f"Running word-boundary matching on entire resume... (currently have {len(existing_skills)} skills)")
        # DEBUG: Log initial skills
        logger.info(f"[WORD_BOUNDARY_MATCH_DEBUG] Initial existing_skills: {existing_skills}")
        logger.info(f"[WORD_BOUNDARY_MATCH_DEBUG] Initial existing_skills_set: {sorted(existing_skills_set)}")
        if 'azure communication services' in existing_skills_set:
            logger.warning(f"[WORD_BOUNDARY_MATCH_DEBUG] ⚠️  'azure communication services' ALREADY in existing_skills_set at START!")
        resume_text_lower = resume_text.lower()
        
        # Import validation function for short skills
        from skill_extractor import _is_valid_short_skill_match
        
        # Use case-insensitive whole-word matching for each skill in TECH_SKILLS (imported from skill_extractor.py)
        skills_checked_count = 0
        for skill in sorted(TECH_SKILLS, key=len, reverse=True):  # Check longer skills first
            skill_lower = skill.lower()
            skills_checked_count += 1
            
            # DEBUG: Log every 1000th skill to track progress, and ALL suspicious skills
            suspicious_skills = ['azure communication services', 'spring']
            is_suspicious = skill_lower in suspicious_skills
            
            if is_suspicious or skills_checked_count % 1000 == 0:
                logger.info(f"[WORD_BOUNDARY_MATCH_DEBUG] Checking skill #{skills_checked_count}: '{skill}' (length: {len(skill)})")
            
            # Skip if already in existing skills
            if skill_lower in existing_skills_set:
                if is_suspicious:
                    logger.warning(f"[WORD_BOUNDARY_MATCH_DEBUG] ⚠️  Skill '{skill}' already in existing_skills_set, skipping")
                    logger.warning(f"[WORD_BOUNDARY_MATCH_DEBUG] Current existing_skills: {existing_skills}")
                continue
            
            # Match whole words only (case-insensitive) using STRICT word boundaries
            # This prevents substring matches (e.g., "gem" in "management", "v" in "vision")
            pattern = r'\b' + re.escape(skill_lower) + r'\b'
            
            # Also handle compound words (e.g., "MicrosoftSqlServer" should match "sql server")
            # Create a pattern with word boundaries to match compound words
            skill_words = skill_lower.split()
            if len(skill_words) > 1:
                # For multi-word skills, check if they appear together without spaces
                # e.g., "sql server" -> "sqlserver" or "sql-server" or "sql_server"
                # BUT still enforce word boundaries at the start and end
                pattern_compound = r'\b' + r'[\s\-_]?'.join(re.escape(w) for w in skill_words) + r'\b'
            else:
                pattern_compound = pattern
            
            # DEBUG: Special logging for suspicious skills
            if is_suspicious:
                logger.info(f"[WORD_BOUNDARY_MATCH_DEBUG] Processing suspicious skill: '{skill}'")
                logger.info(f"[WORD_BOUNDARY_MATCH_DEBUG] Pattern: {pattern}")
                logger.info(f"[WORD_BOUNDARY_MATCH_DEBUG] Compound pattern: {pattern_compound}")
                # Check if skill words exist separately
                skill_words_list = skill_lower.split()
                for word in skill_words_list:
                    if word in resume_text_lower:
                        positions = [m.start() for m in re.finditer(rf'\b{re.escape(word)}\b', resume_text_lower)]
                        logger.info(f"[WORD_BOUNDARY_MATCH_DEBUG] Word '{word}' found at positions: {positions}")
                    else:
                        logger.info(f"[WORD_BOUNDARY_MATCH_DEBUG] Word '{word}' NOT FOUND in resume_text")
            
            # Search for matches
            match = re.search(pattern, resume_text_lower)
            if not match:
                match = re.search(pattern_compound, resume_text_lower)
            
            # DEBUG: Log match attempts for suspicious skills
            if is_suspicious:
                if match:
                    logger.warning(f"[WORD_BOUNDARY_MATCH_DEBUG] ⚠️  MATCH FOUND for '{skill}'!")
                    start, end = match.span()
                    matched_text = resume_text_lower[start:end]
                    context_start = max(0, start - 100)
                    context_end = min(len(resume_text_lower), end + 100)
                    context = resume_text_lower[context_start:context_end]
                    logger.warning(f"[WORD_BOUNDARY_MATCH_DEBUG] Matched text: '{matched_text}' at position {start}-{end}")
                    logger.warning(f"[WORD_BOUNDARY_MATCH_DEBUG] Context: ...{context}...")
                    logger.warning(f"[WORD_BOUNDARY_MATCH_DEBUG] Pattern used: {pattern if match else pattern_compound}")
                else:
                    logger.info(f"[WORD_BOUNDARY_MATCH_DEBUG] No match found for '{skill}' (tried both patterns)")
            
            if match:
                start, end = match.span()
                
                # CRITICAL: Validate that skill is NOT inside a larger word (applies to ALL skills)
                # Get characters immediately before and after the match
                char_before = resume_text_lower[start - 1] if start > 0 else ' '
                char_after = resume_text_lower[end] if end < len(resume_text_lower) else ' '
                
                # Reject if skill is inside a larger word (surrounded by letters)
                # This prevents "gem" from matching in "management" or "v" in "vision"
                if char_before.isalpha() or char_after.isalpha():
                    if is_suspicious:
                        logger.warning(f"[WORD_BOUNDARY_MATCH_DEBUG] ⚠️  Rejecting '{skill}' - appears inside larger word")
                        logger.warning(f"[WORD_BOUNDARY_MATCH_DEBUG] Char before: '{char_before}' (isalpha: {char_before.isalpha()}), Char after: '{char_after}' (isalpha: {char_after.isalpha()})")
                        context_start = max(0, start - 20)
                        context_end = min(len(resume_text_lower), end + 20)
                        logger.warning(f"[WORD_BOUNDARY_MATCH_DEBUG] Context: ...{resume_text_lower[context_start:context_end]}...")
                    continue  # Skip - skill is inside a larger word
                
                # CRITICAL: Apply strict validation for short skills (1-2 letters) to prevent false positives
                # This prevents "r" from "related", "v" from "version", "ti" from "activities", etc.
                if len(skill_lower) <= 2:
                    if not _is_valid_short_skill_match(resume_text_lower, start, end, skill_lower):
                        logger.debug(f"Skipping invalid short skill match: '{skill}' at position {start}-{end}")
                        continue  # Skip invalid short skill matches
                
                # Add skill if validation passed
                existing_skills.append(skill)
                existing_skills_set.add(skill_lower)
                logger.info(f"Added skill via word-boundary matching: {skill}")
                # DEBUG: Special logging for suspicious skills
                if is_suspicious:
                    logger.warning(f"[WORD_BOUNDARY_MATCH_DEBUG] ✓✓✓ ADDED skill '{skill}' to existing_skills list (current count: {len(existing_skills)})")
                    logger.warning(f"[WORD_BOUNDARY_MATCH_DEBUG] Match position: {start}-{end}")
                    logger.warning(f"[WORD_BOUNDARY_MATCH_DEBUG] Matched text from resume: '{resume_text_lower[start:end]}'")
                    logger.warning(f"[WORD_BOUNDARY_MATCH_DEBUG] Current skills list: {existing_skills}")
                    # Double-check: verify the match is correct
                    if skill_lower == 'azure communication services':
                        # Check if 'azure' actually exists in the matched text
                        matched_actual = resume_text_lower[start:end]
                        if 'azure' not in matched_actual:
                            logger.error(f"[WORD_BOUNDARY_MATCH_DEBUG] ⚠️⚠️⚠️ CRITICAL BUG: 'azure communication services' matched but 'azure' not in matched text: '{matched_actual}'")
                            logger.error(f"[WORD_BOUNDARY_MATCH_DEBUG] This indicates a pattern matching bug!")
                if max_skills is not None and len(existing_skills) >= max_skills:  # Stop after finding max skills (if limit set)
                    break
        
        return existing_skills
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """
        Extract technical and soft skills using the enhanced skill_extractor module.
        
        This method now uses the production-ready skill_extractor.py which provides:
        - 2000+ technical skills
        - 50+ soft skills
        - Smart section detection
        - Multiple format support (comma-separated, bullets, etc.)
        - Special character handling (C#, C++, .NET)
        - Alias support (JS->JavaScript, ML->Machine Learning)
        - Overlap prevention for multi-word skills
        """
        # Use the comprehensive skill extractor
        result = extract_skills_advanced(text, return_categories=True)
        
        # Convert to the expected format
        tech_skills = result.get('tech_skills', [])
        soft_skills = result.get('soft_skills', [])
        all_skills = result.get('all_skills', [])
        
        logger.info(f"Skill extractor found {len(tech_skills)} technical skills, {len(soft_skills)} soft skills")
        
        # Return in the expected format
        return {
            'primary_skills': tech_skills,  # All technical skills
            'secondary_skills': soft_skills,  # Soft skills
            'all_skills': all_skills  # Combined
        }
    
    def extract_experience(self, text: str) -> float:
        """
        Calculate total professional experience using the comprehensive ExperienceExtractor.
        Falls back to original method if ExperienceExtractor is not available.
        
        Args:
            text: Resume text to parse
            
        Returns:
            Total experience in years (float)
        """
        # Use the new comprehensive ExperienceExtractor if available
        if ExperienceExtractor:
            try:
                # Try Python extraction first (no AI)
                extractor = ExperienceExtractor(text)
                result = extractor.extract()
                total_experience = result.get('total_experience_years', 0.0)
                
                # Note: AI fallback is not implemented in the new ExperienceExtractor
                # The new extractor handles all patterns without needing AI
                
                # Always use the result from ExperienceExtractor (even if 0)
                # It properly handles freshers by returning 0 when no Experience section exists
                logger.info(f"ExperienceExtractor calculated: {total_experience} years")
                logger.debug(f"Experience segments: {result.get('segments', [])}")
                logger.debug(f"Ignored entries: {result.get('ignored', [])}")
                return float(total_experience)
            except Exception as e:
                logger.warning(f"ExperienceExtractor failed: {e}, falling back to original method")
                import traceback
                logger.error(traceback.format_exc())
        
        # Fallback to original method
        return self._extract_experience_legacy(text)
    
    def _extract_experience_legacy(self, text: str) -> float:
        """Legacy experience extraction method (original implementation).
        Calculate total professional experience (full years): prioritize explicit mentions,
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
        CRITICAL: Excludes education dates to avoid counting graduation dates as work experience.
        """
        # Find Experience section if possible - CRITICAL for separating work from education dates
        experience_text = text
        try:
            section_match = re.search(r'(?is)(experience|work experience|professional experience|employment|work history)[\s\n\r:.-]+(.+?)(?=\n\s*[A-Z][A-Za-z ]{2,}:|\n\s*(education|academic|skills|projects|certifications)\b|\Z)', text)
            if section_match and section_match.group(2):
                experience_text = section_match.group(2)
                logger.info("Found Experience section, using it for date extraction")
            else:
                logger.warning("No Experience section found - will try to exclude education dates")
        except Exception as e:
            logger.warning(f"Error finding experience section: {e}")
        
        # Try to exclude Education section dates to avoid counting graduation year as work start
        education_section = None
        try:
            edu_match = re.search(r'(?is)(education|academic|qualification)[\s\n\r:.-]+(.+?)(?=\n\s*[A-Z][A-Za-z ]{2,}:|\n\s*(experience|skills|projects|certifications)\b|\Z)', text)
            if edu_match and edu_match.group(2):
                education_section = edu_match.group(2)
                logger.info("Found Education section - will exclude its dates from experience calculation")
        except Exception:
            pass
        
        # Extract education years to exclude them
        education_years = set()
        if education_section:
            # Directly extract full 4-digit years (1950-2024)
            year_pattern = r'\b(19\d{2}|20\d{2})\b'
            year_matches = re.findall(year_pattern, education_section)
            for year_str in year_matches:
                try:
                    year = int(year_str)
                    if 1950 <= year <= datetime.now().year:
                        education_years.add(year)
                        logger.debug(f"Found education year to exclude: {year}")
                except ValueError:
                    pass
        
        # Look for date patterns like "Jan 2020 - Present" or "2018 - 2020" in experience section
        month_regex = r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*'
        year_regex = r'(?:19|20)\d{2}'
        date_pattern = rf'(?i){month_regex}\s+{year_regex}|{year_regex}'
        dates = re.findall(date_pattern, experience_text)
        
        # Check for Present/Current/Till Date and add current date if found
        has_present = bool(re.search(r'(?i)(present|current|till date|till now)', experience_text))
        
        # Filter out education years from extracted dates
        work_years = []
        for date_str in dates:
            year_match = re.search(r'\d{4}', date_str)
            if year_match:
                year = int(year_match.group())
                # Exclude if it's an education year (likely graduation date)
                if year not in education_years:
                    work_years.append(year)
                else:
                    logger.debug(f"Excluded education year {year} from work experience calculation")
        
        if has_present:
            current_year = datetime.now().year
            work_years.append(current_year)
        
        if len(work_years) >= 2:
            try:
                # Calculate span from earliest work start to latest work end
                current_year = datetime.now().year
                max_year = min(max(work_years), current_year)
                min_year = min(work_years)
                experience = max(0, max_year - min_year)
                
                # For freshers: If experience is very high (e.g., >20 years) and we have education years,
                # it might be miscalculated. Check if min_year is close to education year.
                if experience > 20 and education_years:
                    # Likely miscalculation - check if we're counting from graduation
                    latest_edu_year = max(education_years) if education_years else 0
                    if min_year <= latest_edu_year + 2:
                        logger.warning(f"Experience calculation may be incorrect ({experience} years). Possible education date confusion. Returning 0.")
                        return 0.0
                
                logger.info(f"Calculated experience from work dates: {experience} years (from {min_year} to {max_year})")
                return float(experience)
            except Exception as e:
                logger.warning(f"Error calculating experience from dates: {e}")
        elif len(work_years) == 1 and has_present:
            # Single year + Present means from that year to now
            # But exclude if it's an education year
            year = work_years[0]
            if year in education_years:
                logger.info(f"Only year found ({year}) is in education section - likely a fresher. Returning 0.")
                return 0.0
            current_year = datetime.now().year
            experience = max(0, current_year - year)
            logger.info(f"Calculated experience from single date: {experience} years (from {year} to {current_year})")
            return float(experience)
        elif len(work_years) == 0:
            # No work dates found - likely a fresher
            logger.info("No work experience dates found - returning 0 for fresher")
            return 0.0
        
        return 0.0
    
    def extract_domain(self, text: str) -> Optional[str]:
        """Extract domain/industry."""
        text_lower = text.lower()
        
        # Check if tech skills are present - auto-add "Information Technology"
        tech_skill_keywords = ['python', 'java', 'sql', 'javascript', 'html', 'css', '.net', 'c++', 
                              'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'react', 'angular', 
                              'node.js', 'django', 'flask', 'spring', 'mongodb', 'mysql', 'postgresql']
        has_tech_skills = any(skill in text_lower for skill in tech_skill_keywords)
        
        found_domains = []
        for domain in self.DOMAINS:
            if domain.lower() in text_lower:
                found_domains.append(domain)
        
        # Auto-add "Information Technology" if tech skills found
        if has_tech_skills and "Information Technology" not in found_domains:
            found_domains.insert(0, "Information Technology")
        
        # Return most frequent or first found
        if found_domains:
            return found_domains[0]
        
        # If no domain found but tech skills present, return IT
        if has_tech_skills:
            return "Information Technology"
        
        return None

    def _filter_domain_by_profile_type(self, domain: str, profile_type: Optional[str]) -> str:
        """
        Post-process extracted domain string based on detected profile type.

        Goal:
        - For clearly IT/software profiles (Java, .Net, full stack, etc.), drop
          stray business domains that only come from project context (e.g. Pharmaceuticals,
          Energy) and keep only core IT domains.

        This is a conservative, profile_type-driven filter _on top of_ the
        pure text-based domain extraction in domain_extraction.py.
        """
        if not domain:
            return ''

        if not profile_type:
            return domain

        # Split "Information Technology, Data Science, Energy" -> ["Information Technology", "Data Science", "Energy"]
        domain_parts = [d.strip() for d in domain.split(',') if d.strip()]
        if not domain_parts:
            return ''

        pt_lower = profile_type.lower()

        # Heuristic: treat these profile_type keywords as strongly IT / software engineering.
        it_profile_keywords = [
            'java',
            'python',
            '.net',
            'full stack',
            'frontend',
            'backend',
            'software engineer',
            'developer',
            'devops',
            'sre',
            'cloud',
            'javascript',
            'react',
            'angular',
            'vue',
            'node',
            'nodejs',
        ]

        is_it_profile = any(k in pt_lower for k in it_profile_keywords)

        if is_it_profile:
            # For clear IT profiles, only keep core IT-related domains.
            # This intentionally drops project-only business domains like
            # "Pharmaceuticals" or "Energy" when the candidate is really a Java/.Net dev.
            allowed_it_domains = {
                'Information Technology',
                'Software Development',
                'Cloud Computing',
            }
            filtered = [d for d in domain_parts if d in allowed_it_domains]

            if filtered:
                logger.info(
                    f"Filtered domains based on IT profile_type '{profile_type}': "
                    f"original={domain_parts} -> filtered={filtered}"
                )
                return ', '.join(filtered)
            
            # If nothing matched but we KNOW the profile_type is IT (Python, Java, etc.),
            # fall back to a safe default: label as Information Technology.
            # This prevents wrong domains like "Agriculture" from appearing for IT professionals.
            logger.info(
                f"No IT-specific domains matched for profile_type '{profile_type}'. "
                f"Falling back to 'Information Technology' instead of {domain_parts}."
            )
            return 'Information Technology'

        # Heuristic: treat these profile_type keywords as strongly Business Development / Sales / Marketing.
        bd_profile_keywords = [
            'business development',
            'bd ',
            'sales',
            'inside sales',
            'presales',
            'pre-sales',
            'marketing',
            'lead generation',
        ]

        is_bd_profile = any(k in pt_lower for k in bd_profile_keywords)

        if is_bd_profile:
            # For Business Development / Sales / Marketing profiles, prefer BD/Marketing domains
            # and drop one-off client industries like "Construction", "Healthcare", etc.
            allowed_bd_domains = {
                'Business Development',
                'Advertising & Marketing',
            }
            filtered = [d for d in domain_parts if d in allowed_bd_domains]

            if filtered:
                logger.info(
                    f"Filtered domains based on BD profile_type '{profile_type}': "
                    f"original={domain_parts} -> filtered={filtered}"
                )
                return ', '.join(filtered)

            # If nothing matched but we KNOW the profile_type is Business Development,
            # fall back to a safe default: label as Business Development.
            logger.info(
                f"No BD-specific domains matched for profile_type '{profile_type}'. "
                f"Falling back to 'Business Development' instead of {domain_parts}."
            )
            return 'Business Development'

        # Default: return original domain string
        return ', '.join(domain_parts)
    
    def extract_education(self, text: str, store_in_db: bool = False, candidate_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract education information using the enhanced EducationExtractor.
        
        Args:
            text: Resume text to parse
            store_in_db: If True, store extracted education in database
            candidate_id: Candidate ID for database storage (required if store_in_db=True)
        
        Returns:
            Dictionary with 'highest_degree' and 'education_details'
        """
        # Use the new EducationExtractor if available
        if EducationExtractor:
            try:
                extractor = EducationExtractor(
                    text,
                    store_in_db=store_in_db,
                    candidate_id=candidate_id
                )
                education_list = extractor.extract()
                
                # Convert to the expected format
                education_info = {
                    'highest_degree': None,
                    'education_details': education_list if education_list else []
                }
                
                # Determine highest degree from the list
                if education_list:
                    # Get the first (highest) degree - keep the full extracted string with specialization
                    highest_degree_str = education_list[0]
                    
                    # Use the extracted string as-is to preserve specialization (e.g., "BTech, Civil Engineering")
                    # Only map to generic categories if the extracted string is too generic
                    highest_degree_lower = highest_degree_str.lower()
                    
                    # If the extracted string is just a generic word like "bachelor" or "bachelors", map it
                    # But if it contains specific degree info (like "BTech" or "B.Tech"), keep it as-is
                    if highest_degree_str.strip().lower() in ['bachelor', 'bachelors', 'bachelor degree']:
                        education_info['highest_degree'] = 'Bachelors'
                    elif highest_degree_str.strip().lower() in ['master', 'masters', 'master degree']:
                        education_info['highest_degree'] = 'Masters'
                    elif highest_degree_str.strip().lower() in ['phd', 'doctorate', 'ph.d']:
                        education_info['highest_degree'] = 'PhD'
                    elif highest_degree_str.strip().lower() == 'diploma':
                        education_info['highest_degree'] = 'Diploma'
                    else:
                        # Keep the full extracted string (preserves "BTech, Civil Engineering" or "B.Tech in CSE")
                        education_info['highest_degree'] = highest_degree_str
                
                logger.info(f"EducationExtractor found {len(education_list)} degrees: {education_list}")
                return education_info
                
            except Exception as e:
                logger.warning(f"EducationExtractor failed: {e}, falling back to regex extraction")
        
        # Fallback to original regex-based extraction
        education_info = {
            'highest_degree': None,
            'education_details': []
        }
        
        text_lower = text.lower()
        
        # Find education section
        edu_section_pattern = r'(?i)(?:education|education\s*&\s*certifications|academic|qualification)[:\s]+(.*?)(?=\n\n[A-Z]|experience|skills|$)'
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
        """
        Extract current location using pure Python regex (NO AI/NLP).
        
        Uses location_identifier.py for comprehensive location extraction
        supporting multiple formats: City/State/ZIP, international locations,
        multi-line addresses, and various noisy resume formats.
        """
        # Use the dedicated location extractor (pure Python, no AI/NLP)
        if extract_location_advanced:
            location = extract_location_advanced(text)
            # Return the extracted location if valid
            if location and location != "Unknown":
                return location
        
        # Fallback: simple location extraction if module not available
        # Try to find common location patterns
        import re
        location_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2})\b',  # City, State
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z][a-z]+)\b',  # City, Country
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text)
            if match:
                return f"{match.group(1)}, {match.group(2)}"
        
        return None
        
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
            
            # Extract skills FIRST (before name extraction) for skill-based filtering
            # This prevents tools/skills like "Visual Studio" from being extracted as names
            logger.info("Extracting skills for name filtering...")
            skills_result = self.extract_skills(resume_text)
            extracted_skills_set = set(skills_result.get('all_skills', []))
            logger.info(f"Extracted {len(extracted_skills_set)} skills for name filtering")
            
            # Try comprehensive AI extraction first (with skills for name filtering)
            ai_data = None
            if self.use_ai_extraction:
                ai_data = self.extract_comprehensive_data_with_ai(resume_text, extracted_skills_set)
            
            # Use AI data if available, otherwise fallback to regex-based extraction
            if ai_data:
                # Use AI-extracted comprehensive data
                # First try AI extracted name
                name = ai_data.get('full_name') or ''
                
                # If name is invalid or missing, extract from text
                invalid_names = {'education', 'experience', 'skills', 'contact', 'objective'}
                degree_keywords = ['b.a.', 'm.a.', 'b.s.', 'm.s.', 'phd', 'mba', 'degree']
                
                if not name or name.lower() in ['unknown', 'education', 'experience']:
                    logger.warning(f"AI name extraction failed or returned invalid: '{name}', trying improved name extraction with skill filtering...")
                    # Try improved name extraction with skill filtering (checks 8-10 lines, handles ALL CAPS, filters skills)
                    name = extract_name(resume_text, extracted_skills=extracted_skills_set)
                    logger.info(f"Improved name extraction returned: {name}")
                    
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
                # ALWAYS use Python-based extraction for phone (NO AI) - uses phone_extractor.py
                phone = self.extract_phone(resume_text)
                # ALWAYS use Python-based extraction (NO AI) - uses comprehensive ExperienceExtractor
                # This ensures accurate date parsing, education exclusion, and range merging
                experience = self.extract_experience(resume_text)
                
                # SKILL EXTRACTION: Use ONLY Python-based extraction (NO AI)
                # Uses skill_extractor.py module (2000+ technical skills, 50+ soft skills)
                # Skills already extracted above for name filtering, reuse the result
                logger.info("Using skills extracted earlier for name filtering...")
                python_skills = skills_result
                
                # Get technical and soft skills from Python extraction
                technical_skills_list = python_skills.get('primary_skills', [])  # Technical skills
                secondary_skills_list = python_skills.get('secondary_skills', [])  # Soft skills
                
                logger.info(f"Python skill_extractor found {len(technical_skills_list)} technical skills, {len(secondary_skills_list)} soft skills")
                
                # Collect all valid technical skills
                technical_skills = []
                technical_skills_lower = set()
                
                # Process Python-extracted technical skills
                for skill in technical_skills_list:
                    if not skill or not isinstance(skill, str):
                        continue
                    skill_stripped = skill.strip()
                    skill_lower = skill_stripped.lower()
                    
                    # Filter out single-letter skills
                    if len(skill_stripped) <= 1:
                        logger.debug(f"Skipping single-letter skill: '{skill_stripped}'")
                        continue
                    
                    if skill_lower not in technical_skills_lower:
                        technical_skills.append(skill_stripped)
                        technical_skills_lower.add(skill_lower)
                
                # Process secondary skills
                secondary_skills = []
                for skill in secondary_skills_list:
                    if not skill or not isinstance(skill, str):
                        continue
                    skill_stripped = skill.strip()
                    
                    # Filter out single-letter skills
                    if len(skill_stripped) <= 1:
                        continue
                    
                    if skill_stripped not in secondary_skills:
                        secondary_skills.append(skill_stripped)
                
                # ALWAYS supplement with word-boundary matching to catch any missed skills
                logger.info(f"Supplementing with word-boundary matching from entire resume...")
                # DEBUG: Log resume text info before skill extraction
                logger.debug(f"[RESUME_PARSER_DEBUG] Resume text length: {len(resume_text)} chars")
                logger.debug(f"[RESUME_PARSER_DEBUG] Checking if 'azure' exists in resume_text: {'azure' in resume_text.lower()}")
                if 'azure' not in resume_text.lower():
                    logger.warning(f"[RESUME_PARSER_DEBUG] 'azure' NOT FOUND in resume_text, but may be extracted as skill - this indicates a bug!")
                # DEBUG: Log skills BEFORE word-boundary matching
                logger.info(f"[RESUME_PARSER_DEBUG] Skills BEFORE word-boundary matching: {technical_skills}")
                logger.info(f"[RESUME_PARSER_DEBUG] Skills count BEFORE: {len(technical_skills)}")
                
                technical_skills = self._extract_skills_from_text_with_word_boundaries(
                    resume_text, technical_skills, technical_skills_lower, max_skills=None  # No limit - extract all skills
                )
                
                # DEBUG: Log skills AFTER word-boundary matching
                logger.info(f"[RESUME_PARSER_DEBUG] Skills AFTER word-boundary matching: {technical_skills}")
                logger.info(f"[RESUME_PARSER_DEBUG] Skills count AFTER: {len(technical_skills)}")
                if any('azure' in skill.lower() for skill in technical_skills):
                    logger.warning(f"[RESUME_PARSER_DEBUG] ⚠️  WARNING: Found Azure-related skills: {[s for s in technical_skills if 'azure' in s.lower()]}")
                    logger.warning(f"[RESUME_PARSER_DEBUG] This is suspicious if 'azure' was not in the original resume text!")
                    logger.warning(f"[RESUME_PARSER_DEBUG] Full technical_skills list: {technical_skills}")
                
                # CRITICAL: Post-process to remove invalid short skills (1-2 letters) that may have been added
                # This is a safety net to catch any short skills that bypassed validation
                from skill_extractor import _is_valid_short_skill_match
                resume_text_lower = resume_text.lower()
                filtered_technical_skills = []
                filtered_technical_skills_lower = set()
                
                # Known false-positive patterns to check for
                false_positive_patterns = {
                    'r': ['related', 'resources', 'reports', 'recovery', 'requirements', 'responsibilities', 'research', 'responsible'],
                    'v': ['version', 'visio', 'virtual', 'various', 'value', 'view', 'views', 'visual'],
                    'ti': ['activities', 'motivation', 'automation', 'documentation', 'certification', 'authentication']
                }
                
                for skill in technical_skills:
                    skill_lower = skill.lower().strip()
                    
                    # For short skills (1-2 letters), apply strict validation
                    if len(skill_lower) <= 2:
                        # Additional check: if this skill appears in false-positive patterns, be extra strict
                        if skill_lower in false_positive_patterns:
                            # Check if the skill appears near any false-positive words
                            pattern = r'\b' + re.escape(skill_lower) + r'\b'
                            matches = list(re.finditer(pattern, resume_text_lower))
                            
                            # Check if ANY occurrence is valid AND not near false-positive words
                            is_valid = False
                            found_false_positive = False
                            
                            for match in matches:
                                start, end = match.span()
                                # INCREASED context window to 50 chars to catch more false positives
                                context_start = max(0, start - 50)
                                context_end = min(len(resume_text_lower), end + 50)
                                context = resume_text_lower[context_start:context_end]
                                
                                # Check if context contains false-positive words (more aggressive)
                                has_false_positive_word = any(
                                    fp_word in context for fp_word in false_positive_patterns[skill_lower]
                                )
                                
                                # STRICT: For known false-positive patterns, reject if ANY false-positive word is nearby
                                if has_false_positive_word:
                                    # Definitely a false positive - mark and check next match
                                    found_false_positive = True
                                    logger.debug(f"Post-processing: Found '{skill}' near false-positive word in context: '{context[:60]}'")
                                    continue  # Check next match
                                
                                # Only accept if validation passes AND no false-positive words nearby
                                if _is_valid_short_skill_match(resume_text_lower, start, end, skill_lower):
                                    is_valid = True
                                    found_false_positive = False  # Override if we found a valid match
                                    break
                            
                            # If we found false-positive words in ALL matches OR no valid match, reject the skill
                            if found_false_positive or not is_valid:
                                logger.debug(f"Post-processing: Removed invalid short skill '{skill}' (failed validation or near false-positive)")
                                continue
                        else:
                            # For other short skills, just validate normally
                            pattern = r'\b' + re.escape(skill_lower) + r'\b'
                            matches = list(re.finditer(pattern, resume_text_lower))
                            
                            # Check if ANY occurrence is valid
                            is_valid = False
                            for match in matches:
                                start, end = match.span()
                                if _is_valid_short_skill_match(resume_text_lower, start, end, skill_lower):
                                    is_valid = True
                                    break
                            
                            if not is_valid:
                                logger.debug(f"Post-processing: Removed invalid short skill '{skill}' (failed validation)")
                                continue
                    
                    # Add valid skill
                    if skill_lower not in filtered_technical_skills_lower:
                        filtered_technical_skills.append(skill)
                        filtered_technical_skills_lower.add(skill_lower)
                
                technical_skills = filtered_technical_skills
                technical_skills_lower = filtered_technical_skills_lower
                
                # DEBUG: Log skills before formatting
                logger.info(f"[RESUME_PARSER_DEBUG] Skills BEFORE formatting: {technical_skills}")
                if any('azure' in skill.lower() for skill in technical_skills):
                    logger.warning(f"[RESUME_PARSER_DEBUG] ⚠️  Azure skills found BEFORE formatting: {[s for s in technical_skills if 'azure' in s.lower()]}")
                
                # Format skills - primary_skills should ONLY contain TECHNICAL_SKILLS
                primary_skills = ', '.join(technical_skills) if technical_skills else ''  # All technical skills
                secondary_skills_str = ', '.join(secondary_skills) if secondary_skills else ''  # Non-technical skills
                
                # DEBUG: Log formatted primary_skills
                logger.info(f"[RESUME_PARSER_DEBUG] Formatted primary_skills string: '{primary_skills}'")
                if 'azure' in primary_skills.lower():
                    logger.warning(f"[RESUME_PARSER_DEBUG] ⚠️  'azure' found in formatted primary_skills string!")
                    logger.warning(f"[RESUME_PARSER_DEBUG] Full primary_skills: '{primary_skills}'")
                
                # all_skills = primary_skills + secondary_skills
                all_skills_combined = technical_skills + secondary_skills
                all_skills_str = ', '.join(all_skills_combined) if all_skills_combined else ''
                
                logger.info(f"✓ Primary skills ({len(technical_skills)}): {primary_skills[:80]}...")
                logger.info(f"✓ Secondary skills ({len(secondary_skills)}): {secondary_skills_str[:80]}...")
                logger.info(f"✓ All skills ({len(all_skills_combined)}): {all_skills_str[:80]}...")
                
                logger.info(f"✓ Python-only extraction completed: {len(technical_skills)} technical skills, {len(secondary_skills)} soft skills")
                
                # Get domains using Python-based keyword extraction (no AI)
                domain_list = extract_domains(resume_text)
                
                # Ensure it's a list
                if not isinstance(domain_list, list):
                    domain_list = [domain_list] if domain_list else []
                
                # If no domains found, try fallback method
                if not domain_list:
                    fallback_domain = self.extract_domain(resume_text)
                    if fallback_domain:
                        domain_list = [fallback_domain]
                
                 # If no domains found, try fallback extraction
                if not domain_list:
                    fallback_domain = self.extract_domain(resume_text)
                    if fallback_domain:
                        domain_list = [fallback_domain]
                
                domain = ', '.join(domain_list) if domain_list else ''
                
                # ALWAYS prioritize Python extraction over AI for accuracy
                # Python extraction is more reliable and doesn't add inferred specializations
                logger.info("Extracting education using Python extraction...")
                education_info = self.extract_education(resume_text)
                
                # Use Python extraction if it found valid education
                if education_info['highest_degree'] and education_info['highest_degree'] != 'Unknown':
                    highest_degree = education_info['highest_degree']
                    education_details = '\n'.join(education_info['education_details']) if education_info['education_details'] else highest_degree
                    logger.info(f"Using Python-extracted education: {highest_degree}")
                else:
                    # Fallback to AI extraction only if Python extraction failed
                    ai_education = ai_data.get('education') or ai_data.get('education_details')
                    if isinstance(ai_education, list):
                        ai_education = ai_education[0] if ai_education else None
                    
                    if ai_education and ai_education != 'Unknown':
                        highest_degree = ai_education
                        education_details = ai_education
                        logger.info(f"Python extraction failed, using AI-extracted education: {highest_degree}")
                    else:
                        # Fallback: try to extract from education_details if available
                        if education_info['education_details']:
                            highest_degree = education_info['education_details'][0] if education_info['education_details'] else ''
                            education_details = '\n'.join(education_info['education_details'])
                            logger.info(f"Using education from details: {highest_degree}")
                        else:
                            # Last resort: set to empty string (not None) to avoid null in JSON
                            highest_degree = ''
                            education_details = ''
                            logger.warning("Both Python and AI extraction failed - no education found")
                
                # Get certifications using dedicated AI extraction
                certifications = extract_certifications_with_ai(
                    text=resume_text,
                    ai_client=self.ai_client,
                    ai_model=self.ai_model
                ) if self.use_ai_extraction and self.ai_client else []
                certifications_str = ', '.join(certifications) if isinstance(certifications, list) and certifications else ''
                
                # Get current company using dedicated AI extraction
                current_company = extract_company_with_ai(
                    text=resume_text,
                    ai_client=self.ai_client,
                    ai_model=self.ai_model
                ) if self.use_ai_extraction and self.ai_client else ''
                
                # Get current designation using Python-based extraction (no AI)
                current_designation = extract_designation(resume_text) or ''
                
                # DEBUG: Log designation extraction result
                if current_designation:
                    logger.info(f"✓ Designation extracted: '{current_designation}'")
                else:
                    logger.warning(f"✗ Designation extraction returned empty/None - will use profile_type fallback")
                
                # CRITICAL: Reject "Director Of Technology" for fresher resumes (likely from institution names)
                # This is a safety net in case extract_designation() still returns it
                if current_designation and current_designation.lower() == 'director of technology':
                    # Check if this is a fresher resume (has education but no experience)
                    has_education = bool(re.search(r'\b(education|educational|academic|qualification)\b', resume_text, re.IGNORECASE))
                    has_institution = bool(re.search(r'\b(gates\s+institute\s+of\s+technology|institute\s+of\s+technology)\b', resume_text, re.IGNORECASE))
                    has_experience_section = bool(re.search(r'\b(experience|work history|employment history|professional experience|career development)\b', resume_text, re.IGNORECASE))
                    
                    # If it's a fresher resume, reject "Director Of Technology"
                    if (has_education or has_institution) and not has_experience_section:
                        logger.warning(f"Rejected 'Director Of Technology' designation - likely from institution name in education section (fresher resume)")
                        logger.info(f"DEBUG: Cleared current_designation (was: 'Director Of Technology'), will use profile_type fallback")
                        current_designation = ''  # Clear it so fallback logic will use profile_type
                
                # CRITICAL: Reject "Data Analyst" for Power Platform resumes
                # Power Platform resumes should have Power Platform specific roles, not generic "Data Analyst"
                if current_designation and current_designation.lower() == 'data analyst':
                    # Check if resume contains Power Platform keywords
                    power_platform_keywords = [
                        'power platform', 'power apps', 'power automate', 'power pages', 'power bi',
                        'dynamics 365', 'dataverse', 'canvas app', 'model-driven app', 'power fx',
                        'microsoft power platform', 'ms power platform'
                    ]
                    resume_lower = resume_text.lower()
                    has_power_platform = any(keyword in resume_lower for keyword in power_platform_keywords)
                    
                    if has_power_platform:
                        logger.warning(f"Rejected 'Data Analyst' designation - Power Platform resume should have Power Platform specific role")
                        logger.info(f"DEBUG: Cleared current_designation (was: 'Data Analyst'), will use profile_type fallback")
                        current_designation = ''  # Clear it so fallback logic will use profile_type
                
                # Get summary using dedicated AI extraction
                summary = extract_summary_with_ai(
                    text=resume_text,
                    ai_client=self.ai_client,
                    ai_model=self.ai_model
                ) if self.use_ai_extraction and self.ai_client else ''
                
                # Get additional data
                location = self.extract_location(resume_text)
                
            else:
                # Extract skills FIRST (before name extraction) for skill-based filtering
                # This prevents tools/skills like "Visual Studio" from being extracted as names
                logger.info("Extracting skills for name filtering...")
                skills_result = self.extract_skills(resume_text)
                extracted_skills_set = set(skills_result.get('all_skills', []))
                logger.info(f"Extracted {len(extracted_skills_set)} skills for name filtering")
                
                # Use improved name extraction with skill filtering (checks 8-10 lines, handles ALL CAPS, strips prefixes, filters skills)
                name = extract_name(resume_text, extracted_skills=extracted_skills_set)
                email = self.extract_email(resume_text)
                phone = self.extract_phone(resume_text)
                # Use Python-based extraction (NO AI) - uses comprehensive ExperienceExtractor
                experience = self.extract_experience(resume_text)
                
                # Skills already extracted above for name filtering, reuse the result
                skills = skills_result
                
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
                    if not skill or not isinstance(skill, str):
                        continue
                    skill_stripped = skill.strip()
                    skill_lower = skill_stripped.lower()
                    
                    # Filter out single-letter skills
                    if len(skill_stripped) <= 1:
                        logger.debug(f"Skipping single-letter skill: '{skill_stripped}'")
                        continue
                    
                    # Reject responsibility-like phrases unless they're explicitly in TECH_SKILLS
                    if skill_lower in responsibility_phrases and skill_lower not in TECH_SKILLS:
                        logger.warning(f"Rejected responsibility phrase as skill: '{skill_stripped}'")
                        continue
                    
                    # Check if this skill is in our TECH_SKILLS list (from skill_extractor.py)
                    if skill_lower in TECH_SKILLS:
                        if skill_lower not in technical_skills_set:
                            technical_skills_list.append(skill_stripped)
                            technical_skills_set.add(skill_lower)
                    else:
                        # Try partial match
                        found_match = False
                        for tech_skill in TECH_SKILLS:
                            if skill_lower in tech_skill or tech_skill in skill_lower:
                                if tech_skill not in technical_skills_set:
                                    technical_skills_list.append(tech_skill)
                                    technical_skills_set.add(tech_skill)
                                    found_match = True
                                    break
                        if not found_match:
                            secondary_skills_list.append(skill_stripped)
                
                # ALWAYS supplement with word-boundary matching to catch any missed skills
                logger.info(f"Supplementing with word-boundary matching from entire resume...")
                technical_skills_list = self._extract_skills_from_text_with_word_boundaries(
                    resume_text, technical_skills_list, technical_skills_set, max_skills=None  # No limit - extract all skills
                )
                
                # CRITICAL: Post-process to remove invalid short skills (1-2 letters) that may have been added
                # This is a safety net to catch any short skills that bypassed validation
                from skill_extractor import _is_valid_short_skill_match
                resume_text_lower = resume_text.lower()
                filtered_technical_skills_list = []
                filtered_technical_skills_set = set()
                
                # Known false-positive patterns to check for
                false_positive_patterns = {
                    'r': ['related', 'resources', 'reports', 'recovery', 'requirements', 'responsibilities', 'research', 'responsible'],
                    'v': ['version', 'visio', 'virtual', 'various', 'value', 'view', 'views', 'visual'],
                    'ti': ['activities', 'motivation', 'automation', 'documentation', 'certification', 'authentication']
                }
                
                for skill in technical_skills_list:
                    skill_lower = skill.lower().strip()
                    
                    # For short skills (1-2 letters), apply strict validation
                    if len(skill_lower) <= 2:
                        # Additional check: if this skill appears in false-positive patterns, be extra strict
                        if skill_lower in false_positive_patterns:
                            # Check if the skill appears near any false-positive words
                            pattern = r'\b' + re.escape(skill_lower) + r'\b'
                            matches = list(re.finditer(pattern, resume_text_lower))
                            
                            # Check if ANY occurrence is valid AND not near false-positive words
                            is_valid = False
                            found_false_positive = False
                            
                            for match in matches:
                                start, end = match.span()
                                # INCREASED context window to 50 chars to catch more false positives
                                context_start = max(0, start - 50)
                                context_end = min(len(resume_text_lower), end + 50)
                                context = resume_text_lower[context_start:context_end]
                                
                                # Check if context contains false-positive words (more aggressive)
                                has_false_positive_word = any(
                                    fp_word in context for fp_word in false_positive_patterns[skill_lower]
                                )
                                
                                # STRICT: For known false-positive patterns, reject if ANY false-positive word is nearby
                                if has_false_positive_word:
                                    # Definitely a false positive - mark and check next match
                                    found_false_positive = True
                                    logger.debug(f"Post-processing: Found '{skill}' near false-positive word in context: '{context[:60]}'")
                                    continue  # Check next match
                                
                                # Only accept if validation passes AND no false-positive words nearby
                                if _is_valid_short_skill_match(resume_text_lower, start, end, skill_lower):
                                    is_valid = True
                                    found_false_positive = False  # Override if we found a valid match
                                    break
                            
                            # If we found false-positive words in ALL matches OR no valid match, reject the skill
                            if found_false_positive or not is_valid:
                                logger.debug(f"Post-processing: Removed invalid short skill '{skill}' (failed validation or near false-positive)")
                                continue
                        else:
                            # For other short skills, just validate normally
                            pattern = r'\b' + re.escape(skill_lower) + r'\b'
                            matches = list(re.finditer(pattern, resume_text_lower))
                            
                            # Check if ANY occurrence is valid
                            is_valid = False
                            for match in matches:
                                start, end = match.span()
                                if _is_valid_short_skill_match(resume_text_lower, start, end, skill_lower):
                                    is_valid = True
                                    break
                            
                            if not is_valid:
                                logger.debug(f"Post-processing: Removed invalid short skill '{skill}' (failed validation)")
                                continue
                    
                    # Add valid skill
                    if skill_lower not in filtered_technical_skills_set:
                        filtered_technical_skills_list.append(skill)
                        filtered_technical_skills_set.add(skill_lower)
                
                technical_skills_list = filtered_technical_skills_list
                technical_skills_set = filtered_technical_skills_set
                
                # DEBUG: Log skills before formatting (fallback path)
                logger.info(f"[RESUME_PARSER_DEBUG] [FALLBACK PATH] Skills BEFORE formatting: {technical_skills_list}")
                if any('azure' in skill.lower() for skill in technical_skills_list):
                    logger.warning(f"[RESUME_PARSER_DEBUG] [FALLBACK PATH] ⚠️  Azure skills found BEFORE formatting: {[s for s in technical_skills_list if 'azure' in s.lower()]}")
                
                # Format primary_skills after potential lenient extraction
                primary_skills = ', '.join(technical_skills_list) if technical_skills_list else ''  # All technical skills
                secondary_skills_str = ', '.join(secondary_skills_list) if secondary_skills_list else ''
                
                # DEBUG: Log formatted primary_skills (fallback path)
                logger.info(f"[RESUME_PARSER_DEBUG] [FALLBACK PATH] Formatted primary_skills string: '{primary_skills}'")
                if 'azure' in primary_skills.lower():
                    logger.warning(f"[RESUME_PARSER_DEBUG] [FALLBACK PATH] ⚠️  'azure' found in formatted primary_skills string!")
                    logger.warning(f"[RESUME_PARSER_DEBUG] [FALLBACK PATH] Full primary_skills: '{primary_skills}'")
                
                # all_skills = primary_skills + secondary_skills (combine lists, then join)
                all_skills_combined = technical_skills_list + secondary_skills_list
                all_skills_str = ', '.join(all_skills_combined) if all_skills_combined else ''
                
                logger.info(f"✓ Primary skills ({len(technical_skills_list)}): {primary_skills[:80]}...")
                logger.info(f"✓ Secondary skills ({len(secondary_skills_list)}): {secondary_skills_str[:80]}...")
                logger.info(f"✓ All skills ({len(all_skills_combined)}): {all_skills_str[:80]}...")
                
                logger.info(f"✓ Regex extraction completed: {len(technical_skills_list)} technical skills")
                
                # Extract domain - auto-add IT if tech skills present
                domain = self.extract_domain(resume_text)
                if not domain and technical_skills_list:
                    domain = "Information Technology"
                    logger.info("✓ Auto-added 'Information Technology' domain based on technical skills")
                elif technical_skills_list and domain != "Information Technology":
                    # Ensure IT is included if tech skills found
                    domain_parts = [d.strip() for d in domain.split(',')] if domain else []
                    if "Information Technology" not in domain_parts:
                        domain_parts.insert(0, "Information Technology")
                        domain = ', '.join(domain_parts)
                        logger.info("✓ Auto-added 'Information Technology' domain based on technical skills")
                # Use enhanced EducationExtractor (Python extraction only)
                education_info = self.extract_education(resume_text)
                
                # Handle None case and provide fallback
                if education_info['highest_degree'] and education_info['highest_degree'] != 'Unknown':
                    highest_degree = education_info['highest_degree']
                    education_details = '\n'.join(education_info['education_details']) if education_info['education_details'] else highest_degree
                    logger.info(f"Using Python-extracted education: {highest_degree}")
                else:
                    # Fallback: try to extract from education_details if available
                    if education_info['education_details']:
                        highest_degree = education_info['education_details'][0] if education_info['education_details'] else 'Unknown'
                        education_details = '\n'.join(education_info['education_details'])
                        logger.info(f"Using education from details: {highest_degree}")
                    else:
                        # Last resort: set to empty string (not None) to avoid null in JSON
                        highest_degree = ''
                        education_details = ''
                        logger.warning("Education extraction failed - no education found")
                
                # Extract current company and designation
                current_company = self._extract_current_company(resume_text)
                # Use improved designation extraction (same as AI path)
                current_designation = extract_designation(resume_text) or ''
                
                # DEBUG: Log designation extraction result
                if current_designation:
                    logger.info(f"✓ Designation extracted (non-AI path): '{current_designation}'")
                else:
                    logger.warning(f"✗ Designation extraction returned empty/None (non-AI path) - will use profile_type fallback")
                
                # CRITICAL: Reject "Director Of Technology" for fresher resumes (likely from institution names)
                # This is a safety net in case extract_designation() still returns it
                if current_designation and current_designation.lower() == 'director of technology':
                    # Check if this is a fresher resume (has education but no experience)
                    has_education = bool(re.search(r'\b(education|educational|academic|qualification)\b', resume_text, re.IGNORECASE))
                    has_institution = bool(re.search(r'\b(gates\s+institute\s+of\s+technology|institute\s+of\s+technology)\b', resume_text, re.IGNORECASE))
                    has_experience_section = bool(re.search(r'\b(experience|work history|employment history|professional experience|career development)\b', resume_text, re.IGNORECASE))
                    
                    # If it's a fresher resume, reject "Director Of Technology"
                    if (has_education or has_institution) and not has_experience_section:
                        logger.warning(f"Rejected 'Director Of Technology' designation - likely from institution name in education section (fresher resume)")
                        logger.info(f"DEBUG: Cleared current_designation (was: 'Director Of Technology'), will use profile_type fallback")
                        current_designation = ''  # Clear it so fallback logic will use profile_type
                certifications_str = ''
                summary = ''
                location = self.extract_location(resume_text)
            
            # Get file info
            file_size_kb = os.path.getsize(file_path) / 1024 if os.path.exists(file_path) else 0
            
            # Derive canonical profile type using enhanced detection (supports multi-profile)
            profile_types, confidence, metadata = determine_profile_types_enhanced(
                primary_skills, 
                secondary_skills_str, 
                resume_text,
                ai_client=self.ai_client if self.use_ai_extraction else None,
                ai_model=self.ai_model if self.use_ai_extraction else None
            )
            profile_type = format_profile_types_for_storage(profile_types)
            
            # SIMPLIFIED DESIGNATION IDENTIFICATION:
            # 1. Use extracted designation if available
            # 2. If designation is empty, derive from profile_type
            # This is a cleaner approach: designation OR profile_type based
            
            def derive_designation_from_profile_type(profile_type: str, primary_skills: str, secondary_skills: str) -> tuple:
                """
                Derive designation from profile_type.
                Returns (designation, role_type, subrole_type)
                """
                # Profile type to designation mapping
                profile_type_designation_map = {
                    "Python": "Python Developer",
                    "Java": "Java Developer",
                    ".Net": ".Net Developer",
                    "JavaScript": "JavaScript Developer",
                    "Full Stack": "Full Stack Developer",
                    "DevOps": "DevOps Engineer",
                    "Data Science": "Data Scientist",
                    "Data Engineering": "Data Engineer",
                    "Salesforce": "Salesforce Developer",
                    "SAP": "SAP Consultant",
                    "Cloud / Infra": "Cloud Engineer",
                    "Testing / QA": "QA Engineer",
                    "Mobile Development": "Mobile Developer",
                    "RPA": "RPA Developer",
                    "Cyber Security": "Security Engineer",
                    "Microsoft Power Platform": "Power Platform Developer",
                    "Business Development": "Business Development Executive",
                }
                
                # Handle comma-separated profile types (take first one)
                primary_profile_type = profile_type.split(',')[0].strip() if ',' in profile_type else profile_type
                
                # Get base designation from profile type
                # CRITICAL: Return None instead of defaulting to "Software Engineer" for unknown profile types
                # This prevents incorrect designations like "Software Engineer" for non-tech resumes (e.g., Security Officer)
                base_designation = profile_type_designation_map.get(primary_profile_type, None)
                
                # If no mapping found, return None (don't infer a false designation)
                if base_designation is None:
                    logger.info(f"DEBUG: No designation mapping found for profile_type '{primary_profile_type}' - returning None instead of defaulting to 'Software Engineer'")
                    return None, None, None
                
                # Determine subrole from skills if available
                subrole = None
                if primary_skills:
                    from role_extract import determine_subrole_type_from_profile_and_skills
                    try:
                        subrole = determine_subrole_type_from_profile_and_skills(
                            primary_profile_type, primary_skills, secondary_skills
                        )
                    except Exception as e:
                        logger.debug(f"Error determining subrole from profile: {e}")
                
                # Build final designation - use only base_designation (no subrole in parentheses)
                final_designation = base_designation
                
                # Normalize role_type
                role_type_normalized = None
                try:
                    from role_processor import normalize_role
                    role_type_normalized = normalize_role(
                        original_role=base_designation,
                        resume_text=resume_text,
                        primary_skills=primary_skills,
                        secondary_skills=secondary_skills
                    )
                    if not role_type_normalized or role_type_normalized == "Others":
                        role_type_normalized = base_designation
                except Exception as e:
                    logger.debug(f"Error normalizing role from profile_type: {e}")
                    role_type_normalized = base_designation
                
                return final_designation, role_type_normalized, subrole
            
            # Extract role and subrole:
            # 1. Find ALL roles from resume text (designation + experience)
            # 2. Select highest based on ROLE_HIERARCHY
            # 3. Match subrole based on skills
            role_type = None
            subrole_type = None
            
            # Check if this is a non-IT profile BEFORE trying to detect roles
            from role_extract import is_non_it_profile
            is_non_it = profile_type and is_non_it_profile(profile_type)
            
            # SIMPLIFIED DESIGNATION IDENTIFICATION:
            # 1. Use extracted designation if available (already extracted at line 1702)
            # 2. If designation is empty, derive from profile_type
            # No complex role detection or normalization from designation
            
            # Initialize role_type and subrole_type (will be set from profile_type if needed)
            role_type = None
            subrole_type = None
            
            # FINAL CHECK: If designation is still empty, use profile_type to derive it
            # This ensures designation is always set (either from extraction OR profile_type)
            if (not current_designation or current_designation.strip() == '') and profile_type:
                logger.info(f"DEBUG: Designation is empty after all processing (current_designation='{current_designation}'), deriving from profile_type: '{profile_type}'")
                logger.info(f"DEBUG: This means extract_designation() did not find a designation in the resume text")
                derived_designation, derived_role_type, derived_subrole = derive_designation_from_profile_type(
                    profile_type, primary_skills, secondary_skills_str
                )
                logger.info(f"DEBUG: Derived designation from profile_type: '{derived_designation}' (role_type: '{derived_role_type}', subrole: '{derived_subrole}')")
                # Only set designation if it was successfully derived (not None)
                if derived_designation:
                    current_designation = derived_designation
                    if not role_type:
                        role_type = derived_role_type
                    if not subrole_type:
                        subrole_type = derived_subrole
                    logger.info(f"✓ Final fallback: Set current_designation from profile_type: '{current_designation}'")
                else:
                    logger.info(f"✗ Could not derive designation from profile_type '{profile_type}' - designation will remain empty")
            else:
                if current_designation:
                    logger.info(f"✓ Using extracted designation (not using profile_type fallback): '{current_designation}'")
                    # Extract role_type from designation using role_processor table
                    try:
                        from role_processor import normalize_role
                        role_type = normalize_role(
                            original_role=current_designation,
                            resume_text=resume_text,
                            primary_skills=primary_skills,
                            secondary_skills=secondary_skills_str
                        )
                        if role_type:
                            logger.info(f"✓ Extracted role_type='{role_type}' from designation '{current_designation}' using role_processor table")
                        else:
                            logger.warning(f"✗ normalize_role() returned None for designation '{current_designation}'")
                    except Exception as e:
                        logger.error(f"Error extracting role_type from designation '{current_designation}': {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                else:
                    logger.warning(f"✗ Designation is empty and profile_type is also empty - designation will remain empty")

            # Override subrole_type so it is always based on profile_type + skills
            # But only for IT-related profiles
            from role_extract import determine_subrole_type_from_profile_and_skills, is_non_it_profile
            if profile_type and is_non_it_profile(profile_type):
                # For non-IT profiles, set subrole_type to None (not applicable)
                subrole_type = None
                logger.info(f"Non-IT profile '{profile_type}': subrole_type set to None")
            else:
                subrole_type = determine_subrole_type_from_profile_and_skills(
                    profile_type, primary_skills, secondary_skills_str
                )
                logger.info(
                    f"✓ Determined subrole_type='{subrole_type}' based on profile_type='{profile_type}' and skills"
                )
            
            # Final log
            if role_type:
                logger.info(f"✓ Final result: role_type='{role_type}', subrole_type='{subrole_type}'")
            
            
            # Calculate sub_profile_type from second highest score
            sub_profile_type = None
            try:
                profile_scores = get_all_profile_type_scores(
                    primary_skills=primary_skills,
                    secondary_skills=secondary_skills_str,
                    resume_text=resume_text
                )
                if profile_scores:
                    # Sort profile types by score (descending)
                    sorted_profiles = sorted(profile_scores.items(), key=lambda x: x[1], reverse=True)
                    # Filter out zero scores
                    non_zero_profiles = [(pt, score) for pt, score in sorted_profiles if score > 0]
                    
                    if len(non_zero_profiles) >= 2:
                        # Get second highest profile type
                        sub_profile_type = non_zero_profiles[1][0]
                        logger.info(f"Set sub_profile_type to second highest score: {sub_profile_type} (score: {non_zero_profiles[1][1]})")
                    else:
                        logger.info(f"Only {len(non_zero_profiles)} profile type(s) with non-zero scores, sub_profile_type set to None")
           
            except Exception as e:
                logger.warning(f"Failed to calculate sub_profile_type: {e}")
            
            # Apply profile-type-aware domain filtering (e.g., for Java/.Net drop stray pharma domains)
            domain = self._filter_domain_by_profile_type(domain, profile_type)

            # DEBUG: Log primary_skills before creating parsed_data
            logger.info(f"[RESUME_PARSER_DEBUG] [FINAL] primary_skills before parsed_data: '{primary_skills}'")
            if 'azure' in primary_skills.lower():
                logger.warning(f"[RESUME_PARSER_DEBUG] [FINAL] ⚠️  'azure' found in primary_skills before parsed_data!")
                logger.warning(f"[RESUME_PARSER_DEBUG] [FINAL] Full primary_skills: '{primary_skills}'")
                logger.warning(f"[RESUME_PARSER_DEBUG] [FINAL] primary_skills type: {type(primary_skills)}, length: {len(primary_skills)}")
            
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
                'education': highest_degree if highest_degree is not None else '',
                'education_details': education_details if education_details is not None else '',
                'current_location': location,
                'current_company': current_company,
                'current_designation': current_designation,
                'certifications': certifications_str,
                'resume_summary': summary,
                'resume_text': resume_text,
                'file_name': os.path.basename(file_path),
                'file_type': file_type,
                'file_size_kb': int(file_size_kb),
                'ai_extraction_used': ai_data is not None,
                'profile_type': profile_type,
                'role_type': role_type,
                'subrole_type': subrole_type,
                'sub_profile_type': sub_profile_type
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
    """
    Standalone function to extract skills from any text.
    Uses the enhanced skill_extractor module for comprehensive extraction.
    """
    result = extract_skills_advanced(text, return_categories=False)
    return result.get('all_skills', [])


def extract_experience_from_text(text: str) -> float:
    """Standalone function to extract experience from any text."""
    parser = ResumeParser()
    return parser.extract_experience(text)

