"""
Location Identifier - Pure Python Logic (No AI/LLMs)

Extracts and identifies location details from resume text using deterministic regex 
and rule-based parsing. Supports multiple formats including addresses, city-state-zip,
international locations, and noisy resume formats.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LocationMatch:
    """Represents a location match with confidence score."""
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    zip_code: Optional[str] = None
    raw_text: str = ""
    confidence: int = 0
    pattern_type: str = ""
    
    def __str__(self):
        """Format location as readable string."""
        parts = []
        if self.city:
            parts.append(self.city)
        if self.state:
            parts.append(self.state)
        if self.country and self.country not in ['USA', 'United States']:
            parts.append(self.country)
        return ', '.join(parts) if parts else "Unknown"


class LocationExtractor:
    """Extract location information from resume text using rule-based patterns."""
    
    # US States dictionary (abbreviation -> full name)
    US_STATES = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
        'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
        'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
        'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
        'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
        'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
        'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
        'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
        'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
        'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
        'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
    }
    
    # Indian States (full name -> abbreviation)
    INDIA_STATES = {
        'Andhra Pradesh': 'AP', 'Arunachal Pradesh': 'AR', 'Assam': 'AS', 'Bihar': 'BR',
        'Chhattisgarh': 'CG', 'Goa': 'GA', 'Gujarat': 'GJ', 'Haryana': 'HR',
        'Himachal Pradesh': 'HP', 'Jharkhand': 'JH', 'Karnataka': 'KA', 'Kerala': 'KL',
        'Madhya Pradesh': 'MP', 'Maharashtra': 'MH', 'Manipur': 'MN', 'Meghalaya': 'ML',
        'Mizoram': 'MZ', 'Nagaland': 'NL', 'Odisha': 'OD', 'Punjab': 'PB',
        'Rajasthan': 'RJ', 'Sikkim': 'SK', 'Tamil Nadu': 'TN', 'Telangana': 'TS',
        'Tripura': 'TR', 'Uttar Pradesh': 'UP', 'Uttarakhand': 'UK', 'West Bengal': 'WB',
        'Delhi': 'DL', 'Puducherry': 'PY', 'Jammu and Kashmir': 'JK', 'Ladakh': 'LA'
    }
    
    # Countries
    COUNTRIES = {
        'India', 'USA', 'United States', 'US', 'United Kingdom', 'UK', 'Canada',
        'Australia', 'Germany', 'France', 'Singapore', 'China', 'Japan', 'Brazil',
        'Mexico', 'Spain', 'Italy', 'Netherlands', 'Sweden', 'Switzerland', 'Ireland',
        'New Zealand', 'South Africa', 'UAE', 'Saudi Arabia', 'Israel', 'Malaysia',
        'Philippines', 'Thailand', 'Vietnam', 'Indonesia', 'Pakistan', 'Bangladesh',
        'Sri Lanka', 'Nepal', 'Poland', 'Russia', 'Argentina', 'Chile', 'Colombia'
    }
    
    # Major cities (helps with validation)
    MAJOR_CITIES_US = {
        'Portland', 'Seattle', 'San Francisco', 'Los Angeles', 'San Diego', 'Phoenix',
        'Denver', 'Dallas', 'Houston', 'Austin', 'Chicago', 'Minneapolis', 'Detroit',
        'Boston', 'New York', 'Philadelphia', 'Washington', 'Atlanta', 'Miami', 'Orlando'
    }
    
    MAJOR_CITIES_INDIA = {
        'Hyderabad', 'Bangalore', 'Bengaluru', 'Mumbai', 'Delhi', 'Chennai', 'Kolkata',
        'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 'Visakhapatnam',
        'Vijayawada', 'Guntur', 'Warangal', 'Tirupati', 'Nellore', 'Kurnool'
    }
    
    # Software/Tool names that should NOT be treated as locations
    TOOL_BLACKLIST = {
        # Microsoft Power Platform
        'power apps', 'powerapps', 'power automate', 'power bi', 'power pages',
        'power platform', 'copilot studio', 'microsoft copilot studio',
        'dataverse', 'sharepoint', 'dynamics 365', 'dynamics',
        # Microsoft Office
        'office', 'excel', 'word', 'powerpoint', 'outlook', 'teams', 'onenote',
        'access', 'publisher', 'visio', 'project',
        # Cloud/Database
        'azure', 'sql server', 'cosmos db', 'blob storage',
        'business central', 'nav',
        # Development tools
        'visual studio', 'vs code', 'github', 'azure devops',
        # Other tools
        'canvas app', 'model-driven app', 'flow', 'logic apps'
    }
    
    def __init__(self):
        """Initialize location extractor with compiled patterns."""
        self._compile_patterns()
        
        # Create reverse lookup for state names
        self.STATE_NAME_TO_ABBR = {v.lower(): k for k, v in self.US_STATES.items()}
        self.INDIA_STATE_NAMES = {name.lower(): name for name in self.INDIA_STATES.keys()}
        self.COUNTRY_NAMES_LOWER = {c.lower(): c for c in self.COUNTRIES}
    
    def _compile_patterns(self):
        """Compile all regex patterns for location extraction."""
        # Pattern 1: City, State ZIP (e.g., Portland, OR 97124)
        # Limit city to 1-3 words to prevent over-matching
        self.pattern_city_state_zip = re.compile(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}),\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)',
            re.MULTILINE
        )
        
        # Pattern 2: City, State (e.g., Portland, Oregon or Portland, OR)
        # Limit city to 1-3 words to prevent over-matching
        self.pattern_city_state = re.compile(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}),\s*([A-Z]{2}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})(?:\s|$|\.|\n|,)',
            re.MULTILINE
        )
        
        # Pattern 2b: City State (no comma, e.g., Beaverton Oregon)
        self.pattern_city_state_no_comma = re.compile(
            r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            re.MULTILINE
        )
        
        # Pattern 3: City, Country (e.g., Hyderabad, India)
        # Limit city to 1-3 words to prevent over-matching
        self.pattern_city_country = re.compile(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}),\s*(India|USA|US|UK|United States|United Kingdom|Canada|Australia|Singapore|Germany|France|UAE)',
            re.IGNORECASE
        )
        
        # Pattern 4: Location: prefix
        # Limit to 1-4 words with better word boundaries
        self.pattern_location_prefix = re.compile(
            r'(?:Location|Current Location|Address|City|Residence|Based in|Located in)\s*:?\s*([A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+){0,3})',
            re.IGNORECASE
        )
        
        # Pattern 5: Parenthetical location (e.g., (Portland, OR) or (Fort Atkinson, WI/Remote))
        # Limit city to 1-3 words
        self.pattern_parenthetical = re.compile(
            r'\(([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}),\s*([A-Z]{2})(?:/Remote|/Hybrid)?\)',
            re.MULTILINE
        )
        
        # Pattern 6: Multi-line address with ZIP and State
        # Limit city and state to 1-3 words to prevent over-matching
        self.pattern_multiline_address = re.compile(
            r'(?:D\.?NO|#)?\s*[\d\-/]+[,\s]+[\w\s]+(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln)[,\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})[,\s]+(\d{5,6})[,\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',
            re.MULTILINE | re.IGNORECASE
        )
        
        # Pattern 7: Street address format (e.g., 1130 Bibbs Road, Voorhees, New Jersey 08043)
        # Limit city and state to 1-3 words
        self.pattern_street_address = re.compile(
            r'\d+\s+[\w\s]+(?:Road|Rd|Street|St|Avenue|Ave|Lane|Ln|Drive|Dr|Court|Ct|Boulevard|Blvd)(?:,\s*(?:Apt#?|Unit|Suite|#)[\w\-\s]+)?,\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}),\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(\d{5})',
            re.IGNORECASE
        )
        
        # Pattern 8: Bullet-constrained (e.g., ● Portland, OR ●)
        # Limit city to 1-3 words
        self.pattern_bullet = re.compile(
            r'[●•▪]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}),\s*([A-Z]{2})\s*[●•▪]',
            re.MULTILINE
        )
        
        # Pattern 9: Company location format (e.g., "Company Pvt Ltd, Hyderabad, India")
        # Match company names that end with corporate suffixes, followed by city, country
        # Limit city to 1-3 words
        self.pattern_company_location = re.compile(
            r'(?:^|[\n\r])\s*([A-Z][\w\s&]+?)\s+(?:Pvt\.?\s+Ltd\.?|Private\s+Limited|Inc\.?|Corp\.?|LLC|Ltd\.?|Limited),\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}),\s+(India|USA|US|UK|United States|United Kingdom|Canada)',
            re.MULTILINE | re.IGNORECASE
        )
        
        # Pattern 10: Company with city-only (no country) - common in Indian resumes
        # e.g., "RAMA corporate and IT solutions, Hyderabad"
        # More flexible pattern that allows "and X" between company name and city
        self.pattern_company_city_only = re.compile(
            r'(?:^|[\n\r])\s*([A-Z][\w\s&]+?(?:corporate|solutions|technologies|systems|software|services|healthcare|pvt|ltd|limited|inc|corp)[^,]*),\s+([A-Z][a-z]+)\b',
            re.MULTILINE | re.IGNORECASE
        )
    
    def _extract_top_3_lines(self, text: str) -> str:
        """
        Extract first 3 non-empty lines from resume text.
        
        Args:
            text: Raw resume text
            
        Returns:
            String containing first 3 non-empty lines
        """
        if not text:
            return ""
        
        lines = text.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        # Get first 3 non-empty lines
        top_3 = non_empty_lines[:3] if len(non_empty_lines) >= 3 else non_empty_lines
        
        return '\n'.join(top_3)
    
    def _extract_location_from_header(self, header_text: str) -> str:
        """
        Try to extract location from resume header (top 3 lines).
        
        Applies simple patterns suitable for contact headers:
        - City, State ZIP
        - City, State
        - City, Country
        - Location: City
        
        Does NOT apply:
        - Company patterns
        - Client location filtering (not needed in header)
        
        Args:
            header_text: First 3 lines of resume
            
        Returns:
            Formatted location string or empty string if not found
        """
        if not header_text:
            return ""
        
        # Clean text
        cleaned_text = self._preprocess_text(header_text)
        
        # Extract candidates using header-appropriate patterns only
        candidates = []
        
        # Apply simple, high-confidence patterns
        candidates.extend(self._extract_location_prefix(cleaned_text))  # "Location: City"
        candidates.extend(self._extract_city_state_zip(cleaned_text))   # "Portland, OR 97124"
        candidates.extend(self._extract_city_state(cleaned_text))       # "Portland, Oregon"
        candidates.extend(self._extract_city_country(cleaned_text))     # "Hyderabad, India"
        candidates.extend(self._extract_parenthetical(cleaned_text))    # "(Portland, OR)"
        candidates.extend(self._extract_single_city(cleaned_text))      # "Hyderabad" (major cities only)
        
        # Filter valid candidates (no client filtering needed in header)
        valid_candidates = [c for c in candidates if self._is_valid_location(c)]
        
        if not valid_candidates:
            return ""
        
        # Score and rank
        for candidate in valid_candidates:
            candidate.confidence = self._score_location(candidate)
        
        valid_candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        # Return the best match
        best = valid_candidates[0]
        return str(best)
    
    def extract_location(self, text: str) -> str:
        """
        Extract the best location from resume text.
        
        PRIORITY LOGIC:
        1. Check top 3 lines first (contact header)
        2. If found → Use it immediately (most reliable)
        3. If not found → Proceed with full document extraction
        
        Enhanced with context-aware filtering to distinguish between:
        - Candidate's actual location (where they live/work)
        - Client locations (remote/offshore projects)
        - Company locations (employer's base)
        
        Args:
            text: Raw resume text
            
        Returns:
            Formatted location string (e.g., "Portland, Oregon, USA")
            or "Unknown" if no confident match found
        """
        if not text:
            return "Unknown"
        
        # STEP 1: Priority check - Top 3 lines first
        top_3_lines = self._extract_top_3_lines(text)
        header_location = self._extract_location_from_header(top_3_lines)
        
        if header_location and header_location != "Unknown":
            # Found in header! Use this (highest priority)
            return header_location
        
        # STEP 2: Not found in header, proceed with full document extraction
        
        # Clean text
        cleaned_text = self._preprocess_text(text)
        
        # Extract all potential locations
        candidates = []
        
        # Try all patterns (order matters - most specific first)
        candidates.extend(self._extract_city_state_zip(cleaned_text))
        candidates.extend(self._extract_street_address(cleaned_text))
        candidates.extend(self._extract_location_prefix(cleaned_text))  # Prioritize explicit location fields
        candidates.extend(self._extract_company_location(cleaned_text))  # Company locations (high priority)
        candidates.extend(self._extract_company_city_only(cleaned_text))  # Company with city only
        candidates.extend(self._extract_city_state(cleaned_text))
        candidates.extend(self._extract_city_state_no_comma(cleaned_text))
        candidates.extend(self._extract_city_country(cleaned_text))
        candidates.extend(self._extract_parenthetical(cleaned_text))
        candidates.extend(self._extract_multiline_address(cleaned_text))
        candidates.extend(self._extract_bullet_format(cleaned_text))
        candidates.extend(self._extract_single_city(cleaned_text))  # Single major cities
        
        # Filter invalid candidates and client locations
        valid_candidates = []
        for c in candidates:
            if self._is_valid_location(c) and not self._is_client_location(c, cleaned_text):
                valid_candidates.append(c)
        
        if not valid_candidates:
            return "Unknown"
        
        # Apply frequency analysis and context-aware scoring
        valid_candidates = self._apply_frequency_boost(valid_candidates, cleaned_text)
        
        # Score and rank candidates
        for candidate in valid_candidates:
            candidate.confidence = self._score_location(candidate)
        
        # Sort by confidence (highest first)
        valid_candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        # Return best match if confidence is sufficient
        best_match = valid_candidates[0]
        if best_match.confidence >= 50:  # Minimum confidence threshold
            return str(best_match)
        
        return "Unknown"
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove phone numbers (simple pattern)
        text = re.sub(r'\+?\d[\d\s\-\(\)]{7,}\d', '', text)
        
        # Remove bullet characters (>, •, ●, ▪, -, *)
        text = re.sub(r'^[\s>•●▪\-\*]+', '', text, flags=re.MULTILINE)
        
        # Fix "MS" being interpreted as Mississippi
        # Replace "MS Office", "MS Teams", etc. with full word to avoid state confusion
        text = re.sub(r'\bMS\s+Office\b', 'Microsoft Office', text, flags=re.IGNORECASE)
        text = re.sub(r'\bMS\s+Teams\b', 'Microsoft Teams', text, flags=re.IGNORECASE)
        text = re.sub(r'\bMS\s+Excel\b', 'Microsoft Excel', text, flags=re.IGNORECASE)
        text = re.sub(r'\bMS\s+Word\b', 'Microsoft Word', text, flags=re.IGNORECASE)
        text = re.sub(r'\bMS\s+PowerPoint\b', 'Microsoft PowerPoint', text, flags=re.IGNORECASE)
        
        # Fix common PDF extraction artifacts in city names
        # "San Franc isco" -> "San Francisco"
        text = re.sub(r'San\s+Franc\s*isco', 'San Francisco', text, flags=re.IGNORECASE)
        text = re.sub(r'Los\s+Angel\s*es', 'Los Angeles', text, flags=re.IGNORECASE)
        text = re.sub(r'New\s+York\s+', 'New York ', text, flags=re.IGNORECASE)
        
        # Normalize multiple spaces to single space
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _extract_city_state_zip(self, text: str) -> List[LocationMatch]:
        """Extract City, State ZIP format (highest confidence)."""
        matches = []
        for match in self.pattern_city_state_zip.finditer(text):
            city, state_abbr, zip_code = match.groups()
            
            # Validate state abbreviation
            if state_abbr in self.US_STATES:
                matches.append(LocationMatch(
                    city=city.strip().title(),
                    state=self.US_STATES[state_abbr],
                    country='USA',
                    zip_code=zip_code,
                    raw_text=match.group(0),
                    pattern_type='city_state_zip'
                ))
        
        return matches
    
    def _extract_city_state(self, text: str) -> List[LocationMatch]:
        """Extract City, State format."""
        matches = []
        for match in self.pattern_city_state.finditer(text):
            city, state_part = match.groups()
            city = city.strip()
            state_part = state_part.strip()
            
            # Check if state is abbreviation
            if len(state_part) == 2 and state_part in self.US_STATES:
                matches.append(LocationMatch(
                    city=city.title(),
                    state=self.US_STATES[state_part],
                    country='USA',
                    raw_text=match.group(0),
                    pattern_type='city_state'
                ))
            # Check if state is full name
            elif state_part.lower() in self.STATE_NAME_TO_ABBR:
                matches.append(LocationMatch(
                    city=city.title(),
                    state=state_part.title(),
                    country='USA',
                    raw_text=match.group(0),
                    pattern_type='city_state'
                ))
            # Check if it's an Indian state
            elif state_part.lower() in self.INDIA_STATE_NAMES:
                matches.append(LocationMatch(
                    city=city.title(),
                    state=self.INDIA_STATE_NAMES[state_part.lower()],
                    country='India',
                    raw_text=match.group(0),
                    pattern_type='city_state'
                ))
        
        return matches
    
    def _extract_city_state_no_comma(self, text: str) -> List[LocationMatch]:
        """Extract City State format without comma (e.g., Beaverton Oregon)."""
        matches = []
        for match in self.pattern_city_state_no_comma.finditer(text):
            city, state_part = match.groups()
            city = city.strip()
            state_part = state_part.strip()
            
            # Check if state is full name (US)
            if state_part.lower() in self.STATE_NAME_TO_ABBR:
                matches.append(LocationMatch(
                    city=city.title(),
                    state=state_part.title(),
                    country='USA',
                    raw_text=match.group(0),
                    pattern_type='city_state_no_comma'
                ))
            # Check if it's an Indian state
            elif state_part.lower() in self.INDIA_STATE_NAMES:
                matches.append(LocationMatch(
                    city=city.title(),
                    state=self.INDIA_STATE_NAMES[state_part.lower()],
                    country='India',
                    raw_text=match.group(0),
                    pattern_type='city_state_no_comma'
                ))
        
        return matches
    
    def _extract_city_country(self, text: str) -> List[LocationMatch]:
        """Extract City, Country format."""
        matches = []
        for match in self.pattern_city_country.finditer(text):
            city, country = match.groups()
            
            # Normalize country name
            country_normalized = self._normalize_country(country)
            
            matches.append(LocationMatch(
                city=city.strip().title(),
                country=country_normalized,
                raw_text=match.group(0),
                pattern_type='city_country'
            ))
        
        return matches
    
    def _extract_location_prefix(self, text: str) -> List[LocationMatch]:
        """Extract locations after 'Location:' prefix."""
        matches = []
        for match in self.pattern_location_prefix.finditer(text):
            location_text = match.group(1).strip()
            
            # Try to parse the location text
            parsed = self._parse_location_string(location_text)
            if parsed:
                parsed.pattern_type = 'location_prefix'
                matches.append(parsed)
            else:
                # Try single city name if it's a known major city
                city_name = location_text.split()[0].strip().title() if location_text else None
                if city_name:
                    if city_name in self.MAJOR_CITIES_US:
                        matches.append(LocationMatch(
                            city=city_name,
                            country='USA',
                            raw_text=location_text,
                            pattern_type='location_prefix'
                        ))
                    elif city_name in self.MAJOR_CITIES_INDIA:
                        matches.append(LocationMatch(
                            city=city_name,
                            country='India',
                            raw_text=location_text,
                            pattern_type='location_prefix'
                        ))
        
        return matches
    
    def _extract_parenthetical(self, text: str) -> List[LocationMatch]:
        """Extract locations in parentheses."""
        matches = []
        for match in self.pattern_parenthetical.finditer(text):
            city, state_abbr = match.groups()
            
            if state_abbr in self.US_STATES:
                matches.append(LocationMatch(
                    city=city.strip().title(),
                    state=self.US_STATES[state_abbr],
                    country='USA',
                    raw_text=match.group(0),
                    pattern_type='parenthetical'
                ))
        
        return matches
    
    def _extract_multiline_address(self, text: str) -> List[LocationMatch]:
        """Extract from multi-line postal addresses."""
        matches = []
        for match in self.pattern_multiline_address.finditer(text):
            city, zip_code, state_or_country = match.groups()
            
            city = city.strip().title()
            state_or_country = state_or_country.strip()
            
            # Check if it's an Indian state
            if state_or_country.lower() in self.INDIA_STATE_NAMES:
                matches.append(LocationMatch(
                    city=city,
                    state=self.INDIA_STATE_NAMES[state_or_country.lower()],
                    country='India',
                    zip_code=zip_code,
                    raw_text=match.group(0),
                    pattern_type='multiline_address'
                ))
            # Check if it's a US state
            elif state_or_country.lower() in self.STATE_NAME_TO_ABBR:
                matches.append(LocationMatch(
                    city=city,
                    state=state_or_country.title(),
                    country='USA',
                    zip_code=zip_code,
                    raw_text=match.group(0),
                    pattern_type='multiline_address'
                ))
        
        return matches
    
    def _extract_street_address(self, text: str) -> List[LocationMatch]:
        """Extract from street address format (e.g., 1130 Bibbs Road, Voorhees, New Jersey 08043)."""
        matches = []
        for match in self.pattern_street_address.finditer(text):
            city, state_part, zip_code = match.groups()
            
            city = city.strip().title()
            state_part = state_part.strip()
            
            # Check if state is full name (US)
            if state_part.lower() in self.STATE_NAME_TO_ABBR:
                matches.append(LocationMatch(
                    city=city,
                    state=state_part.title(),
                    country='USA',
                    zip_code=zip_code,
                    raw_text=match.group(0),
                    pattern_type='street_address'
                ))
        
        return matches
    
    def _extract_company_location(self, text: str) -> List[LocationMatch]:
        """
        Extract locations from company headers (e.g., "Company Pvt Ltd, Hyderabad, India").
        
        This helps identify candidate's actual work location from employer information.
        """
        matches = []
        for match in self.pattern_company_location.finditer(text):
            company_name, city, country = match.groups()
            
            city = city.strip().title()
            country_normalized = self._normalize_country(country)
            
            matches.append(LocationMatch(
                city=city,
                country=country_normalized,
                raw_text=match.group(0),
                pattern_type='company_location'
            ))
        
        return matches
    
    def _extract_company_city_only(self, text: str) -> List[LocationMatch]:
        """
        Extract locations from company headers with city-only format (no country).
        Common in Indian resumes: "Company solutions, Hyderabad"
        """
        matches = []
        for match in self.pattern_company_city_only.finditer(text):
            company_part, city = match.groups()
            
            city = city.strip().title()
            
            # Infer country based on city if it's a known major city
            country = None
            if city in self.MAJOR_CITIES_INDIA:
                country = 'India'
            elif city in self.MAJOR_CITIES_US:
                country = 'USA'
            
            if country:  # Only add if we can infer country
                matches.append(LocationMatch(
                    city=city,
                    country=country,
                    raw_text=match.group(0),
                    pattern_type='company_city_only'
                ))
        
        return matches
    
    def _extract_single_city(self, text: str) -> List[LocationMatch]:
        """
        Extract major cities when they appear alone (no state/country).
        Only matches known major cities to avoid false positives.
        """
        matches = []
        
        # Look for major cities in a line by themselves or after common prefixes
        # Pattern: word boundary, optional prefix, city name, word boundary
        city_pattern = r'(?:^|\n|Location:\s*|City:\s*|Based in:\s*)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:\s*$|\s*\n|[,;.])'
        
        for match in re.finditer(city_pattern, text, re.MULTILINE):
            city_name = match.group(1).strip()
            
            # Check if it's a major US city
            if city_name in self.MAJOR_CITIES_US:
                matches.append(LocationMatch(
                    city=city_name,
                    country='USA',
                    raw_text=match.group(0),
                    pattern_type='single_city'
                ))
            # Check if it's a major Indian city
            elif city_name in self.MAJOR_CITIES_INDIA:
                matches.append(LocationMatch(
                    city=city_name,
                    country='India',
                    raw_text=match.group(0),
                    pattern_type='single_city'
                ))
        
        return matches
    
    def _extract_bullet_format(self, text: str) -> List[LocationMatch]:
        """Extract bullet-constrained locations."""
        matches = []
        for match in self.pattern_bullet.finditer(text):
            city, state_abbr = match.groups()
            
            if state_abbr in self.US_STATES:
                matches.append(LocationMatch(
                    city=city.strip().title(),
                    state=self.US_STATES[state_abbr],
                    country='USA',
                    raw_text=match.group(0),
                    pattern_type='bullet'
                ))
        
        return matches
    
    def _parse_location_string(self, location_str: str) -> Optional[LocationMatch]:
        """Parse a location string into components."""
        # Try common patterns
        parts = [p.strip() for p in location_str.split(',')]
        
        if len(parts) >= 2:
            city = parts[0].title()
            second_part = parts[1].strip()
            
            # Check if second part is state (US)
            if len(second_part) == 2 and second_part in self.US_STATES:
                return LocationMatch(
                    city=city,
                    state=self.US_STATES[second_part],
                    country='USA',
                    raw_text=location_str
                )
            elif second_part.lower() in self.STATE_NAME_TO_ABBR:
                return LocationMatch(
                    city=city,
                    state=second_part.title(),
                    country='USA',
                    raw_text=location_str
                )
            # Check if it's Indian state
            elif second_part.lower() in self.INDIA_STATE_NAMES:
                return LocationMatch(
                    city=city,
                    state=self.INDIA_STATE_NAMES[second_part.lower()],
                    country='India',
                    raw_text=location_str
                )
            # Check if it's a country
            elif second_part.lower() in self.COUNTRY_NAMES_LOWER:
                return LocationMatch(
                    city=city,
                    country=self.COUNTRY_NAMES_LOWER[second_part.lower()],
                    raw_text=location_str
                )
        
        return None
    
    def _clean_city_name(self, city: str) -> str:
        """
        Clean city name by removing common prefixes like 'Phone', 'Email', etc.
        
        Example:
            "Csm Phone Hyderabad" → "Hyderabad"
            "Email Address Portland" → "Portland"
            "Contact Number Mumbai" → "Mumbai"
        """
        if not city:
            return city
        
        # Invalid prefixes that should be removed
        invalid_prefixes = [
            'email address', 'phone number', 'mobile number', 'contact number',
            'email', 'phone', 'contact', 'address', 'mobile', 'cell',
            'tel', 'telephone', 'fax', 'csm', 'customer', 'number',
            'name', 'title', 'position', 'role', 'department'
        ]
        
        # Try to remove prefixes (longest first)
        words = city.split()
        cleaned_words = []
        skip_next = False
        
        for i, word in enumerate(words):
            if skip_next:
                skip_next = False
                continue
            
            word_lower = word.lower()
            
            # Check if this word is an invalid prefix
            is_prefix = word_lower in invalid_prefixes
            
            # Check if this word + next word form an invalid prefix
            if i < len(words) - 1:
                two_words = f"{word_lower} {words[i+1].lower()}"
                if two_words in invalid_prefixes:
                    skip_next = True
                    continue
            
            # If not a prefix, keep it
            if not is_prefix:
                cleaned_words.append(word)
        
        return ' '.join(cleaned_words) if cleaned_words else city
    
    def _is_valid_location(self, location: LocationMatch) -> bool:
        """Validate if location match is reasonable."""
        # Must have at least city
        if not location.city:
            return False
        
        # Clean the city name by removing invalid prefixes
        cleaned_city = self._clean_city_name(location.city)
        if cleaned_city != location.city:
            location.city = cleaned_city
        
        # After cleaning, check if we still have a city
        if not location.city or len(location.city) < 3:
            return False
        
        # City should be reasonable length
        if len(location.city) > 50:
            return False
        
        # Check if city name is actually a software/tool name
        city_lower = location.city.lower()
        if city_lower in self.TOOL_BLACKLIST:
            return False
        
        # Check if city contains tool names
        for tool in self.TOOL_BLACKLIST:
            if tool in city_lower:
                return False
        
        # City should not be common non-location words
        invalid_cities = {
            'experience', 'education', 'skills', 'summary', 'objective',
            'work', 'employment', 'professional', 'technical', 'projects',
            'present', 'current', 'previous', 'responsibilities',
            'studio', 'platform', 'tools', 'software', 'application',
            'scientist', 'architect', 'engineer', 'developer', 'manager'
        }
        if city_lower in invalid_cities:
            return False
        
        # Should have city + (state or country)
        if not location.state and not location.country:
            # Only allow if city is in known major cities
            if location.city not in self.MAJOR_CITIES_US and location.city not in self.MAJOR_CITIES_INDIA:
                return False
        
        return True
    
    def _score_location(self, location: LocationMatch) -> int:
        """
        Score location match based on confidence indicators.
        
        Scoring:
        - 100: City + State + ZIP (complete US address)
        - 90: City + State + Country
        - 85: Multi-line address with postal code
        - 80: City + State (validated)
        - 75: City + Country (validated)
        - 70: Parenthetical or bullet format
        - 60: Location prefix with complete info
        - 50: City only (if major city)
        """
        score = 0
        
        # Base score from pattern type
        pattern_scores = {
            'city_state_zip': 100,
            'street_address': 95,
            'location_prefix': 90,  # Explicit location fields get high priority
            'multiline_address': 85,
            'company_location': 82,  # Company locations are reliable
            'city_state': 80,
            'company_city_only': 78,  # Company with city only (inferred country)
            'city_country': 75,
            'city_state_no_comma': 72,
            'parenthetical': 70,
            'bullet': 70,
            'single_city': 65  # Major cities standalone (lower priority)
        }
        score = pattern_scores.get(location.pattern_type, 50)
        
        # Bonus for ZIP code
        if location.zip_code:
            score += 10
        
        # Bonus for known major cities
        if location.city in self.MAJOR_CITIES_US or location.city in self.MAJOR_CITIES_INDIA:
            score += 5
        
        # Bonus for having all components
        if location.city and location.state and location.country:
            score += 5
        
        return min(score, 100)  # Cap at 100
    
    def _is_client_location(self, location: LocationMatch, full_text: str) -> bool:
        """
        Check if location appears in a client/project context (not candidate's location).
        
        Filters out locations that appear in phrases like:
        - "for [ClientName] - City, State"
        - "System Analyst for ClientName - San Francisco, CA"
        - "Project for Client | City, State"
        
        Does NOT filter:
        - Company headers (already in company_location pattern)
        - Explicit location fields ("Location: City")
        - Work history entries
        """
        if not location.raw_text:
            return False
        
        # Don't filter explicit location fields or company locations
        if location.pattern_type in ['location_prefix', 'company_location', 'company_city_only']:
            return False
        
        # Find the context around this location in the original text
        # Expand context window to 200 chars before (to catch job titles/headings)
        raw_text_escaped = re.escape(location.raw_text[:30])  # Use first 30 chars to avoid regex issues
        pattern = rf'.{{0,200}}{raw_text_escaped}.{{0,50}}'
        match = re.search(pattern, full_text, re.IGNORECASE)
        
        if not match:
            return False
        
        context = match.group(0)  # Don't lowercase - need to preserve case for pattern matching
        
        # Client location indicators
        # Pattern: "for ClientName - Location" or "for ClientName, Location"
        client_patterns = [
            r'\bfor\s+[A-Z]\w+\s*[-–—,]\s*',       # "for Finaplex - " or "for Finaplex, "
            r'\bproject\s+for\s+[A-Z]\w+',         # "project for Accenture"
            r'\bworking\s+for\s+[A-Z]\w+\s*[-–—]', # "working for Microsoft -"
            r'\bclient:\s*[A-Z]\w+',                # "Client: Amazon"
            r'\b(?:consultant|contractor)\s+for\s+[A-Z]\w+',  # "Consultant for Google"
        ]
        
        for pattern in client_patterns:
            if re.search(pattern, context):
                return True
        
        return False
    
    def _apply_frequency_boost(self, candidates: List[LocationMatch], full_text: str) -> List[LocationMatch]:
        """
        Boost confidence for locations that appear multiple times.
        
        This helps identify the candidate's actual location when it appears
        consistently across multiple company entries.
        """
        # Count frequency of each unique location (by city)
        city_frequency = {}
        for candidate in candidates:
            if candidate.city:
                city_key = candidate.city.lower()
                city_frequency[city_key] = city_frequency.get(city_key, 0) + 1
        
        # Find max frequency
        max_frequency = max(city_frequency.values()) if city_frequency else 1
        
        # Boost candidates based on frequency
        for candidate in candidates:
            if candidate.city:
                city_key = candidate.city.lower()
                frequency = city_frequency.get(city_key, 1)
                
                # Add frequency bonus (up to 20 points)
                if frequency > 1:
                    frequency_bonus = min(20, frequency * 5)
                    candidate.confidence += frequency_bonus
                    
                    # Extra boost if it's the most frequent
                    if frequency == max_frequency and max_frequency >= 3:
                        candidate.confidence += 10
        
        return candidates
    
    def _normalize_country(self, country: str) -> str:
        """Normalize country name."""
        country_lower = country.lower()
        
        if country_lower in ['us', 'usa', 'united states']:
            return 'USA'
        elif country_lower in ['uk', 'united kingdom']:
            return 'UK'
        elif country_lower == 'india':
            return 'India'
        
        return country.title()


# Convenience function for easy use
def extract_location(text: str) -> str:
    """
    Extract location from resume text.
    
    Args:
        text: Raw resume text
        
    Returns:
        Formatted location string or "Unknown"
        
    Example:
        >>> text = "Location: Portland, OR 97124"
        >>> extract_location(text)
        'Portland, Oregon, USA'
    """
    extractor = LocationExtractor()
    return extractor.extract_location(text)


# Test function
def test_location_extractor():
    """Test the location extractor with sample inputs."""
    
    test_cases = [
        ("Location: Hyderabad", "Hyderabad, India"),
        ("Hyderabad, India", "Hyderabad, India"),
        ("Portland, OR 97124", "Portland, Oregon"),
        ("(Fort Atkinson, WI/Remote)", "Fort Atkinson, Wisconsin"),
        ("Beaverton Oregon", "Beaverton, Oregon"),
        ("● Portland, OR ●", "Portland, Oregon"),
        ("Software Engineer\nBeaverton, Oregon\n2020-Present", "Beaverton, Oregon"),
        ("D.NO 1-121/1, Ramalayam Street\nKorumilli, 533309, Andhra Pradesh", "Korumilli, Andhra Pradesh"),
        ("1130 Bibbs Road, Apt#4, Voorhees, New Jersey 08043", "Voorhees, New Jersey"),
        ("Current Location: Seattle, WA", "Seattle, Washington"),
        # New test case for client location filtering (with multiple company occurrences)
        ("Mavensoft Systems Pvt Ltd, Hyderabad, India\nDirector\nSystem Analyst for Finaplex - San Francisco, CA\nNapier Healthcare Pvt Ltd, Hyderabad, India\nManager", "Hyderabad, India"),
        # Test PDF artifact fix
        ("Location: San Franc isco, CA", "San Francisco"),
        # Test tool name filtering (Sathish case)
        ("> RAMA corporate and IT solutions, Hyderabad\nMicrosoft Copilot Studio, MS Office", "Hyderabad"),
        # Test company city-only format
        ("Cubespace technologies Pvt Ltd, Bangalore\nFeb 2020 to April 2023", "Bangalore"),
    ]
    
    print("=" * 80)
    print("Location Extractor - Test Results")
    print("=" * 80)
    
    extractor = LocationExtractor()
    
    passed = 0
    failed = 0
    
    for i, (test_input, expected_contains) in enumerate(test_cases, 1):
        result = extractor.extract_location(test_input)
        
        # Check if result contains expected key parts
        success = expected_contains.split(',')[0] in result  # At least city matches
        
        if success:
            status = "✓"
            passed += 1
        else:
            status = "✗"
            failed += 1
        
        print(f"\n{i}. {status} Input: {test_input[:60]}...")
        print(f"   Output: {result}")
        print(f"   Expected to contain: {expected_contains}")
    
    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)
    print("\nNOTE: Location extraction uses pure Python regex and rules")
    print("      (NO AI models are used!)")


if __name__ == "__main__":
    # Run tests
    test_location_extractor()

