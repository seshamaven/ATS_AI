"""
Phone Number Extractor - Pure Python Logic (No AI/LLMs)

Extracts and normalizes phone numbers from raw resume text.
Supports international formats (+91, +1), various separators, and common prefixes.
"""

import re
from typing import List, Set


class PhoneExtractor:
    """Extract phone numbers from text using regex patterns."""
    
    # Phone number prefixes/labels (case-insensitive)
    PHONE_PREFIXES = [
        r'\bphone\s*:?\s*',
        r'\bcell\s*:?\s*',
        r'\bmobile\s*:?\s*',
        r'\bmob\s*:?\s*',
        r'\btel\s*:?\s*',
        r'\btelephone\s*:?\s*',
        r'\bcall\s*:?\s*',
        r'\bp\s*:?\s*',
        r'\bt\s*:?\s*',
        r'\bph\s*:?\s*',
        r'\bcontact\s+no\.?\s*:?\s*',
        r'\bphone\s+number\s*:?\s*',
        r'\boffice\s*:?\s*',
        r'\bwhatsapp\s*:?\s*',
        r'\bwork\s*:?\s*',
        r'\bdirect\s*:?\s*',
        r'\bemergency\s*:?\s*',
        r'\blandline\s*:?\s*',
        r'\bresidence\s*:?\s*',
        r'\bfax\s*:?\s*',
        r'\bmobile\s+no\.?\s*:?\s*',
    ]
    
    def __init__(self):
        """Initialize phone extractor with compiled patterns."""
        # Combine all prefixes into one pattern
        self.prefix_pattern = re.compile(
            '(' + '|'.join(self.PHONE_PREFIXES) + ')',
            re.IGNORECASE
        )
        
        # Main phone number pattern - matches various formats
        # Supports: +country_code, (area_code), and digits with separators (-, ., /, space)
        self.phone_pattern = re.compile(
            r'''
            (?:^|(?<=\s)|(?<=:)|(?<=\())              # Start of string, after space, colon, or (
            (?:\+?\d{1,3}[\s\-\.]?)?                   # Optional country code: +91, +1, 91, 1
            (?:\(?\d{2,5}\)?[\s\-\./]?)?               # Optional area code: (503), 503, (91216)
            \d{3,5}[\s\-\./]?\d{3,5}(?:[\s\-\./]?\d{1,5})? # Main number with separators
            (?=\s|$|[,;]|\.(?:\s|$))                  # End of string, space, punctuation
            ''',
            re.VERBOSE
        )
        
        # Pattern to match phone numbers after prefixes
        self.phone_after_prefix_pattern = re.compile(
            r'[\+\d][\d\s\-\.\(\)/]+\d',
            re.IGNORECASE
        )
    
    def extract_phone_numbers(self, text: str) -> List[str]:
        """
        Extract and normalize phone numbers from text.
        
        Args:
            text: Raw text containing phone numbers
            
        Returns:
            List of normalized phone numbers (strings with only digits and +)
        """
        if not text:
            return []
        
        found_numbers: Set[str] = set()
        
        # Method 1: Extract numbers after known prefixes
        found_numbers.update(self._extract_with_prefixes(text))
        
        # Method 2: Extract numbers using general pattern
        found_numbers.update(self._extract_with_pattern(text))
        
        # Clean, validate, and normalize
        cleaned_numbers = []
        for number in found_numbers:
            normalized = self._normalize_phone_number(number)
            if self._is_valid_phone(normalized):
                cleaned_numbers.append(normalized)
        
        # Remove duplicates and sort
        return sorted(list(set(cleaned_numbers)))
    
    def _extract_with_prefixes(self, text: str) -> Set[str]:
        """Extract phone numbers that appear after known prefixes."""
        results = set()
        
        # Find all prefix matches
        for match in self.prefix_pattern.finditer(text):
            prefix_end = match.end()
            # Extract text after prefix (next 30 chars max)
            remaining_text = text[prefix_end:prefix_end + 30]
            
            # Find phone number in remaining text
            phone_match = self.phone_after_prefix_pattern.search(remaining_text)
            if phone_match:
                phone_text = phone_match.group(0)
                results.add(phone_text.strip())
        
        return results
    
    def _extract_with_pattern(self, text: str) -> Set[str]:
        """Extract phone numbers using general pattern matching."""
        results = set()
        
        # Split text into lines for better matching
        lines = text.split('\n')
        
        for line in lines:
            # Find all potential phone numbers in the line
            matches = self.phone_pattern.finditer(line)
            for match in matches:
                phone_text = match.group(0).strip()
                results.add(phone_text)
        
        return results
    
    def _normalize_phone_number(self, phone_str: str) -> str:
        """
        Normalize phone number to consistent format.
        
        Preserves country code (+91, +1) and digits only.
        Removes all other characters (spaces, dashes, dots, parentheses).
        
        Args:
            phone_str: Raw phone number string
            
        Returns:
            Normalized phone number (e.g., "+919121676672", "5037246862")
        """
        if not phone_str:
            return ""
        
        # Check if starts with +
        has_plus = phone_str.strip().startswith('+')
        
        # Extract only digits
        digits_only = re.sub(r'\D', '', phone_str)
        
        # Add back + for international numbers if it was there
        if has_plus and digits_only:
            return '+' + digits_only
        
        return digits_only
    
    def _is_valid_phone(self, phone_str: str) -> bool:
        """
        Validate if extracted string is a valid phone number.
        
        Rules:
        - Must have at least 6 digits
        - Must not exceed 15 digits (international standard)
        - Should not be all same digit (like 000000, 111111)
        - Should not be concatenated years (like 200820122017)
        
        Args:
            phone_str: Normalized phone number string
            
        Returns:
            True if valid, False otherwise
        """
        if not phone_str:
            return False
        
        # Extract digits only (remove +)
        digits = phone_str.replace('+', '')
        
        # Check length
        if len(digits) < 9 or len(digits) > 15:
            return False
        
        # Reject if all digits are the same (like 000000, 111111)
        if len(set(digits)) == 1:
            return False
        
        # Reject common false positives (year-like numbers, simple sequences)
        if digits in ['123456', '234567', '345678', '456789', '567890']:
            return False
        
        # Reject concatenated years pattern (e.g., 200820122017, 201520182021)
        # Years are typically 19xx or 20xx
        if self._is_concatenated_years(digits):
            return False
        
        # Reject 9-digit numbers that look like profile/account IDs (not phone numbers)
        # Valid 9-digit phone numbers typically start with specific patterns
        # Profile IDs like "357756472" don't follow phone number patterns
        if len(digits) == 9:
            # US phone numbers (without country code) are 10 digits, not 9
            # 9-digit numbers are likely IDs, not phone numbers
            # Exception: some international formats might be 9 digits
            # Check if it looks like a valid phone pattern (area code + number)
            # Most valid 9-digit phones start with 2-9 (not 0 or 1)
            if not digits[0] in '23456789':
                return False
        
        return True
    
    def _is_concatenated_years(self, digits: str) -> bool:
        """
        Check if the number looks like concatenated years.
        
        Examples of rejected patterns:
        - 200820122017 (2008, 2012, 2017)
        - 201520182021 (2015, 2018, 2021)
        - 19952000 (1995, 2000)
        
        Args:
            digits: String of digits only
            
        Returns:
            True if looks like concatenated years, False otherwise
        """
        if len(digits) < 8:
            return False
        
        # Check if string can be split into valid years (4 digits each starting with 19 or 20)
        # Try to find year patterns
        year_pattern = re.compile(r'(19\d{2}|20\d{2})')
        years_found = year_pattern.findall(digits)
        
        # If we find 2+ years and they cover most of the string, it's likely concatenated years
        if len(years_found) >= 2:
            total_year_digits = len(years_found) * 4
            # If years cover 80%+ of the digits, reject as concatenated years
            if total_year_digits >= len(digits) * 0.8:
                return True
        
        # Also check for exact matches of common concatenated year lengths
        # 8 digits = 2 years, 12 digits = 3 years, 16 digits = 4 years
        if len(digits) in [8, 12, 16]:
            # Check if all 4-digit chunks are valid years
            chunks = [digits[i:i+4] for i in range(0, len(digits), 4)]
            valid_year_chunks = sum(1 for chunk in chunks if chunk.startswith(('19', '20')) and chunk.isdigit())
            if valid_year_chunks == len(chunks):
                return True
        
        return False
    
    def extract_phone_numbers_detailed(self, text: str) -> List[dict]:
        """
        Extract phone numbers with additional details.
        
        Args:
            text: Raw text containing phone numbers
            
        Returns:
            List of dictionaries with 'number', 'original', 'type' keys
        """
        if not text:
            return []
        
        found_numbers = []
        processed = set()  # Track processed numbers to avoid duplicates
        
        # Method 1: Extract with prefix context
        for match in self.prefix_pattern.finditer(text):
            prefix = match.group(0).strip()
            prefix_end = match.end()
            remaining_text = text[prefix_end:prefix_end + 30]
            
            phone_match = self.phone_after_prefix_pattern.search(remaining_text)
            if phone_match:
                original = phone_match.group(0).strip()
                normalized = self._normalize_phone_number(original)
                
                if self._is_valid_phone(normalized) and normalized not in processed:
                    phone_type = self._classify_phone_type(prefix)
                    found_numbers.append({
                        'number': normalized,
                        'original': original,
                        'type': phone_type,
                        'prefix': prefix.strip(':').strip()
                    })
                    processed.add(normalized)
        
        # Method 2: Extract without prefix (fallback)
        for match in self.phone_pattern.finditer(text):
            original = match.group(0).strip()
            normalized = self._normalize_phone_number(original)
            
            if self._is_valid_phone(normalized) and normalized not in processed:
                found_numbers.append({
                    'number': normalized,
                    'original': original,
                    'type': 'unknown',
                    'prefix': None
                })
                processed.add(normalized)
        
        return found_numbers
    
    def _classify_phone_type(self, prefix: str) -> str:
        """Classify phone number type based on prefix."""
        prefix_lower = prefix.lower().strip()
        
        if 'mobile' in prefix_lower or 'cell' in prefix_lower or 'mob' in prefix_lower:
            return 'mobile'
        elif 'work' in prefix_lower or 'office' in prefix_lower:
            return 'work'
        elif 'home' in prefix_lower or 'residence' in prefix_lower or 'landline' in prefix_lower:
            return 'home'
        elif 'fax' in prefix_lower:
            return 'fax'
        elif 'whatsapp' in prefix_lower:
            return 'whatsapp'
        elif 'emergency' in prefix_lower:
            return 'emergency'
        else:
            return 'phone'


# Convenience functions for easy use
def extract_phone_numbers(text: str) -> List[str]:
    """
    Extract and normalize phone numbers from text.
    
    Args:
        text: Raw text containing phone numbers
        
    Returns:
        List of normalized phone numbers
        
    Example:
        >>> text = "Contact: (503) 724-6862 or mobile: +91-9701820970"
        >>> extract_phone_numbers(text)
        ['+919701820970', '5037246862']
    """
    extractor = PhoneExtractor()
    return extractor.extract_phone_numbers(text)


def extract_phone_numbers_detailed(text: str) -> List[dict]:
    """
    Extract phone numbers with additional context.
    
    Args:
        text: Raw text containing phone numbers
        
    Returns:
        List of dictionaries with number details
        
    Example:
        >>> text = "Mobile: +91-9701820970"
        >>> extract_phone_numbers_detailed(text)
        [{'number': '+919701820970', 'original': '+91-9701820970', 
          'type': 'mobile', 'prefix': 'Mobile'}]
    """
    extractor = PhoneExtractor()
    return extractor.extract_phone_numbers_detailed(text)


# Test function
def test_phone_extractor():
    """Test the phone extractor with sample inputs."""
    
    test_cases = [
        "(916) 612 7176",
        "5034340832",
        "503-434-0832",
        "(503)487-7969",
        "917.400.6121",
        "Phone: 503-724-6862",
        "(971) 600-5115",
        "617-308-6099",
        "Cell: 908-625-5702",
        "+91 9121676672",
        "+91-9848512516",
        "+91-9652376156",
        "mobile no:93470 23900",
        "Phone: +91 9701820970",
        "Tel: 503 724 6862",
        "Telephone: 503.724.6862",
        "Call: (503)7246862",
        "T: +1 (503) 724-6862",
        "P: +91 (97018) 20970",
        "Contact No.: +91 (91216) 76672",
        "Mob: 97018 20970",
        "Mobile: +91 97018 20970",
        "Ph: 9701820970",
        "Phone number : 503 - 724 - 6862",
        "Office: (503)724-6862",
        "WhatsApp: +91-9701820970",
        "Work: +1-503-724-6862",
        "Direct: 5037246862",
        "Emergency: 503/724/6862",
        "Landline: 040-23456789",
        "Residence: 272-645-3889",
        "Fax: 503-724-6862",
    ]
    
    print("=" * 80)
    print("Phone Number Extractor - Test Results")
    print("=" * 80)
    
    extractor = PhoneExtractor()
    
    for i, test_case in enumerate(test_cases, 1):
        extracted = extractor.extract_phone_numbers(test_case)
        print(f"\n{i}. Input:  {test_case}")
        print(f"   Output: {extracted}")
    
    print("\n" + "=" * 80)
    print("Testing with combined text:")
    print("=" * 80)
    
    combined_text = "\n".join(test_cases)
    all_numbers = extractor.extract_phone_numbers(combined_text)
    
    print(f"\nTotal unique phone numbers found: {len(all_numbers)}")
    print("\nAll extracted numbers:")
    for number in all_numbers:
        print(f"  - {number}")
    
    print("\n" + "=" * 80)
    print("Testing detailed extraction:")
    print("=" * 80)
    
    sample_text = """
    Contact Information:
    Mobile: +91-9701820970
    Office: (503) 724-6862
    WhatsApp: +91 9121676672
    """
    
    detailed = extractor.extract_phone_numbers_detailed(sample_text)
    print("\nDetailed extraction results:")
    for item in detailed:
        print(f"  Number: {item['number']}")
        print(f"  Original: {item['original']}")
        print(f"  Type: {item['type']}")
        print(f"  Prefix: {item['prefix']}")
        print()


if __name__ == "__main__":
    # Run tests
    test_phone_extractor()


