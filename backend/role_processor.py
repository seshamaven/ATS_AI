"""
Role Processor Module
=====================

Helps categorize and map original roles to normalized roles.
Provides functions to:
- Insert role mappings into the database
- Query normalized roles for original roles
- Bulk import role mappings
- Auto-categorize roles based on keywords

Author: ATS System
"""

import logging
import re
import json
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher
import mysql.connector
from mysql.connector import Error
from ats_config import ATSConfig

logger = logging.getLogger(__name__)

# Import skill extraction and role inference
try:
    from skill_extractor import extract_skills
    SKILL_EXTRACTOR_AVAILABLE = True
except ImportError:
    SKILL_EXTRACTOR_AVAILABLE = False
    logger.warning("skill_extractor not available, skill-based role detection disabled")

try:
    from role_extract import infer_role_from_skills
    ROLE_EXTRACT_AVAILABLE = True
except ImportError:
    ROLE_EXTRACT_AVAILABLE = False
    logger.warning("role_extract not available, skill-based role detection disabled")


class RoleProcessor:
    """Manages role normalization and mapping operations."""
    
    def __init__(self, config: Dict = None):
        """Initialize RoleProcessor with database config."""
        self.config = config or ATSConfig.get_mysql_config()
        self.connection = None
        self.cursor = None
    
    def connect(self) -> bool:
        """Establish database connection."""
        try:
            self.connection = mysql.connector.connect(**self.config)
            self.cursor = self.connection.cursor(dictionary=True)
            logger.info("Connected to database for role processing")
            return True
        except Error as e:
            logger.error(f"Error connecting to database: {e}")
            return False
    
    def disconnect(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def insert_role_mapping(self, normalized_role: str, original_role: str) -> bool:
        """
        Insert a single role mapping.
        Adds original_role to the JSON array for the normalized_role.
        
        Args:
            normalized_role: The normalized role (e.g., "Software Engineer")
            original_role: The original role (e.g., ".NET Developer")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if normalized role already exists
            check_query = "SELECT roles FROM role_processor WHERE Normalised_roles = %s"
            self.cursor.execute(check_query, (normalized_role,))
            result = self.cursor.fetchone()
            
            if result:
                # Role exists, append to JSON array if not already present
                existing_roles = json.loads(result['roles'])
                if original_role not in existing_roles:
                    existing_roles.append(original_role)
                    update_query = """
                        UPDATE role_processor 
                        SET roles = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE Normalised_roles = %s
                    """
                    self.cursor.execute(update_query, (json.dumps(existing_roles), normalized_role))
                    logger.info(f"Added '{original_role}' to '{normalized_role}'")
                else:
                    logger.info(f"'{original_role}' already exists for '{normalized_role}'")
            else:
                # New normalized role, create with single original role
                insert_query = """
                    INSERT INTO role_processor (Normalised_roles, roles)
                    VALUES (%s, %s)
                """
                roles_json = json.dumps([original_role])
                self.cursor.execute(insert_query, (normalized_role, roles_json))
                logger.info(f"Created '{normalized_role}' with '{original_role}'")
            
            self.connection.commit()
            return True
        except Error as e:
            logger.error(f"Error inserting role mapping: {e}")
            self.connection.rollback()
            return False
    
    def bulk_insert_role_mappings(self, mappings: List[Tuple[str, str]]) -> int:
        """
        Bulk insert role mappings.
        Groups by normalized role and aggregates original roles into JSON arrays.
        
        Args:
            mappings: List of tuples (normalized_role, original_role)
            
        Returns:
            Number of successfully inserted normalized roles
        """
        if not mappings:
            return 0
        
        # Group mappings by normalized role
        grouped = {}
        for normalized_role, original_role in mappings:
            if normalized_role not in grouped:
                grouped[normalized_role] = []
            if original_role not in grouped[normalized_role]:
                grouped[normalized_role].append(original_role)
        
        success_count = 0
        try:
            for normalized_role, original_roles in grouped.items():
                # Check if normalized role already exists
                check_query = "SELECT roles FROM role_processor WHERE Normalised_roles = %s"
                self.cursor.execute(check_query, (normalized_role,))
                result = self.cursor.fetchone()
                
                if result:
                    # Merge with existing roles
                    existing_roles = json.loads(result['roles'])
                    merged_roles = list(set(existing_roles + original_roles))
                    update_query = """
                        UPDATE role_processor 
                        SET roles = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE Normalised_roles = %s
                    """
                    self.cursor.execute(update_query, (json.dumps(merged_roles), normalized_role))
                else:
                    # Insert new normalized role
                    insert_query = """
                        INSERT INTO role_processor (Normalised_roles, roles)
                        VALUES (%s, %s)
                    """
                    self.cursor.execute(insert_query, (normalized_role, json.dumps(original_roles)))
                
                success_count += 1
            
            self.connection.commit()
            logger.info(f"Bulk inserted {success_count} normalized roles with {len(mappings)} total mappings")
        except Error as e:
            logger.error(f"Error bulk inserting role mappings: {e}")
            self.connection.rollback()
        
        return success_count
    
    def get_normalized_role(self, original_role: str) -> Optional[str]:
        """
        Get normalized role for an original role.
        Searches within JSON arrays.
        
        Args:
            original_role: The original role to look up
            
        Returns:
            Normalized role if found, None otherwise
        """
        try:
            query = """
                SELECT Normalised_roles 
                FROM role_processor 
                WHERE JSON_CONTAINS(roles, %s)
                LIMIT 1
            """
            # JSON_CONTAINS needs the value as a JSON string
            role_json = json.dumps(original_role)
            self.cursor.execute(query, (role_json,))
            result = self.cursor.fetchone()
            return result['Normalised_roles'] if result else None
        except Error as e:
            logger.error(f"Error getting normalized role: {e}")
            return None
    
    def get_all_original_roles(self, normalized_role: str) -> List[str]:
        """
        Get all original roles for a normalized role.
        
        Args:
            normalized_role: The normalized role
            
        Returns:
            List of original roles
        """
        try:
            query = """
                SELECT roles 
                FROM role_processor 
                WHERE Normalised_roles = %s
            """
            self.cursor.execute(query, (normalized_role,))
            result = self.cursor.fetchone()
            if result:
                roles_list = json.loads(result['roles'])
                return sorted(roles_list)
            return []
        except Error as e:
            logger.error(f"Error getting original roles: {e}")
            return []
    
    def get_all_normalized_roles(self) -> List[str]:
        """Get all unique normalized roles."""
        try:
            query = """
                SELECT Normalised_roles 
                FROM role_processor 
                ORDER BY Normalised_roles
            """
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            return [row['Normalised_roles'] for row in results]
        except Error as e:
            logger.error(f"Error getting normalized roles: {e}")
            return []
    
    def get_all_mappings(self) -> Dict[str, List[str]]:
        """
        Get all role mappings as a dictionary.
        
        Returns:
            Dictionary mapping normalized_role -> list of original roles
        """
        mappings = {}
        try:
            query = """
                SELECT Normalised_roles, roles 
                FROM role_processor 
                ORDER BY Normalised_roles
            """
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            
            for row in results:
                normalized = row['Normalised_roles']
                roles_json = row['roles']
                original_roles = json.loads(roles_json)
                mappings[normalized] = sorted(original_roles)
            
            return mappings
        except Error as e:
            logger.error(f"Error getting all mappings: {e}")
            return {}
    
    def auto_categorize_role(self, original_role: str, normalized_roles: List[str]) -> Optional[str]:
        """
        Automatically categorize an original role into a normalized role based on keywords.
        
        Args:
            original_role: The original role to categorize
            normalized_roles: List of available normalized roles
            
        Returns:
            Best matching normalized role or None
        """
        original_lower = original_role.lower()
        
        # Define keyword patterns for common normalized roles
        category_keywords = {
            'software engineer': [
                'developer', 'programmer', 'engineer', 'software', 'coder',
                'java', 'python', 'c#', '.net', 'dotnet', 'javascript',
                'node', 'react', 'angular', 'vue', 'full stack', 'backend',
                'frontend', 'fullstack'
            ],
            'data engineer': [
                'data engineer', 'etl', 'data pipeline', 'big data',
                'hadoop', 'spark', 'data warehouse', 'data integration'
            ],
            'data scientist': [
                'data scientist', 'ml engineer', 'machine learning',
                'ai engineer', 'deep learning', 'nlp', 'computer vision',
                'data analyst', 'analytics'
            ],
            'devops engineer': [
                'devops', 'sre', 'site reliability', 'ci/cd', 'kubernetes',
                'docker', 'terraform', 'cloud engineer', 'aws', 'azure', 'gcp'
            ],
            'qa engineer': [
                'qa', 'quality assurance', 'tester', 'test engineer',
                'automation', 'sdet', 'testing'
            ],
            'database administrator': [
                'dba', 'database', 'sql', 'oracle', 'mysql', 'postgresql',
                'mongodb', 'nosql'
            ],
            'systems engineer': [
                'systems engineer', 'system administrator', 'sysadmin',
                'network engineer', 'infrastructure', 'it support'
            ],
            'security engineer': [
                'security', 'cybersecurity', 'penetration tester',
                'ethical hacker', 'soc analyst', 'siem'
            ],
            'project manager': [
                'project manager', 'program manager', 'scrum master',
                'product manager', 'product owner', 'pm'
            ],
            'architect': [
                'architect', 'solution architect', 'enterprise architect',
                'technical architect', 'system architect'
            ],
            'mobile developer': [
                'mobile', 'android', 'ios', 'react native', 'flutter',
                'swift', 'kotlin'
            ],
            'ui/ux designer': [
                'ui', 'ux', 'designer', 'ui/ux', 'user interface',
                'user experience', 'interaction designer'
            ]
        }
        
        # Score each normalized role
        best_match = None
        best_score = 0.0
        
        for normalized_role in normalized_roles:
            normalized_lower = normalized_role.lower()
            score = 0.0
            
            # Check if normalized role has keyword patterns
            if normalized_lower in category_keywords:
                keywords = category_keywords[normalized_lower]
                for keyword in keywords:
                    if keyword in original_lower:
                        score += 1.0
            
            # Also check direct similarity
            similarity = SequenceMatcher(None, original_lower, normalized_lower).ratio()
            score += similarity * 0.5
            
            # Check word overlap
            original_words = set(original_lower.split())
            normalized_words = set(normalized_lower.split())
            word_overlap = len(original_words & normalized_words) / max(len(normalized_words), 1)
            score += word_overlap * 0.3
            
            if score > best_score:
                best_score = score
                best_match = normalized_role
        
        # Only return if score is above threshold
        if best_score >= 0.3:
            return best_match
        
        return None
    
    def categorize_and_insert(self, original_roles: List[str], normalized_roles: List[str]) -> Dict[str, int]:
        """
        Automatically categorize multiple original roles and insert mappings.
        
        Args:
            original_roles: List of original roles to categorize
            normalized_roles: List of available normalized roles
            
        Returns:
            Dictionary with categorization statistics
        """
        stats = {
            'total': len(original_roles),
            'categorized': 0,
            'uncategorized': 0,
            'inserted': 0
        }
        
        mappings = []
        uncategorized = []
        
        for original_role in original_roles:
            normalized = self.auto_categorize_role(original_role, normalized_roles)
            if normalized:
                mappings.append((normalized, original_role))
                stats['categorized'] += 1
            else:
                uncategorized.append(original_role)
                stats['uncategorized'] += 1
        
        # Bulk insert categorized mappings
        if mappings:
            inserted = self.bulk_insert_role_mappings(mappings)
            stats['inserted'] = inserted
        
        logger.info(f"Categorization complete: {stats}")
        if uncategorized:
            logger.warning(f"Uncategorized roles: {uncategorized[:10]}...")  # Show first 10
        
        return stats
    
    def normalize_role_from_resume(
        self, 
        original_role: Optional[str] = None, 
        resume_text: Optional[str] = None,
        primary_skills: Optional[str] = None,
        secondary_skills: Optional[str] = None,
        fuzzy_threshold: float = 0.75
    ) -> str:
        """
        Normalize a job role from resume by matching against role_processor table.
        
        This function:
        1. First tries to match the original_role/designation from resume
        2. If original_role is missing/empty (fresher resume), extracts skills and infers role
        3. Matches against entries in role_processor table
        4. If match found in roles JSON array, returns corresponding Normalised_role
        5. If no match found, returns "Others"
        
        Args:
            original_role: The original role/designation from resume (optional)
            resume_text: Full resume text (optional, used if original_role is missing)
            primary_skills: Comma-separated primary/technical skills (optional)
            secondary_skills: Comma-separated secondary skills (optional)
            fuzzy_threshold: Similarity threshold for fuzzy matching (0.0 to 1.0)
            
        Returns:
            Normalized role name or "Others" if no match found
        """
        # Step 1: Try to normalize from original_role if provided
        if original_role and isinstance(original_role, str):
            original_role = original_role.strip()
            if original_role:
                try:
                    # Normalize the input role for matching
                    normalized_input = self._normalize_role_text(original_role)
                    
                    # First, try exact match using JSON_CONTAINS
                    exact_match = self._find_exact_match(normalized_input)
                    if exact_match:
                        logger.info(f"Exact match found: '{original_role}' -> '{exact_match}'")
                        return exact_match
                    
                    # If no exact match, try fuzzy matching
                    fuzzy_match = self._find_fuzzy_match(normalized_input, original_role, fuzzy_threshold)
                    if fuzzy_match:
                        logger.info(f"Fuzzy match found: '{original_role}' -> '{fuzzy_match}' (threshold: {fuzzy_threshold})")
                        return fuzzy_match
                    
                    logger.info(f"No match found for '{original_role}', trying skill-based inference...")
                except Error as e:
                    logger.error(f"Error normalizing role '{original_role}': {e}")
        
        # Step 2: If original_role is missing/empty, infer from skills (for fresher resumes)
        logger.info("Original role not found or empty, attempting skill-based role inference...")
        inferred_role = self._infer_role_from_skills(resume_text, primary_skills, secondary_skills)
        
        if inferred_role:
            # Try to normalize the inferred role
            try:
                normalized_input = self._normalize_role_text(inferred_role)
                exact_match = self._find_exact_match(normalized_input)
                if exact_match:
                    logger.info(f"Skill-based inference: '{inferred_role}' -> '{exact_match}'")
                    return exact_match
                
                fuzzy_match = self._find_fuzzy_match(normalized_input, inferred_role, fuzzy_threshold)
                if fuzzy_match:
                    logger.info(f"Skill-based inference (fuzzy): '{inferred_role}' -> '{fuzzy_match}'")
                    return fuzzy_match
                
                # If inferred role doesn't match any normalized role, try direct mapping
                mapped_role = self._map_inferred_role_to_normalized(inferred_role)
                if mapped_role:
                    logger.info(f"Skill-based inference (mapped): '{inferred_role}' -> '{mapped_role}'")
                    return mapped_role
            except Error as e:
                logger.error(f"Error normalizing inferred role '{inferred_role}': {e}")
        
        # Step 3: No match found
        logger.info("No role match found, returning 'Others'")
        return "Others"
    
    def _infer_role_from_skills(
        self, 
        resume_text: Optional[str] = None,
        primary_skills: Optional[str] = None,
        secondary_skills: Optional[str] = None
    ) -> Optional[str]:
        """
        Infer role from skills when designation is missing (for fresher resumes).
        
        Args:
            resume_text: Full resume text
            primary_skills: Comma-separated primary/technical skills
            secondary_skills: Comma-separated secondary skills
            
        Returns:
            Inferred role name or None
        """
        # Extract skills if not provided
        if not primary_skills and resume_text and SKILL_EXTRACTOR_AVAILABLE:
            try:
                skills_result = extract_skills(resume_text, return_categories=True)
                primary_skills_list = skills_result.get('tech_skills', [])
                secondary_skills_list = skills_result.get('soft_skills', [])
                primary_skills = ', '.join(primary_skills_list) if primary_skills_list else None
                secondary_skills = ', '.join(secondary_skills_list) if secondary_skills_list else None
                logger.info(f"Extracted {len(primary_skills_list)} technical skills and {len(secondary_skills_list)} soft skills from resume")
            except Exception as e:
                logger.error(f"Error extracting skills from resume: {e}")
        
        # Infer role from skills
        if primary_skills and ROLE_EXTRACT_AVAILABLE:
            try:
                result = infer_role_from_skills(primary_skills, secondary_skills or "")
                if result:
                    inferred_role, _ = result  # Get role (ignore subrole)
                    logger.info(f"Inferred role from skills: '{inferred_role}'")
                    return inferred_role
            except Exception as e:
                logger.error(f"Error inferring role from skills: {e}")
        
        return None
    
    def _map_inferred_role_to_normalized(self, inferred_role: str) -> Optional[str]:
        """
        Map inferred role (from role_extract) to normalized role (from role_processor table).
        
        Args:
            inferred_role: Role inferred from skills (e.g., "Software Engineer", "Data Scientist")
            
        Returns:
            Normalized role name or None
        """
        # Mapping from inferred roles to normalized roles
        role_mapping = {
            # Software Engineering roles
            "Software Engineer": "Software Engineer",
            "Backend Developer": "Software Engineer",
            "Frontend Developer": "Software Engineer",
            "Full Stack Developer": "Software Engineer",
            
            # Data roles
            "Data Scientist": "Software Engineer",  # Map to Software Engineer or create Data Scientist category
            "Data Engineer": "Software Engineer",
            "Data Analyst": "Data Analyst",
            
            # DevOps/Cloud
            "DevOps Engineer": "Software Engineer",
            "Cloud Engineer": "Software Engineer",
            
            # QA
            "QA Engineer": "Software Engineer",
            
            # Database
            "Database Administrator": "Database Administrator",
            
            # Mobile
            "Mobile Developer": "Software Engineer",
            
            # SAP/Consultant
            "SAP Consultant": "Consultant",
        }
        
        # Try exact match first
        if inferred_role in role_mapping:
            return role_mapping[inferred_role]
        
        # Try case-insensitive match
        inferred_lower = inferred_role.lower()
        for key, value in role_mapping.items():
            if key.lower() == inferred_lower:
                return value
        
        # Try partial match
        for key, value in role_mapping.items():
            if key.lower() in inferred_lower or inferred_lower in key.lower():
                return value
        
        return None
    
    def _normalize_role_text(self, text: str) -> str:
        """Normalize text for matching (lowercase, remove special chars, normalize spaces)."""
        if not text:
            return ""
        
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove common variations and abbreviations
        normalized = re.sub(r'\b(sr\.?|senior)\b', 'sr', normalized)
        normalized = re.sub(r'\b(jr\.?|junior)\b', 'jr', normalized)
        normalized = re.sub(r'\b(\.net|dotnet)\b', 'net', normalized)
        normalized = re.sub(r'\b(asp\.net|aspnet)\b', 'aspnet', normalized)
        normalized = re.sub(r'\bc#\b', 'csharp', normalized)  # Handle C# specially
        normalized = re.sub(r'\bc\+\+\b', 'cpp', normalized)  # Handle C++
        
        # Remove special characters but keep spaces
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _find_exact_match(self, normalized_input: str) -> Optional[str]:
        """Find exact match using JSON_CONTAINS."""
        try:
            # Try exact match first
            query = """
                SELECT Normalised_roles 
                FROM role_processor 
                WHERE JSON_CONTAINS(roles, %s)
                LIMIT 1
            """
            role_json = json.dumps(normalized_input)
            self.cursor.execute(query, (role_json,))
            result = self.cursor.fetchone()
            if result:
                return result['Normalised_roles']
            
            # Also try case-insensitive search by checking all roles
            query_all = "SELECT Normalised_roles, roles FROM role_processor"
            self.cursor.execute(query_all)
            results = self.cursor.fetchall()
            
            for row in results:
                roles_list = json.loads(row['roles'])
                for role in roles_list:
                    if self._normalize_role_text(role) == normalized_input:
                        return row['Normalised_roles']
            
            return None
        except Error as e:
            logger.error(f"Error in exact match search: {e}")
            return None
    
    def _find_fuzzy_match(self, normalized_input: str, original_input: str, threshold: float) -> Optional[str]:
        """Find fuzzy match using similarity scoring."""
        try:
            query = "SELECT Normalised_roles, roles FROM role_processor"
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            
            best_match = None
            best_score = 0.0
            
            for row in results:
                roles_list = json.loads(row['roles'])
                normalized_role = row['Normalised_roles']
                
                for role in roles_list:
                    normalized_role_text = self._normalize_role_text(role)
                    
                    # Calculate similarity
                    similarity = SequenceMatcher(None, normalized_input, normalized_role_text).ratio()
                    
                    # Also check word overlap
                    input_words = set(normalized_input.split())
                    role_words = set(normalized_role_text.split())
                    if input_words and role_words:
                        word_overlap = len(input_words & role_words) / max(len(input_words), len(role_words))
                        # Combined score
                        combined_score = (similarity * 0.7) + (word_overlap * 0.3)
                    else:
                        combined_score = similarity
                    
                    if combined_score > best_score and combined_score >= threshold:
                        best_score = combined_score
                        best_match = normalized_role
            
            return best_match
        except Error as e:
            logger.error(f"Error in fuzzy match search: {e}")
            return None


# ============================================================================
# STANDALONE FUNCTION FOR ROLE NORMALIZATION
# ============================================================================

def normalize_role(
    original_role: Optional[str] = None,
    resume_text: Optional[str] = None,
    primary_skills: Optional[str] = None,
    secondary_skills: Optional[str] = None,
    config: Dict = None,
    fuzzy_threshold: float = 0.75
) -> str:
    """
    Standalone function to normalize a job role from resume.
    
    This is the main function to use for normalizing roles:
    1. First tries to match the original_role/designation from resume
    2. If original_role is missing/empty (fresher resume), extracts skills and infers role
    3. Matches against entries in role_processor table
    4. If match found in roles JSON array, returns corresponding Normalised_role
    5. If no match found, returns "Others"
    
    Args:
        original_role: The original role/designation from resume (optional)
        resume_text: Full resume text (optional, used if original_role is missing)
        primary_skills: Comma-separated primary/technical skills (optional)
        secondary_skills: Comma-separated secondary skills (optional)
        config: Optional database config dict. If None, uses ATSConfig defaults
        fuzzy_threshold: Similarity threshold for fuzzy matching (0.0 to 1.0, default 0.75)
        
    Returns:
        Normalized role name or "Others" if no match found
        
    Example:
        >>> # With designation
        >>> normalized = normalize_role(".NET Developer")
        >>> print(normalized)
        'Software Engineer'
        
        >>> # Fresher resume - no designation, use skills
        >>> normalized = normalize_role(
        ...     resume_text="Skills: Python, Java, React, Node.js",
        ...     primary_skills="Python, Java, React, Node.js"
        ... )
        >>> print(normalized)
        'Software Engineer'
        
        >>> # Unknown role
        >>> normalized = normalize_role("Unknown Role")
        >>> print(normalized)
        'Others'
    """
    try:
        with RoleProcessor(config=config) as rp:
            return rp.normalize_role_from_resume(
                original_role=original_role,
                resume_text=resume_text,
                primary_skills=primary_skills,
                secondary_skills=secondary_skills,
                fuzzy_threshold=fuzzy_threshold
            )
    except Exception as e:
        logger.error(f"Error in normalize_role function: {e}")
        return "Others"


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    # Example usage
    with RoleProcessor() as rp:
        # Example 1: Manual mapping
        rp.insert_role_mapping("Software Engineer", ".NET Developer")
        rp.insert_role_mapping("Software Engineer", "Java Developer")
        rp.insert_role_mapping("Software Engineer", "Python Developer")
        
        # Example 2: Get normalized role
        normalized = rp.get_normalized_role(".NET Developer")
        print(f".NET Developer -> {normalized}")
        
        # Example 3: Get all original roles for a normalized role
        original_roles = rp.get_all_original_roles("Software Engineer")
        print(f"Software Engineer includes: {original_roles}")
        
        # Example 4: Normalize role from resume (NEW)
        test_roles = [
            "backend java developer",
            "Senior .NET Developer",
            "Python Web Developer",
            "Unknown Role XYZ",
            "asp.net web developer"
        ]
        print("\n=== Role Normalization Examples ===")
        for role in test_roles:
            normalized = rp.normalize_role_from_resume(role)
            print(f"'{role}' -> '{normalized}'")
        
        # Example 5: Using standalone function
        print("\n=== Using Standalone Function ===")
        normalized = normalize_role("Java Developer")
        print(f"normalize_role('Java Developer') -> '{normalized}'")
        
        # Example 6: Auto-categorize
        original_roles_list = [
            ".NET Developer", "Java Developer", "Python Developer",
            "React Developer", "Angular Developer", "Node.js Developer"
        ]
        normalized_roles_list = [
            "Software Engineer", "Data Engineer", "DevOps Engineer",
            "QA Engineer", "Mobile Developer"
        ]
        
        stats = rp.categorize_and_insert(original_roles_list, normalized_roles_list)
        print(f"Categorization stats: {stats}")

