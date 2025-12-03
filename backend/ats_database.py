"""
Database Manager for ATS System.
Handles all MySQL operations for resumes, job descriptions, and rankings.
"""

import logging
import json
from typing import Dict, List, Any, Optional
import mysql.connector
from mysql.connector import Error
from datetime import datetime
from ats_config import ATSConfig
from profile_type_utils import (
    DEFAULT_PROFILE_TYPE,
    canonicalize_profile_type,
    canonicalize_profile_type_list,
)

logger = logging.getLogger(__name__)


class ATSDatabase:
    """MySQL database manager for ATS operations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize database manager with config."""
        self.config = config or ATSConfig.get_mysql_config()
        self.connection = None
        self.cursor = None
        self._connection_error = None
    
    def connect(self) -> bool:
        """Establish MySQL connection. Attempts to create database if it doesn't exist."""
        try:
            # First, try to connect to the database
            self.connection = mysql.connector.connect(**self.config)
            self.cursor = self.connection.cursor(dictionary=True)
            logger.info(f"Connected to MySQL database: {self.config['database']}")
            # Ensure required columns exist (role_type, subrole_type)
            self._ensure_role_columns_exist()
            return True
        except Error as e:

            error_msg = str(e)
            error_msg_lower = error_msg.lower()
            
            # Store error for better error messages
            self._connection_error = error_msg

            error_msg = str(e).lower()
            # Store error for better error messages
            self._connection_error = str(e)

            
            # Check if database doesn't exist (error 1049)
            if "1049" in str(e) or "unknown database" in error_msg_lower:
                logger.warning(f"Database '{self.config['database']}' does not exist. Attempting to create it...")
                try:
                    # Connect without specifying database
                    temp_config = self.config.copy()
                    database_name = temp_config.pop('database')
                    
                    temp_connection = mysql.connector.connect(**temp_config)
                    temp_cursor = temp_connection.cursor()
                    
                    # Create database
                    temp_cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{database_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                    temp_connection.commit()
                    temp_cursor.close()
                    temp_connection.close()
                    
                    logger.info(f"Database '{database_name}' created successfully")
                    
                    # Now try to connect again
                    self.connection = mysql.connector.connect(**self.config)
                    self.cursor = self.connection.cursor(dictionary=True)
                    logger.info(f"Connected to MySQL database: {self.config['database']}")
                    # Ensure required columns exist (role_type, subrole_type)
                    self._ensure_role_columns_exist()
                    return True
                except Error as create_error:
                    logger.error(f"Failed to create database: {create_error}")
                    logger.error(f"Please create the database manually: CREATE DATABASE {self.config['database']}")
                    self.connection = None
                    self.cursor = None
                    return False
            else:
                # Other connection errors
                logger.error(f"Error connecting to MySQL: {error_msg}")
                logger.error(f"Connection config: host={self.config.get('host')}, user={self.config.get('user')}, database={self.config.get('database')}, port={self.config.get('port')}")
                logger.error("Please check:")
                logger.error("  1. MySQL server is running")
                logger.error("  2. Database credentials are correct")
                logger.error("  3. Database exists (or run: CREATE DATABASE ats_db)")
                logger.error("  4. User has proper permissions")
                self.connection = None
                self.cursor = None
                return False
    
    def is_connected(self) -> bool:
        """Check if database is connected and cursor is available."""
        return (self.connection is not None and 
                self.cursor is not None and 
                self.connection.is_connected())
    
    def _ensure_role_columns_exist(self):
        """Ensure role_type and subrole_type columns exist in resume_metadata table."""
        try:
            # Check if role_type column exists
            self.cursor.execute("""
                SELECT COUNT(*) as col_count
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = %s 
                  AND TABLE_NAME = 'resume_metadata' 
                  AND COLUMN_NAME = 'role_type'
            """, (self.config['database'],))
            
            result = self.cursor.fetchone()
            col_count = result['col_count'] if result else 0
            
            if col_count == 0:
                logger.info("Adding missing column 'role_type' to resume_metadata table...")
                self.cursor.execute("""
                    ALTER TABLE resume_metadata 
                    ADD COLUMN role_type VARCHAR(100) COMMENT 'Role type classification' 
                    AFTER profile_type
                """)
                self.connection.commit()
                logger.info("âœ“ Column 'role_type' added successfully")
            
            # Check if subrole_type column exists
            self.cursor.execute("""
                SELECT COUNT(*) as col_count
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = %s 
                  AND TABLE_NAME = 'resume_metadata' 
                  AND COLUMN_NAME = 'subrole_type'
            """, (self.config['database'],))
            
            result = self.cursor.fetchone()
            col_count = result['col_count'] if result else 0
            
            if col_count == 0:
                logger.info("Adding missing column 'subrole_type' to resume_metadata table...")
                self.cursor.execute("""
                    ALTER TABLE resume_metadata 
                    ADD COLUMN subrole_type VARCHAR(100) COMMENT 'Sub-role type classification' 
                    AFTER role_type
                """)
                self.connection.commit()
                logger.info("âœ“ Column 'subrole_type' added successfully")
            
            # Add indexes if they don't exist
            try:
                self.cursor.execute("CREATE INDEX idx_role_type ON resume_metadata(role_type)")
                logger.debug("âœ“ Index 'idx_role_type' created")
            except Error as e:
                if "Duplicate key name" not in str(e):
                    logger.warning(f"Could not create idx_role_type: {e}")
            
            try:
                self.cursor.execute("CREATE INDEX idx_subrole_type ON resume_metadata(subrole_type)")
                logger.debug("âœ“ Index 'idx_subrole_type' created")
            except Error as e:
                if "Duplicate key name" not in str(e):
                    logger.warning(f"Could not create idx_subrole_type: {e}")
                    
        except Error as e:
            logger.warning(f"Could not verify/add role columns: {e}")
            # Don't fail connection if columns can't be added - might be permission issue
    
    def disconnect(self):
        """Close MySQL connection."""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection and self.connection.is_connected():
                self.connection.close()
                logger.info("MySQL connection closed")
        except Error as e:
            logger.error(f"Error closing MySQL connection: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        connected = self.connect()
        if not connected:
            # Fail fast so callers don't try to use a None cursor/connection
            error_detail = getattr(self, '_connection_error', 'Unknown error')
            config_info = f"host={self.config.get('host')}, user={self.config.get('user')}, database={self.config.get('database')}, port={self.config.get('port')}"
            raise RuntimeError(
                f"Failed to connect to MySQL database.\n"
                f"Error: {error_detail}\n"
                f"Config: {config_info}\n"
                f"Please check:\n"
                f"1. MySQL server is running\n"
                f"2. Database '{self.config.get('database')}' exists\n"
                f"3. User '{self.config.get('user')}' has access\n"
                f"4. Password is correct\n"
                f"5. .env file is in ATS_AI/backend/ directory"
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    # Resume Operations
    
    def insert_resume(self, resume_data: Dict[str, Any]) -> Optional[int]:
        """
        Insert resume metadata into database.
        
        Note: Embeddings are stored in Pinecone, not in the database.
        
        Args:
            resume_data: Dictionary with resume fields
        
        Returns:
            candidate_id if successful, None otherwise
        """
        try:
            
            query = """
                INSERT INTO resume_metadata (
                    name, email, phone,
                    total_experience, primary_skills, secondary_skills, all_skills, profile_type,
                    role_type, subrole_type, sub_profile_type,
                    domain, sub_domain,
                    education, education_details,
                    current_location, preferred_locations,
                    current_company, current_designation,
                    notice_period, expected_salary, current_salary,
                    resume_summary,
                    file_name, file_type, file_size_kb, file_base64,
                    status
                ) VALUES (
                    %(name)s, %(email)s, %(phone)s,
                    %(total_experience)s, %(primary_skills)s, %(secondary_skills)s, %(all_skills)s, %(profile_type)s,
                    %(role_type)s, %(subrole_type)s, %(sub_profile_type)s,
                    %(domain)s, %(sub_domain)s,
                    %(education)s, %(education_details)s,
                    %(current_location)s, %(preferred_locations)s,
                    %(current_company)s, %(current_designation)s,
                    %(notice_period)s, %(expected_salary)s, %(current_salary)s,
                    %(resume_summary)s,
                    %(file_name)s, %(file_type)s, %(file_size_kb)s, %(file_base64)s,
                    %(status)s
                )
            """
            
            # Prepare data with defaults and validate/truncate VARCHAR fields
            # Handle profile_type: if comma-separated (multi-profile), use as-is (already canonicalized)
            # Otherwise, canonicalize single profile type
            profile_type_value = resume_data.get('profile_type', DEFAULT_PROFILE_TYPE)
            if ',' in str(profile_type_value):
                # Multi-profile already formatted by format_profile_types_for_storage()
                # No need to canonicalize again, but truncate to VARCHAR(100) limit
                profile_type_final = str(profile_type_value).strip()[:100]
            else:
                # Single profile type - canonicalize it
                profile_type_final = canonicalize_profile_type(profile_type_value)
            
            # Truncate VARCHAR fields to their schema limits to prevent "Data too long" errors
            data = {
                'name': (resume_data.get('name') or '')[:255],
                'email': (resume_data.get('email') or '')[:255],
                'phone': (resume_data.get('phone') or '')[:50],
                'total_experience': resume_data.get('total_experience', 0.0),
                'primary_skills': resume_data.get('primary_skills'),
                'secondary_skills': resume_data.get('secondary_skills'),
                'all_skills': resume_data.get('all_skills'),
                'profile_type': profile_type_final,
                # Role-related fields are optional; store NULL when not provided
                'role_type': (resume_data.get('role_type') or '')[:100] if resume_data.get('role_type') else None,
                'subrole_type': (resume_data.get('subrole_type') or '')[:100] if resume_data.get('subrole_type') else None,
                'sub_profile_type': (resume_data.get('sub_profile_type') or '')[:100] if resume_data.get('sub_profile_type') else None,
                'domain': (resume_data.get('domain') or '')[:255],
                'sub_domain': (resume_data.get('sub_domain') or '')[:255],
                'education': (resume_data.get('education') or '')[:500],
                'education_details': resume_data.get('education_details'),
                'current_location': (resume_data.get('current_location') or '')[:255],
                'preferred_locations': resume_data.get('preferred_locations'),
                'current_company': (resume_data.get('current_company') or '')[:255],
                'current_designation': (resume_data.get('current_designation') or '')[:255],
                'notice_period': (resume_data.get('notice_period') or '')[:100],
                'expected_salary': (resume_data.get('expected_salary') or '')[:100],
                'current_salary': (resume_data.get('current_salary') or '')[:100],
                'resume_summary': resume_data.get('resume_summary'),
                'file_name': (resume_data.get('file_name') or '')[:500],
                'file_type': (resume_data.get('file_type') or '')[:50],
                'file_size_kb': resume_data.get('file_size_kb'),
                'file_base64': resume_data.get('file_base64'),
                'status': (resume_data.get('status') or 'active')[:50]
            }
            
            self.cursor.execute(query, data)
            self.connection.commit()
            
            candidate_id = self.cursor.lastrowid
            logger.info(f"Inserted resume with candidate_id: {candidate_id}")
            return candidate_id
            
        except Error as e:
            # Enhanced error logging to identify which column is too long
            error_msg = str(e)
            error_code = e.errno if hasattr(e, 'errno') else None
            logger.error(f"Error code: {error_code}")
            if "Data too long for column" in error_msg:
                logger.error(f"Error inserting resume: {e}")
                logger.error(f"Profile type value length: {len(str(profile_type_final))} chars, value: {profile_type_final[:100]}")
                logger.error(f"Name length: {len(str(data.get('name', '')))} chars")
                logger.error(f"Email length: {len(str(data.get('email', '')))} chars")
                logger.error(f"Phone length: {len(str(data.get('phone', '')))} chars")
                logger.error(f"File name length: {len(str(data.get('file_name', '')))} chars")
                logger.error(f"File type length: {len(str(data.get('file_type', '')))} chars")
                logger.error(f"Education length: {len(str(data.get('education', '')))} chars")
                logger.error(f"Current designation length: {len(str(data.get('current_designation', '')))} chars")
                logger.error(f"Role type: {data.get('role_type')}")
                logger.error(f"Subrole type: {data.get('subrole_type')}")
            elif error_code == 1054:
                logger.error("Unknown column error - table schema may be out of date")
                logger.error("Missing columns detected. Please run:")
                logger.error("  ALTER TABLE resume_metadata ADD COLUMN role_type VARCHAR(100) AFTER profile_type;")
                logger.error("  ALTER TABLE resume_metadata ADD COLUMN subrole_type VARCHAR(100) AFTER role_type;")
           
            else:
                logger.error(f"Full error details: {e}")
                logger.error(f"Query: {query[:500]}...")
                logger.error(f"Data keys: {list(data.keys())}")
                # Log non-None data values for debugging
                for key, value in data.items():
                    if value is not None:
                        logger.error(f"  {key}: {str(value)[:100]}")
            

            if self.connection:
                self.connection.rollback()
                # Store error for retrieval
            self.last_error = str(e)
            self.last_error_code = error_code
            return None
    
    def get_resume_by_id(self, candidate_id: int) -> Optional[Dict[str, Any]]:
        """Get resume by candidate ID."""
        try:
            if not self.is_connected():
                logger.error("Database not connected. Cannot get resume.")
                return None
            query = "SELECT * FROM resume_metadata WHERE candidate_id = %s"
            self.cursor.execute(query, (candidate_id,))
            result = self.cursor.fetchone()
            
            return result
        except Error as e:
            logger.error(f"Error fetching resume: {e}")
            return None
    
    def get_all_resumes(self, status: str = 'active', limit: int = 1000) -> List[Dict[str, Any]]:
        """Get resumes for processing/indexing, including file data when available."""
        try:
            if not self.is_connected():
                logger.error("Database not connected. Cannot get resume.")
                return []
            query = """
                SELECT 
                    candidate_id,
                    name,
                    email,
                    total_experience,
                    primary_skills,
                    domain,
                    education,
                    profile_type,
                    current_location,
                    current_company,
                    current_designation,
                    resume_summary,
                    file_name,
                    file_type,
                    file_size_kb,
                    file_base64,
                    created_at
                FROM resume_metadata
                WHERE status = %s
                ORDER BY created_at DESC
                LIMIT %s
            """
            self.cursor.execute(query, (status, limit))
            results = self.cursor.fetchall()
            return results
        except Error as e:
            logger.error(f"Error fetching resumes: {e}")
            return []
    
    def search_resumes_by_skills(self, skills: List[str], limit: int = 50) -> List[Dict[str, Any]]:
        """Search resumes by skills using FULLTEXT search."""
        try:
            skills_query = ' '.join(skills)
            query = """
                SELECT candidate_id, name, email, total_experience,
                       primary_skills, secondary_skills, domain, education,
                       MATCH(primary_skills, secondary_skills, all_skills) 
                       AGAINST(%s IN NATURAL LANGUAGE MODE) as relevance_score
                FROM resume_metadata
                WHERE MATCH(primary_skills, secondary_skills, all_skills) 
                      AGAINST(%s IN NATURAL LANGUAGE MODE)
                      AND status = 'active'
                ORDER BY relevance_score DESC
                LIMIT %s
            """
            self.cursor.execute(query, (skills_query, skills_query, limit))
            return self.cursor.fetchall()
        except Error as e:
            logger.error(f"Error searching resumes by skills: {e}")
            return []

    def filter_candidates(self, filters: Dict[str, Any], limit: int = 10000) -> List[Dict[str, Any]]:
        """
        Apply structured metadata filters before semantic/vector search.
        """
        def _listify(value):
            if value is None:
                return []
            if isinstance(value, list):
                return [v for v in value if v not in (None, '', [])]
            if isinstance(value, str):
                trimmed = value.strip()
                return [trimmed] if trimmed else []
            return [value]
        
        try:
            query = """
                SELECT 
                    candidate_id,
                    name,
                    email,
                    phone,
                    total_experience,
                    primary_skills,
                    secondary_skills,
                    all_skills,
                    profile_type,
                    domain,
                    sub_domain,
                    education,
                    education_details,
                    current_location,
                    preferred_locations,
                    current_company,
                    current_designation,
                    notice_period,
                    expected_salary,
                    current_salary,
                    resume_summary,
                    status,
                    created_at,
                    updated_at
                FROM resume_metadata
                WHERE status = %s
            """
            params = ['active']
            conditions = []
            filters = filters or {}
            
            min_experience = filters.get('min_experience')
            max_experience = filters.get('max_experience')
            if min_experience is not None:
                conditions.append("total_experience >= %s")
                params.append(float(min_experience))
            if max_experience is not None:
                conditions.append("total_experience <= %s")
                params.append(float(max_experience))
            
            education_terms = _listify(filters.get('education'))
            if education_terms:
                clauses = []
                for term in education_terms:
                    clauses.append("LOWER(education) LIKE %s")
                    params.append(f"%{term.lower()}%")
                conditions.append(f"({' OR '.join(clauses)})")
            
            domain_terms = _listify(filters.get('domain') or filters.get('domains'))
            if domain_terms:
                clauses = []
                for term in domain_terms:
                    clauses.append("LOWER(domain) LIKE %s")
                    params.append(f"%{term.lower()}%")
                conditions.append(f"({' OR '.join(clauses)})")
            
            location_terms = _listify(
                filters.get('current_location') or filters.get('location') or filters.get('locations')
            )
            if location_terms:
                clauses = []
                for term in location_terms:
                    clauses.append("LOWER(current_location) LIKE %s")
                    params.append(f"%{term.lower()}%")
                conditions.append(f"({' OR '.join(clauses)})")
            
            title_terms = _listify(filters.get('current_designation') or filters.get('job_title'))
            if title_terms:
                clauses = []
                for term in title_terms:
                    clauses.append("LOWER(current_designation) LIKE %s")
                    params.append(f"%{term.lower()}%")
                conditions.append(f"({' OR '.join(clauses)})")
            
            profile_types = canonicalize_profile_type_list(
                _listify(filters.get('profile_type') or filters.get('profile_types'))
            )
            if profile_types:
                # Support comma-separated profile types using FIND_IN_SET
                # Handle both formats: "Type1,Type2" (new) and "Type1, Type2" (old/legacy)
                # This allows matching "Microsoft Power Platform,Integration / APIs" when searching for "Microsoft Power Platform"
                clauses = []
                for pt in profile_types:
                    # Try both formats: with space and without space
                    # FIND_IN_SET works with comma-only format, so we normalize the stored value
                    clauses.append(f"(FIND_IN_SET(%s, profile_type) > 0 OR FIND_IN_SET(%s, REPLACE(profile_type, ', ', ',')) > 0)")
                    params.extend([pt, pt])
                conditions.append(f"({' OR '.join(clauses)})")
            
            skill_terms = _listify(filters.get('primary_skills') or filters.get('skills'))
            if skill_terms:
                clauses = []
                for term in skill_terms:
                    like_term = f"%{term.lower()}%"
                    clauses.append("(LOWER(primary_skills) LIKE %s OR LOWER(all_skills) LIKE %s)")
                    params.extend([like_term, like_term])
                # Require all requested skills to be present
                conditions.append(f"({' AND '.join(clauses)})")
            
            if conditions:
                query += " AND " + " AND ".join(conditions)
            
            query += " ORDER BY updated_at DESC LIMIT %s"
            params.append(limit)
            
            self.cursor.execute(query, tuple(params))
            return self.cursor.fetchall()
        except Error as e:
            logger.error(f"Error applying structured filters: {e}")
            return []
    
    def update_resume(self, candidate_id: int, updates: Dict[str, Any]) -> bool:
        """Update resume fields."""
        try:
            if not self.is_connected():
                logger.error("Database not connected. Cannot get resume.")
                return None
            if 'profile_type' in updates:
                # Handle multi-profile (comma-separated) vs single profile type
                profile_type_value = updates['profile_type']
                if ',' in str(profile_type_value):
                    # Multi-profile already formatted - use as-is
                    updates['profile_type'] = str(profile_type_value).strip()
                else:
                    # Single profile type - canonicalize it
                    updates['profile_type'] = canonicalize_profile_type(profile_type_value)
            
            # Build dynamic UPDATE query
            set_clauses = []
            values = []
            
            for key, value in updates.items():
                set_clauses.append(f"{key} = %s")
                values.append(value)
            
            if not set_clauses:
                return False
            
            query = f"UPDATE resume_metadata SET {', '.join(set_clauses)} WHERE candidate_id = %s"
            values.append(candidate_id)
            
            self.cursor.execute(query, tuple(values))
            self.connection.commit()
            
            logger.info(f"Updated resume {candidate_id}")
            return True
        except Error as e:
            logger.error(f"Error updating resume: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def delete_resume(self, candidate_id: int) -> bool:
        """Soft delete resume (set status to archived)."""
        try:
            query = "UPDATE resume_metadata SET status = 'archived' WHERE candidate_id = %s"
            self.cursor.execute(query, (candidate_id,))
            self.connection.commit()
            logger.info(f"Archived resume {candidate_id}")
            return True
        except Error as e:
            logger.error(f"Error deleting resume: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    # Job Description Operations
    
    def insert_job_description(self, jd_data: Dict[str, Any], embedding: List[float] = None) -> bool:
        """Insert job description into database."""
        try:
            embedding_json = json.dumps(embedding) if embedding else None
            
            query = """
                INSERT INTO job_descriptions (
                    job_id, job_title, job_description,
                    required_skills, preferred_skills,
                    min_experience, max_experience,
                    domain, sub_domain, education_required,
                    location, employment_type, salary_range,
                    role, sub_role, profile_type, profile_sub_type, primary_skills,
                    jd_summary, embedding, embedding_model, status
                ) VALUES (
                    %(job_id)s, %(job_title)s, %(job_description)s,
                    %(required_skills)s, %(preferred_skills)s,
                    %(min_experience)s, %(max_experience)s,
                    %(domain)s, %(sub_domain)s, %(education_required)s,
                    %(location)s, %(employment_type)s, %(salary_range)s,
                    %(role)s, %(sub_role)s, %(profile_type)s, %(profile_sub_type)s, %(primary_skills)s,
                    %(jd_summary)s, %(embedding)s, %(embedding_model)s, %(status)s
                )
                ON DUPLICATE KEY UPDATE
                    job_title = VALUES(job_title),
                    job_description = VALUES(job_description),
                    required_skills = VALUES(required_skills),
                    preferred_skills = VALUES(preferred_skills),
                    min_experience = VALUES(min_experience),
                    max_experience = VALUES(max_experience),
                    domain = VALUES(domain),
                    sub_domain = VALUES(sub_domain),
                    education_required = VALUES(education_required),
                    location = VALUES(location),
                    employment_type = VALUES(employment_type),
                    salary_range = VALUES(salary_range),
                    role = VALUES(role),
                    sub_role = VALUES(sub_role),
                    profile_type = VALUES(profile_type),
                    profile_sub_type = VALUES(profile_sub_type),
                    primary_skills = VALUES(primary_skills),
                    jd_summary = VALUES(jd_summary),
                    embedding = VALUES(embedding),
                    embedding_model = VALUES(embedding_model),
                    status = VALUES(status),
                    updated_at = CURRENT_TIMESTAMP
            """
            
            # Convert primary_skills list to comma-separated string if needed
            primary_skills = jd_data.get('primary_skills', '')
            if isinstance(primary_skills, list):
                primary_skills = ', '.join([str(s).strip() for s in primary_skills if s])
            
            data = {
                'job_id': jd_data.get('job_id'),
                'job_title': jd_data.get('job_title'),
                'job_description': jd_data.get('job_description'),
                'required_skills': jd_data.get('required_skills'),
                'preferred_skills': jd_data.get('preferred_skills'),
                'min_experience': jd_data.get('min_experience', 0.0),
                'max_experience': jd_data.get('max_experience'),
                'domain': jd_data.get('domain'),
                'sub_domain': jd_data.get('sub_domain'),
                'education_required': jd_data.get('education_required'),
                'location': jd_data.get('location'),
                'employment_type': jd_data.get('employment_type'),
                'salary_range': jd_data.get('salary_range'),
                'role': jd_data.get('role'),
                'sub_role': jd_data.get('sub_role'),
                'profile_type': jd_data.get('profile_type'),
                'profile_sub_type': jd_data.get('profile_sub_type'),
                'primary_skills': primary_skills,
                'jd_summary': jd_data.get('jd_summary'),
                'embedding': embedding_json,
                'embedding_model': jd_data.get('embedding_model', 'text-embedding-ada-002'),
                'status': jd_data.get('status', 'active')
            }
            
            self.cursor.execute(query, data)
            self.connection.commit()
            logger.info(f"Inserted job description: {jd_data.get('job_id')}")
            return True
        except Error as e:
            logger.error(f"Error inserting job description: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def update_job_description_metadata(self, job_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update extracted job metadata in job_descriptions table.
        
        Args:
            job_id: Job ID to update
            metadata: Dictionary with keys: role, sub_role, profile_type, 
                     profile_sub_type, primary_skills
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert primary_skills list to comma-separated string if needed
            primary_skills = metadata.get('primary_skills', '')
            if isinstance(primary_skills, list):
                primary_skills = ', '.join([str(s).strip() for s in primary_skills if s])
            
            query = """
                UPDATE job_descriptions SET
                    role = %(role)s,
                    sub_role = %(sub_role)s,
                    profile_type = %(profile_type)s,
                    profile_sub_type = %(profile_sub_type)s,
                    primary_skills = %(primary_skills)s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_id = %(job_id)s
            """
            
            data = {
                'job_id': job_id,
                'role': metadata.get('role'),
                'sub_role': metadata.get('sub_role'),
                'profile_type': metadata.get('profile_type'),
                'profile_sub_type': metadata.get('profile_sub_type'),
                'primary_skills': primary_skills
            }
            
            self.cursor.execute(query, data)
            self.connection.commit()
            logger.info(f"Updated job description metadata for job_id: {job_id}")
            return True
        except Error as e:
            logger.error(f"Error updating job description metadata: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def insert_job_description_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        DEPRECATED: Use update_job_description_metadata instead.
        Kept for backward compatibility.
        """
        logger.warning("insert_job_description_metadata is deprecated. Use update_job_description_metadata with job_id instead.")
        return False
    
    def insert_into_job_description(self, metadata: Dict[str, Any]) -> bool:
        """
        Insert job metadata into job_description table (singular).
        
        Args:
            metadata: Dictionary with keys: role, sub_role, profile_type, 
                     profile_sub_type, primary_skills, secondary_skills
                     Optional: job_id (string like "JD_123" - will be ignored, uses auto-increment)
        
        Returns:
            True if successful, False otherwise
        """
        query = """
            INSERT INTO job_description (
                role, sub_role, profile_type, profile_sub_type, 
                primary_skills, secondary_skills
            ) VALUES (
                %(role)s, %(sub_role)s, %(profile_type)s, 
                %(profile_sub_type)s, %(primary_skills)s, %(secondary_skills)s
            )
        """
        
        try:
            # Convert primary_skills and secondary_skills to strings if needed
            primary_skills = metadata.get('primary_skills', '')
            if isinstance(primary_skills, list):
                primary_skills = ', '.join([str(s).strip() for s in primary_skills if s])
            
            secondary_skills = metadata.get('secondary_skills', '')
            if isinstance(secondary_skills, list):
                secondary_skills = ', '.join([str(s).strip() for s in secondary_skills if s])
            
            # Always use auto-increment (job_id is INT in job_description table)
            # The job_id from metadata (string like "JD_123") is ignored
            # This table has its own auto-incrementing INT primary key
            data = {
                'role': metadata.get('role') or None,
                'sub_role': metadata.get('sub_role') or None,
                'profile_type': metadata.get('profile_type') or None,
                'profile_sub_type': metadata.get('profile_sub_type') or None,
                'primary_skills': primary_skills or None,
                'secondary_skills': secondary_skills or None
            }
            
            logger.info(f"Executing insert into job_description with data:")
            logger.info(f"  - role: {data.get('role')}")
            logger.info(f"  - profile_type: {data.get('profile_type')}")
            logger.info(f"  - primary_skills count: {len(primary_skills.split(', ')) if primary_skills else 0}")
            logger.info(f"  - secondary_skills count: {len(secondary_skills.split(', ')) if secondary_skills else 0}")
            self.cursor.execute(query, data)
            self.connection.commit()
            inserted_id = self.cursor.lastrowid
            logger.info(f"✓ Successfully inserted into job_description table with auto-increment id: {inserted_id}")
            logger.info(f"  - role: {data.get('role')}")
            logger.info(f"  - sub_role: {data.get('sub_role')}")
            logger.info(f"  - profile_type: {data.get('profile_type')}")
            logger.info(f"  - profile_sub_type: {data.get('profile_sub_type')}")
            logger.info(f"  - primary_skills: {primary_skills[:100] if primary_skills else 'None'}...")
            logger.info(f"  - secondary_skills: {secondary_skills[:100] if secondary_skills else 'None'}...")
            return True
        except Error as e:
            error_msg = str(e)
            logger.error(f"✗ Error inserting into job_description: {error_msg}")
            logger.error(f"  Query: {query}")
            logger.error(f"  Metadata received: {metadata}")
            if 'data' in locals():
                logger.error(f"  Data prepared: {data}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def get_job_description(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job description by ID."""
        try:
            query = "SELECT * FROM job_descriptions WHERE job_id = %s"
            self.cursor.execute(query, (job_id,))
            result = self.cursor.fetchone()
            
            if result and result.get('embedding'):
                result['embedding'] = json.loads(result['embedding'])
            
            return result
        except Error as e:
            logger.error(f"Error fetching job description: {e}")
            return None
    
    # Ranking Operations
    
    def insert_ranking_result(self, ranking_data: Dict[str, Any]) -> bool:
        """Insert ranking result into history."""
        try:
            query = """
                INSERT INTO ranking_history (
                    job_id, candidate_id,
                    total_score, match_percent,
                    skills_score, experience_score, domain_score, education_score,
                    matched_skills, missing_skills,
                    experience_match, domain_match,
                    rank_position, ranking_algorithm_version
                ) VALUES (
                    %(job_id)s, %(candidate_id)s,
                    %(total_score)s, %(match_percent)s,
                    %(skills_score)s, %(experience_score)s, %(domain_score)s, %(education_score)s,
                    %(matched_skills)s, %(missing_skills)s,
                    %(experience_match)s, %(domain_match)s,
                    %(rank_position)s, %(ranking_algorithm_version)s
                )
            """
            
            self.cursor.execute(query, ranking_data)
            self.connection.commit()
            return True
        except Error as e:
            logger.error(f"Error inserting ranking result: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def get_rankings_for_job(self, job_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get top rankings for a specific job."""
        try:
            query = """
                SELECT rh.*, rm.name, rm.email, rm.phone, rm.current_location
                FROM ranking_history rh
                JOIN resume_metadata rm ON rh.candidate_id = rm.candidate_id
                WHERE rh.job_id = %s
                ORDER BY rh.total_score DESC
                LIMIT %s
            """
            self.cursor.execute(query, (job_id, limit))
            return self.cursor.fetchall()
        except Error as e:
            logger.error(f"Error fetching rankings: {e}")
            return []
    
    def insert_or_update_profile_scores(self, candidate_id: int, profile_scores: Dict[str, float]) -> bool:
        """
        Insert or update profile type scores for a candidate.
        
        Args:
            candidate_id: Candidate ID
            profile_scores: Dictionary mapping profile_type -> raw_score (actual calculated values like 12, 25, 100)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.is_connected():
                logger.error("Database not connected. Cannot insert/update profile scores.")
                return False
            # Mapping from profile type names to database column names
            profile_type_to_column = {
                "Java": "java_score",
                ".Net": "dotnet_score",
                "Python": "python_score",
                "JavaScript": "javascript_score",
                "Full Stack": "fullstack_score",
                "DevOps": "devops_score",
                "Data Engineering": "data_engineering_score",
                "Data Science": "data_science_score",
                "Testing / QA": "testing_qa_score",
                "SAP": "sap_score",
                "ERP": "erp_score",
                "Cloud / Infra": "cloud_infra_score",
                "Business Intelligence (BI)": "business_intelligence_score",
                "Microsoft Power Platform": "microsoft_power_platform_score",
                "RPA": "rpa_score",
                "Cyber Security": "cyber_security_score",
                "Mobile Development": "mobile_development_score",
                "Salesforce": "salesforce_score",
                "Low Code / No Code": "low_code_no_code_score",
                "Database": "database_score",
                "Integration / APIs": "integration_apis_score",
                "UI/UX": "ui_ux_score",
                "Support": "support_score",
                "Business Development": "business_development_score",
            }
            
            # Build the INSERT ... ON DUPLICATE KEY UPDATE query
            columns = ["candidate_id"] + list(profile_type_to_column.values())
            placeholders = ["%s"] * len(columns)
            values = [candidate_id]
            
            # Add scores in the same order as columns (after candidate_id)
            for profile_type, column_name in profile_type_to_column.items():
                score = profile_scores.get(profile_type, 0.0)
                values.append(score)
            
            # Build UPDATE clause for ON DUPLICATE KEY UPDATE
            update_clauses = [f"{col} = VALUES({col})" for col in profile_type_to_column.values()]
            
            query = f"""
                INSERT INTO candidate_profile_scores (
                    {', '.join(columns)}
                ) VALUES (
                    {', '.join(placeholders)}
                )
                ON DUPLICATE KEY UPDATE
                    {', '.join(update_clauses)},
                    updated_at = CURRENT_TIMESTAMP
            """
            
            self.cursor.execute(query, values)
            self.connection.commit()
            logger.info(f"Successfully stored profile scores for candidate_id={candidate_id}")
            return True
        except Error as e:
            logger.error(f"Error inserting/updating profile scores: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def search_by_skill(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Step 1 (Primary Search): Search candidates where primary_skills contains the query.
        
        Args:
            query: Skill name to search (e.g., "Django", "Flask", "Python")
            limit: Maximum results to return
            
        Returns:
            List of matching candidates
        """
        try:
            search_query = f"%{query}%"
            sql = """
                SELECT 
                    rm.candidate_id,
                    rm.name,
                    rm.email,
                    rm.phone,
                    rm.total_experience,
                    rm.primary_skills,
                    rm.secondary_skills,
                    rm.all_skills,
                    rm.profile_type,
                    rm.domain,
                    rm.education,
                    rm.current_location,
                    rm.current_company,
                    rm.current_designation,
                    rm.resume_summary,
                    rm.created_at
                FROM resume_metadata rm
                WHERE rm.status = 'active'
                AND rm.primary_skills LIKE %s
                ORDER BY rm.total_experience DESC
                LIMIT %s
            """
            self.cursor.execute(sql, (search_query, limit))
            return self.cursor.fetchall()
        except Error as e:
            logger.error(f"Error in search_by_skill: {e}")
            return []
    
    def search_by_skill_with_score(self, query: str, score_column: str = "python_score", limit: int = 100) -> List[Dict[str, Any]]:
        """
        Step 2 (Fallback Search): Search candidates from candidate_profile_scores
        where score > 0 AND skillset contains the query.
        
        Args:
            query: Skill name to search (e.g., "Python", "Java")
            score_column: Score column to check (e.g., "python_score", "java_score")
            limit: Maximum results to return
            
        Returns:
            List of matching candidates with scores
        """
        try:
            search_query = f"%{query}%"
            sql = f"""
                SELECT 
                    rm.candidate_id,
                    rm.name,
                    rm.email,
                    rm.phone,
                    rm.total_experience,
                    rm.primary_skills,
                    rm.secondary_skills,
                    rm.all_skills,
                    rm.profile_type,
                    rm.domain,
                    rm.education,
                    rm.current_location,
                    rm.current_company,
                    rm.current_designation,
                    rm.resume_summary,
                    rm.created_at,
                    cps.{score_column} as skill_match_score
                FROM resume_metadata rm
                INNER JOIN candidate_profile_scores cps ON rm.candidate_id = cps.candidate_id
                WHERE rm.status = 'active'
                AND cps.{score_column} > 0
                AND rm.primary_skills LIKE %s
                ORDER BY cps.{score_column} DESC, rm.total_experience DESC
                LIMIT %s
            """
            self.cursor.execute(sql, (search_query, limit))
            return self.cursor.fetchall()
        except Error as e:
            logger.error(f"Error in search_by_skill_with_score: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            stats = {}
            
            # Total resumes
            self.cursor.execute("SELECT COUNT(*) as total FROM resume_metadata WHERE status = 'active'")
            stats['total_resumes'] = self.cursor.fetchone()['total']
            
            # Total job descriptions
            self.cursor.execute("SELECT COUNT(*) as total FROM job_descriptions WHERE status = 'active'")
            stats['total_jobs'] = self.cursor.fetchone()['total']
            
            # Total rankings
            self.cursor.execute("SELECT COUNT(*) as total FROM ranking_history")
            stats['total_rankings'] = self.cursor.fetchone()['total']
            
            # Average experience
            self.cursor.execute("SELECT AVG(total_experience) as avg_exp FROM resume_metadata WHERE status = 'active'")
            result = self.cursor.fetchone()
            stats['avg_experience'] = round(result['avg_exp'], 2) if result['avg_exp'] else 0
            
            return stats
        except Error as e:
            logger.error(f"Error fetching statistics: {e}")
            return {}


def create_ats_database() -> ATSDatabase:
    """Factory function to create database instance."""
    return ATSDatabase()

