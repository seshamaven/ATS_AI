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
            # Ensure Pinecone indexing columns exist
            self._ensure_pinecone_columns_exist()
            # Ensure Chat_history table exists
            self._ensure_chat_history_table_exists()
            return True
        except Error as e:
            error_msg = str(e)
            error_msg_lower = error_msg.lower()
            
            # Store error for better error messages
            self._connection_error = error_msg
            
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
                    # Ensure Pinecone indexing columns exist
                    self._ensure_pinecone_columns_exist()
                    # Ensure Chat_history table exists
                    self._ensure_chat_history_table_exists()
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
    
    def _ensure_pinecone_columns_exist(self):
        """Ensure pinecone_indexed and embedding_generated_at columns exist in resume_metadata table."""
        try:
            # Check if pinecone_indexed column exists
            self.cursor.execute("""
                SELECT COUNT(*) as col_count
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = %s 
                  AND TABLE_NAME = 'resume_metadata' 
                  AND COLUMN_NAME = 'pinecone_indexed'
            """, (self.config['database'],))
            
            result = self.cursor.fetchone()
            col_count = result['col_count'] if result else 0
            
            if col_count == 0:
                logger.info("Adding missing column 'pinecone_indexed' to resume_metadata table...")
                self.cursor.execute("""
                    ALTER TABLE resume_metadata 
                    ADD COLUMN pinecone_indexed BOOLEAN DEFAULT FALSE 
                    COMMENT 'Whether resume has been indexed in Pinecone'
                """)
                self.connection.commit()
                logger.info("✓ Column 'pinecone_indexed' added successfully")
            
            # Check if embedding_generated_at column exists
            self.cursor.execute("""
                SELECT COUNT(*) as col_count
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = %s 
                  AND TABLE_NAME = 'resume_metadata' 
                  AND COLUMN_NAME = 'embedding_generated_at'
            """, (self.config['database'],))
            
            result = self.cursor.fetchone()
            col_count = result['col_count'] if result else 0
            
            if col_count == 0:
                logger.info("Adding missing column 'embedding_generated_at' to resume_metadata table...")
                self.cursor.execute("""
                    ALTER TABLE resume_metadata 
                    ADD COLUMN embedding_generated_at TIMESTAMP NULL 
                    COMMENT 'Timestamp when embedding was generated and indexed in Pinecone'
                """)
                self.connection.commit()
                logger.info("✓ Column 'embedding_generated_at' added successfully")
            
            # Add index if it doesn't exist
            try:
                self.cursor.execute("CREATE INDEX idx_pinecone_indexed ON resume_metadata(pinecone_indexed)")
                logger.debug("✓ Index 'idx_pinecone_indexed' created")
            except Error as e:
                if "Duplicate key name" not in str(e):
                    logger.warning(f"Could not create idx_pinecone_indexed: {e}")
                    
        except Error as e:
            logger.warning(f"Could not verify/add Pinecone columns: {e}")
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
                    status, pinecone_indexed
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
                    %(status)s, %(pinecone_indexed)s
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
                'status': (resume_data.get('status') or 'active')[:50],
                'pinecone_indexed': resume_data.get('pinecone_indexed', False)  # Default to False - will be indexed later
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
    
    def get_resumes_by_index_status(self, indexed: bool = False, status: str = 'active', limit: int = 10000) -> List[Dict[str, Any]]:
        """
        Get resumes that are indexed or not indexed in Pinecone.
        
        Args:
            indexed: If True, returns resumes that are already indexed. If False, returns unindexed resumes.
            status: Resume status filter (default: 'active')
            limit: Maximum number of resumes to return
            
        Returns:
            List of resume dictionaries
        """
        try:
            if not self.is_connected():
                logger.error("Database not connected. Cannot get resumes.")
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
                    role_type,
                    subrole_type,
                    sub_profile_type,
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
                WHERE status = %s AND pinecone_indexed = %s
                ORDER BY created_at DESC
                LIMIT %s
            """
            self.cursor.execute(query, (status, indexed, limit))
            results = self.cursor.fetchall()
            logger.info(f"Found {len(results)} resumes with pinecone_indexed={indexed}")
            return results
        except Error as e:
            logger.error(f"Error fetching resumes by index status: {e}")
            return []
    
    def update_pinecone_index_status(self, candidate_id: int, indexed: bool = True) -> bool:
        """
        Update Pinecone indexing status for a resume.
        
        Args:
            candidate_id: Candidate ID to update
            indexed: Whether the resume is indexed (True) or not (False)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.is_connected():
                logger.error("Database not connected. Cannot update index status.")
                return False
            
            from datetime import datetime
            timestamp = datetime.now() if indexed else None
            
            query = """
                UPDATE resume_metadata 
                SET pinecone_indexed = %s,
                    embedding_generated_at = %s
                WHERE candidate_id = %s
            """
            self.cursor.execute(query, (indexed, timestamp, candidate_id))
            self.connection.commit()
            
            logger.info(f"Updated pinecone_indexed={indexed} for candidate_id={candidate_id}")
            return True
        except Error as e:
            logger.error(f"Error updating Pinecone index status: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
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
                logger.error(f"[PROFILE_SCORES] Database not connected. Cannot insert/update profile scores for candidate_id={candidate_id}.")
                return False
            
            # Validate candidate_id
            if not candidate_id or candidate_id <= 0:
                logger.error(f"[PROFILE_SCORES] Invalid candidate_id={candidate_id}. Cannot store profile scores.")
                return False
            
            # Validate profile_scores
            if not profile_scores or not isinstance(profile_scores, dict):
                logger.error(f"[PROFILE_SCORES] Invalid profile_scores for candidate_id={candidate_id}. Expected dict, got {type(profile_scores)}")
                return False
            
            logger.debug(f"[PROFILE_SCORES] Attempting to store scores for candidate_id={candidate_id} with {len(profile_scores)} profile types")
            
            # Helper function to convert profile type name to column name
            def profile_type_to_column_name(profile_type: str) -> str:
                """
                Convert profile type name to database column name.
                Rules:
                1. Handle special cases first
                2. Convert to lowercase
                3. Replace spaces with underscores
                4. Remove special characters (/, -, (), etc.)
                5. Append "_score" suffix
                
                Examples:
                - "Java" → "java_score"
                - ".Net" → "dotnet_score"
                - "Full Stack" → "fullstack_score"
                - "Cloud / Infra" → "cloud_infra_score"
                - "C/C++" → "c_cpp_score"
                - "UI/UX" → "ui_ux_score"
                - "Go / Golang" → "go_golang_score"
                """
                import re
                # Handle special cases first (before any processing)
                special_cases = {
                    ".Net": "dotnet",
                    ".NET": "dotnet",
                    "C/C++": "c_cpp",
                    "F#": "fsharp",
                    "VB.NET": "vb_net",
                    "Objective-C": "objective_c",
                    "Full Stack": "fullstack",  # Database uses fullstack_score, not full_stack_score
                }
                if profile_type in special_cases:
                    return f"{special_cases[profile_type]}_score"
                
                # Convert to lowercase
                col_name = profile_type.lower()
                # Replace spaces with underscores
                col_name = col_name.replace(' ', '_')
                # Remove special characters: /, -, (), etc., but keep underscores
                col_name = re.sub(r'[^\w]', '_', col_name)
                # Replace multiple underscores with single underscore
                col_name = re.sub(r'_+', '_', col_name)
                # Remove leading/trailing underscores
                col_name = col_name.strip('_')
                # Append "_score" suffix
                return f"{col_name}_score"
            
            # Generate mapping from PROFILE_TYPE_RULES dynamically
            # Import here to avoid circular imports
            profile_type_to_column = None  # Initialize for error handling
            try:
                from profile_type_utils import PROFILE_TYPE_RULES
                profile_type_to_column = {}
                for profile_type, _ in PROFILE_TYPE_RULES:
                    column_name = profile_type_to_column_name(profile_type)
                    profile_type_to_column[profile_type] = column_name
                logger.info(f"Generated profile_type_to_column mapping with {len(profile_type_to_column)} entries")
            except ImportError:
                logger.warning("Could not import PROFILE_TYPE_RULES, using fallback mapping")
                # Fallback to original mapping if import fails
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
            
            if not profile_type_to_column:
                logger.error(f"[PROFILE_SCORES] Failed to generate profile_type_to_column mapping for candidate_id={candidate_id}")
                return False
            
            # Check if table exists
            try:
                self.cursor.execute("SHOW TABLES LIKE 'candidate_profile_scores'")
                table_exists = self.cursor.fetchone() is not None
                if not table_exists:
                    logger.error(f"[PROFILE_SCORES] Table 'candidate_profile_scores' does not exist. Cannot store scores for candidate_id={candidate_id}.")
                    logger.error(f"[PROFILE_SCORES] Please create the table with the required columns.")
                    return False
            except Error as e:
                logger.error(f"[PROFILE_SCORES] Error checking if table exists: {e}")
                return False
            
            # Get existing columns from the table and check for PRIMARY KEY
            candidate_id_has_key = False  # Initialize outside try block for later use
            try:
                self.cursor.execute("DESCRIBE candidate_profile_scores")
                describe_rows = self.cursor.fetchall()
                
                # Handle both dictionary (dictionary=True) and tuple results
                if describe_rows and isinstance(describe_rows[0], dict):
                    # Dictionary cursor: use 'Field' key
                    existing_columns = {row['Field'] for row in describe_rows}
                else:
                    # Tuple cursor: use index 0
                    existing_columns = {row[0] for row in describe_rows}
                
                logger.debug(f"[PROFILE_SCORES] Table has {len(existing_columns)} columns")
                
                # Check if candidate_id has PRIMARY KEY or UNIQUE constraint (required for ON DUPLICATE KEY UPDATE)
                candidate_id_is_unique = False
                has_id_column = False
                
                for row in describe_rows:
                    # Handle both dictionary and tuple formats
                    if isinstance(row, dict):
                        col_name = row.get('Field', '')
                        key_info = str(row.get('Key', ''))
                    else:
                        col_name = row[0] if len(row) > 0 else ''
                        key_info = str(row[3]) if len(row) > 3 else ''
                    
                    if col_name == 'candidate_id':
                        if 'PRI' in key_info or 'UNI' in key_info:
                            candidate_id_has_key = True
                            candidate_id_is_unique = True
                    elif col_name == 'id' and 'PRI' in key_info:
                        has_id_column = True
                
                # Check for UNIQUE constraint on candidate_id separately
                if not candidate_id_is_unique:
                    try:
                        self.cursor.execute("SHOW KEYS FROM candidate_profile_scores WHERE Column_name = 'candidate_id' AND Non_unique = 0")
                        unique_keys = self.cursor.fetchall()
                        if unique_keys:
                            candidate_id_is_unique = True
                            candidate_id_has_key = True
                    except:
                        pass
                
                if not candidate_id_has_key:
                    logger.warning(f"[PROFILE_SCORES] WARNING: candidate_id column does not have PRIMARY KEY or UNIQUE constraint.")
                    logger.warning(f"[PROFILE_SCORES] ON DUPLICATE KEY UPDATE will not work properly.")
                    if has_id_column:
                        logger.warning(f"[PROFILE_SCORES] Table has 'id' as PRIMARY KEY. Consider adding UNIQUE constraint:")
                        logger.warning(f"[PROFILE_SCORES]   ALTER TABLE candidate_profile_scores ADD UNIQUE KEY unique_candidate_id (candidate_id);")
                    else:
                        logger.warning(f"[PROFILE_SCORES] Consider adding PRIMARY KEY: ALTER TABLE candidate_profile_scores ADD PRIMARY KEY (candidate_id);")
                    
                    # Try to check if there's a separate primary key constraint
                    try:
                        self.cursor.execute("SHOW KEYS FROM candidate_profile_scores WHERE Key_name = 'PRIMARY'")
                        primary_key_info = self.cursor.fetchall()
                        if primary_key_info:
                            logger.info(f"[PROFILE_SCORES] Table has PRIMARY KEY on: {[row[4] for row in primary_key_info]}")
                        else:
                            logger.error(f"[PROFILE_SCORES] No PRIMARY KEY found on candidate_profile_scores table.")
                    except:
                        pass
                else:
                    logger.debug(f"[PROFILE_SCORES] candidate_id has PRIMARY/UNIQUE constraint - ON DUPLICATE KEY UPDATE will work")
            except Error as e:
                logger.error(f"[PROFILE_SCORES] Error getting table columns: {e}")
                return False
            
            # Filter profile_type_to_column to only include columns that exist in the table
            filtered_profile_type_to_column = {}
            missing_columns = []
            
            for profile_type, column_name in profile_type_to_column.items():
                if column_name in existing_columns:
                    filtered_profile_type_to_column[profile_type] = column_name
                else:
                    missing_columns.append((profile_type, column_name))
            
            if missing_columns:
                logger.warning(f"[PROFILE_SCORES] Table missing {len(missing_columns)} columns. Skipping those profile types:")
                for profile_type, column_name in missing_columns[:10]:  # Show first 10
                    logger.warning(f"[PROFILE_SCORES]   - {profile_type} -> {column_name}")
                if len(missing_columns) > 10:
                    logger.warning(f"[PROFILE_SCORES]   ... and {len(missing_columns) - 10} more")
            
            if not filtered_profile_type_to_column:
                logger.error(f"[PROFILE_SCORES] No valid columns found in table for candidate_id={candidate_id}. Cannot store any scores.")
                return False
            
            logger.info(f"[PROFILE_SCORES] Will store scores for {len(filtered_profile_type_to_column)} profile types (out of {len(profile_type_to_column)} total)")
            
            # Build the INSERT ... ON DUPLICATE KEY UPDATE query using only existing columns
            columns = ["candidate_id"] + list(filtered_profile_type_to_column.values())
            placeholders = ["%s"] * len(columns)
            values = [candidate_id]
            
            # Add scores in the same order as columns (after candidate_id)
            for profile_type, column_name in filtered_profile_type_to_column.items():
                score = profile_scores.get(profile_type, 0.0)
                # Ensure score is a valid float
                try:
                    score = float(score) if score is not None else 0.0
                except (ValueError, TypeError):
                    logger.warning(f"[PROFILE_SCORES] Invalid score value for {profile_type}: {score}. Using 0.0")
                    score = 0.0
                values.append(score)
            
            # Build UPDATE clause for ON DUPLICATE KEY UPDATE (only for existing columns)
            update_clauses = [f"{col} = VALUES({col})" for col in filtered_profile_type_to_column.values()]
            
            # Check if row already exists for this candidate_id
            row_exists = False
            if not candidate_id_has_key:
                # If no PRIMARY/UNIQUE on candidate_id, check if row exists manually
                try:
                    self.cursor.execute("SELECT COUNT(*) FROM candidate_profile_scores WHERE candidate_id = %s", (candidate_id,))
                    row_exists = self.cursor.fetchone()[0] > 0
                    if row_exists:
                        logger.debug(f"[PROFILE_SCORES] Row already exists for candidate_id={candidate_id}, will use UPDATE")
                    else:
                        logger.debug(f"[PROFILE_SCORES] No existing row for candidate_id={candidate_id}, will use INSERT")
                except Error as e:
                    logger.warning(f"[PROFILE_SCORES] Could not check if row exists: {e}")
            
            # Build query - use ON DUPLICATE KEY UPDATE if candidate_id has key, otherwise use manual UPDATE
            if candidate_id_has_key:
                # Standard approach: ON DUPLICATE KEY UPDATE
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
            else:
                # Fallback: Check if exists, then UPDATE or INSERT
                if row_exists:
                    # Build UPDATE query
                    update_parts = [f"{col} = %s" for col in filtered_profile_type_to_column.values()]
                    update_parts.append("updated_at = CURRENT_TIMESTAMP")
                    query = f"""
                        UPDATE candidate_profile_scores
                        SET {', '.join(update_parts)}
                        WHERE candidate_id = %s
                    """
                    # Values for UPDATE: scores first, then candidate_id
                    update_values = [values[i+1] for i in range(len(filtered_profile_type_to_column))] + [candidate_id]
                    values = update_values
                    logger.debug(f"[PROFILE_SCORES] Using UPDATE query (no PRIMARY KEY on candidate_id)")
                else:
                    # Use INSERT (without ON DUPLICATE KEY UPDATE)
                    query = f"""
                        INSERT INTO candidate_profile_scores (
                            {', '.join(columns)}
                        ) VALUES (
                            {', '.join(placeholders)}
                        )
                    """
                    logger.debug(f"[PROFILE_SCORES] Using INSERT query (no PRIMARY KEY on candidate_id, row doesn't exist)")
            
            logger.debug(f"[PROFILE_SCORES] Executing query with {len(columns)} columns for candidate_id={candidate_id}")
            logger.debug(f"[PROFILE_SCORES] Columns: {columns[:5]}... (showing first 5)")
            logger.debug(f"[PROFILE_SCORES] Values count: {len(values)}, First few values: {values[:3]}...")
            
            try:
                self.cursor.execute(query, values)
                rows_affected = self.cursor.rowcount
                self.connection.commit()
                
                if rows_affected == 0:
                    logger.warning(f"[PROFILE_SCORES] Query executed but no rows affected for candidate_id={candidate_id}.")
                    if not candidate_id_has_key:
                        logger.warning(f"[PROFILE_SCORES] This might be because candidate_id doesn't have PRIMARY/UNIQUE key.")
                        logger.warning(f"[PROFILE_SCORES] Consider running: ALTER TABLE candidate_profile_scores ADD UNIQUE KEY unique_candidate_id (candidate_id);")
                else:
                    logger.info(f"[PROFILE_SCORES] Successfully stored profile scores for candidate_id={candidate_id} (rows affected: {rows_affected})")
                
                # Log summary of non-zero scores stored (only for columns we're actually storing)
                stored_non_zero = []
                for profile_type, column_name in filtered_profile_type_to_column.items():
                    score = profile_scores.get(profile_type, 0.0)
                    if score > 0:
                        stored_non_zero.append((profile_type, score, column_name))
                
                if stored_non_zero:
                    logger.info(f"[PROFILE_SCORES] Stored {len(stored_non_zero)} non-zero scores:")
                    for profile_type, score, col_name in stored_non_zero[:5]:  # Show first 5
                        logger.info(f"[PROFILE_SCORES]   - {profile_type}: {score} (column: {col_name})")
                    if len(stored_non_zero) > 5:
                        logger.info(f"[PROFILE_SCORES]   ... and {len(stored_non_zero) - 5} more")
                else:
                    logger.warning(f"[PROFILE_SCORES] No non-zero scores to store for candidate_id={candidate_id}")
                
                return True
            except Error as e:
                # Re-raise to be caught by outer exception handler
                raise
        except Error as e:
            error_code = e.errno if hasattr(e, 'errno') else None
            error_msg = str(e)
            
            logger.error(f"[PROFILE_SCORES] Database error inserting/updating profile scores for candidate_id={candidate_id}")
            logger.error(f"[PROFILE_SCORES] Error code: {error_code}, Message: {error_msg}")
            
            # Provide specific error messages for common issues
            if error_code == 1146:  # Table doesn't exist
                logger.error(f"[PROFILE_SCORES] Table 'candidate_profile_scores' does not exist. Please create it.")
            elif error_code == 1054:  # Unknown column
                logger.error(f"[PROFILE_SCORES] Column mismatch detected. The table schema may not match the expected columns.")
                # Log expected columns if available
                if profile_type_to_column:
                    logger.error(f"[PROFILE_SCORES] Expected columns: {list(profile_type_to_column.values())[:5]}... (showing first 5)")
            elif error_code == 1064:  # SQL syntax error
                logger.error(f"[PROFILE_SCORES] SQL syntax error. Query may be malformed.")
                # Query variable is not accessible here, but error message should contain details
            elif error_code == 1406:  # Data too long
                logger.error(f"[PROFILE_SCORES] Data too long for a column. Check score values.")
            
            if self.connection:
                try:
                    self.connection.rollback()
                except Error as rollback_error:
                    logger.error(f"[PROFILE_SCORES] Error during rollback: {rollback_error}")
            
            return False
        except Exception as e:
            logger.error(f"[PROFILE_SCORES] Unexpected error inserting/updating profile scores for candidate_id={candidate_id}: {e}", exc_info=True)
            if self.connection:
                try:
                    self.connection.rollback()
                except Exception as rollback_error:
                    logger.error(f"[PROFILE_SCORES] Error during rollback: {rollback_error}")
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
     # =============================================
    # Chat History Operations
    # =============================================
    
    def insert_chat_history(
        self,
        chat_msg: str,
        response: str,
        candidate_id: Optional[int] = None,
        role: Optional[str] = None,
        sub_role: Optional[str] = None,
        profile_type: Optional[str] = None,
        sub_profile_type: Optional[str] = None
    ) -> Optional[int]:
        """
        Insert a chat history record into the Chat_history table.
        
        This method saves user queries and AI responses from the search API.
        
        Args:
            chat_msg: The user's input/query message
            response: The AI/system response (usually JSON stringified)
            candidate_id: Optional candidate ID if the search relates to a specific candidate
            role: Role type classification (e.g., "Developer", "Tester")
            sub_role: Sub-role type classification (e.g., "Backend Developer")
            profile_type: Profile type from search context (e.g., "Java", ".Net")
            sub_profile_type: Sub-profile type from search context
        
        Returns:
            The inserted record's ID if successful, None otherwise
        """
        try:
            if not self.is_connected():
                logger.error("Database not connected. Cannot insert chat history.")
                return None
            
            query = """
                INSERT INTO Chat_history (
                    Candidate_id, Chat_msg, role, sub_role, 
                    profile_type, sub_profile_type, response
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s
                )
            """
            
            # Truncate fields to their schema limits
            data = (
                candidate_id,
                chat_msg[:65535] if chat_msg else '',  # TEXT limit
                (role or '')[:100] if role else None,
                (sub_role or '')[:100] if sub_role else None,
                (profile_type or '')[:100] if profile_type else None,
                (sub_profile_type or '')[:100] if sub_profile_type else None,
                response  # LONGTEXT - no limit needed
            )
            
            self.cursor.execute(query, data)
            self.connection.commit()
            
            record_id = self.cursor.lastrowid
            logger.info(f"Inserted chat history record with id: {record_id}")
            return record_id
            
        except Error as e:
            logger.error(f"Error inserting chat history: {e}")
            if self.connection:
                self.connection.rollback()
            return None
    
    def get_chat_history_by_candidate(
        self, 
        candidate_id: int, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get chat history for a specific candidate.
        
        Args:
            candidate_id: The candidate's ID
            limit: Maximum number of records to return
        
        Returns:
            List of chat history records
        """
        try:
            if not self.is_connected():
                logger.error("Database not connected. Cannot get chat history.")
                return []
            
            query = """
                SELECT 
                    id, Candidate_id, Chat_msg, role, sub_role,
                    profile_type, sub_profile_type, response, created_at
                FROM Chat_history
                WHERE Candidate_id = %s
                ORDER BY created_at DESC
                LIMIT %s
            """
            self.cursor.execute(query, (candidate_id, limit))
            return self.cursor.fetchall()
            
        except Error as e:
            logger.error(f"Error fetching chat history: {e}")
            return []
    
    def get_recent_chat_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get the most recent chat history records.
        
        Args:
            limit: Maximum number of records to return
        
        Returns:
            List of recent chat history records
        """
        try:
            if not self.is_connected():
                logger.error("Database not connected. Cannot get chat history.")
                return []
            
            query = """
                SELECT 
                    id, Candidate_id, Chat_msg, role, sub_role,
                    profile_type, sub_profile_type, response, created_at
                FROM Chat_history
                ORDER BY created_at DESC
                LIMIT %s
            """
            self.cursor.execute(query, (limit,))
            return self.cursor.fetchall()
            
        except Error as e:
            logger.error(f"Error fetching recent chat history: {e}")
            return []
    
    def _ensure_chat_history_table_exists(self):
        """Ensure the Chat_history table exists in the database."""
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS Chat_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    Candidate_id INT NULL,
                    Chat_msg TEXT NOT NULL,
                    role VARCHAR(100),
                    sub_role VARCHAR(100),
                    profile_type VARCHAR(100),
                    sub_profile_type VARCHAR(100),
                    response LONGTEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_candidate_id (Candidate_id),
                    INDEX idx_role (role),
                    INDEX idx_profile_type (profile_type),
                    INDEX idx_created_at (created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            self.connection.commit()
            logger.debug("Chat_history table verified/created")
        except Error as e:
            logger.warning(f"Could not verify/create Chat_history table: {e}")
    
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

