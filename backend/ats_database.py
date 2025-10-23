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

logger = logging.getLogger(__name__)


class ATSDatabase:
    """MySQL database manager for ATS operations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize database manager with config."""
        self.config = config or ATSConfig.get_mysql_config()
        self.connection = None
        self.cursor = None
    
    def connect(self) -> bool:
        """Establish MySQL connection."""
        try:
            self.connection = mysql.connector.connect(**self.config)
            self.cursor = self.connection.cursor(dictionary=True)
            logger.info(f"Connected to MySQL database: {self.config['database']}")
            return True
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return False
    
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
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    # Resume Operations
    
    def insert_resume(self, resume_data: Dict[str, Any], embedding: List[float] = None) -> Optional[int]:
        """
        Insert resume metadata and embedding into database.
        
        Args:
            resume_data: Dictionary with resume fields
            embedding: 1536-dimension embedding vector
        
        Returns:
            candidate_id if successful, None otherwise
        """
        try:
            # Convert embedding to JSON string
            embedding_json = json.dumps(embedding) if embedding else None
            
            query = """
                INSERT INTO resume_metadata (
                    name, email, phone,
                    total_experience, primary_skills, secondary_skills, all_skills,
                    domain, sub_domain,
                    education, education_details,
                    current_location, preferred_locations,
                    current_company, current_designation,
                    notice_period, expected_salary, current_salary,
                    resume_text, resume_summary,
                    file_name, file_type, file_size_kb,
                    embedding, embedding_model,
                    status, source
                ) VALUES (
                    %(name)s, %(email)s, %(phone)s,
                    %(total_experience)s, %(primary_skills)s, %(secondary_skills)s, %(all_skills)s,
                    %(domain)s, %(sub_domain)s,
                    %(education)s, %(education_details)s,
                    %(current_location)s, %(preferred_locations)s,
                    %(current_company)s, %(current_designation)s,
                    %(notice_period)s, %(expected_salary)s, %(current_salary)s,
                    %(resume_text)s, %(resume_summary)s,
                    %(file_name)s, %(file_type)s, %(file_size_kb)s,
                    %(embedding)s, %(embedding_model)s,
                    %(status)s, %(source)s
                )
            """
            
            # Prepare data with defaults
            data = {
                'name': resume_data.get('name'),
                'email': resume_data.get('email'),
                'phone': resume_data.get('phone'),
                'total_experience': resume_data.get('total_experience', 0.0),
                'primary_skills': resume_data.get('primary_skills'),
                'secondary_skills': resume_data.get('secondary_skills'),
                'all_skills': resume_data.get('all_skills'),
                'domain': resume_data.get('domain'),
                'sub_domain': resume_data.get('sub_domain'),
                'education': resume_data.get('education'),
                'education_details': resume_data.get('education_details'),
                'current_location': resume_data.get('current_location'),
                'preferred_locations': resume_data.get('preferred_locations'),
                'current_company': resume_data.get('current_company'),
                'current_designation': resume_data.get('current_designation'),
                'notice_period': resume_data.get('notice_period'),
                'expected_salary': resume_data.get('expected_salary'),
                'current_salary': resume_data.get('current_salary'),
                'resume_text': resume_data.get('resume_text'),
                'resume_summary': resume_data.get('resume_summary'),
                'file_name': resume_data.get('file_name'),
                'file_type': resume_data.get('file_type'),
                'file_size_kb': resume_data.get('file_size_kb'),
                'embedding': embedding_json,
                'embedding_model': resume_data.get('embedding_model', 'text-embedding-ada-002'),
                'status': resume_data.get('status', 'active'),
                'source': resume_data.get('source', 'api')
            }
            
            self.cursor.execute(query, data)
            self.connection.commit()
            
            candidate_id = self.cursor.lastrowid
            logger.info(f"Inserted resume with candidate_id: {candidate_id}")
            return candidate_id
            
        except Error as e:
            logger.error(f"Error inserting resume: {e}")
            if self.connection:
                self.connection.rollback()
            return None
    
    def get_resume_by_id(self, candidate_id: int) -> Optional[Dict[str, Any]]:
        """Get resume by candidate ID."""
        try:
            query = "SELECT * FROM resume_metadata WHERE candidate_id = %s"
            self.cursor.execute(query, (candidate_id,))
            result = self.cursor.fetchone()
            
            if result and result.get('embedding'):
                # Parse embedding from JSON
                result['embedding'] = json.loads(result['embedding'])
            
            return result
        except Error as e:
            logger.error(f"Error fetching resume: {e}")
            return None
    
    def get_all_resumes(self, status: str = 'active', limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all resumes with optional filtering."""
        try:
            query = """
                SELECT candidate_id, name, email, total_experience, 
                       primary_skills, domain, education, embedding
                FROM resume_metadata 
                WHERE status = %s
                LIMIT %s
            """
            self.cursor.execute(query, (status, limit))
            results = self.cursor.fetchall()
            
            # Parse embeddings
            for result in results:
                if result.get('embedding'):
                    result['embedding'] = json.loads(result['embedding'])
            
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
    
    def update_resume(self, candidate_id: int, updates: Dict[str, Any]) -> bool:
        """Update resume fields."""
        try:
            # Build dynamic UPDATE query
            set_clauses = []
            values = []
            
            for key, value in updates.items():
                if key == 'embedding' and isinstance(value, list):
                    value = json.dumps(value)
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
                    jd_summary, embedding, embedding_model, status
                ) VALUES (
                    %(job_id)s, %(job_title)s, %(job_description)s,
                    %(required_skills)s, %(preferred_skills)s,
                    %(min_experience)s, %(max_experience)s,
                    %(domain)s, %(sub_domain)s, %(education_required)s,
                    %(location)s, %(employment_type)s, %(salary_range)s,
                    %(jd_summary)s, %(embedding)s, %(embedding_model)s, %(status)s
                )
            """
            
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

