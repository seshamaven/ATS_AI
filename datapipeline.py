"""
Data Pipeline module for Regulation Library API.
Handles database connections, data fetching, and data processing operations.
"""

import logging
from typing import List, Dict, Any, Optional
import mysql.connector
from mysql.connector import Error
from datetime import date
from config import Config

# Configure logging
logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Handles MySQL database connection management."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or Config.get_mysql_config()
        self.connection = None
    
    def connect(self) -> bool:
        """Establish MySQL connection."""
        try:
            self.connection = mysql.connector.connect(**self.config)
            if self.connection.is_connected():
                logger.info("Successfully connected to MySQL database")
                return True
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return False
    
    def disconnect(self):
        """Close MySQL connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL connection closed")
    
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self.connection and self.connection.is_connected()
    
    def get_connection(self):
        """Get the database connection object."""
        return self.connection


class RegulationDataFetcher:
    """Handles fetching regulation data from the database."""
    
    def __init__(self, db_connection: DatabaseConnection = None):
        self.db_connection = db_connection or DatabaseConnection()
    
    def fetch_all_regulations(self) -> List[Dict[str, Any]]:
        """Fetch all rows from reglibrary table."""
        if not self.db_connection.is_connected():
            raise Exception("Database not connected")
        
        try:
            cursor = self.db_connection.get_connection().cursor(dictionary=True)
            query = "SELECT * FROM reglibrary"
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            logger.info(f"Fetched {len(rows)} regulations from database")
            return rows
        except Error as e:
            logger.error(f"Error fetching regulations: {e}")
            raise
    
    def fetch_regulation_by_id(self, regulation_id: int) -> Optional[Dict[str, Any]]:
        """Fetch regulation by ID."""
        if not self.db_connection.is_connected():
            raise Exception("Database not connected")
        
        try:
            cursor = self.db_connection.get_connection().cursor(dictionary=True)
            query = "SELECT * FROM reglibrary WHERE id = %s"
            cursor.execute(query, (regulation_id,))
            row = cursor.fetchone()
            cursor.close()
            return row
        except Error as e:
            logger.error(f"Error fetching regulation {regulation_id}: {e}")
            raise
    
    def fetch_regulations_by_criteria(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch regulations based on specific criteria."""
        if not self.db_connection.is_connected():
            raise Exception("Database not connected")
        
        try:
            cursor = self.db_connection.get_connection().cursor(dictionary=True)
            
            # Build dynamic query based on criteria
            where_clauses = []
            params = []
            
            for field, value in criteria.items():
                if value is not None:
                    where_clauses.append(f"{field} = %s")
                    params.append(value)
            
            if where_clauses:
                query = f"SELECT * FROM reglibrary WHERE {' AND '.join(where_clauses)}"
            else:
                query = "SELECT * FROM reglibrary"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            cursor.close()
            
            logger.info(f"Fetched {len(rows)} regulations with criteria: {criteria}")
            return rows
        except Error as e:
            logger.error(f"Error fetching regulations with criteria {criteria}: {e}")
            raise
    
    def get_regulation_count(self) -> int:
        """Get total count of regulations in the database."""
        if not self.db_connection.is_connected():
            raise Exception("Database not connected")
        
        try:
            cursor = self.db_connection.get_connection().cursor()
            query = "SELECT COUNT(*) FROM reglibrary"
            cursor.execute(query)
            count = cursor.fetchone()[0]
            cursor.close()
            logger.info(f"Total regulations count: {count}")
            return count
        except Error as e:
            logger.error(f"Error getting regulation count: {e}")
            raise


class RegulationDataProcessor:
    """Handles processing and transformation of regulation data."""
    
    @staticmethod
    def create_document_from_regulation(row: Dict[str, Any]) -> str:
        """Create a document by concatenating relevant fields from a regulation row."""
        import json
        
        document_parts = []
        
        # Add regulation text (using new field name 'Regulation')
        if row.get('Regulation'):
            document_parts.append(f"Regulation: {row['Regulation']}")
        
        # Add summary (using new field name 'Summary')
        if row.get('Summary'):
            document_parts.append(f"Summary: {row['Summary']}")
        
        # Add action items description (using new field name 'Action_Items_Description')
        if row.get('Action_Items_Description'):
            try:
                if isinstance(row['Action_Items_Description'], str):
                    action_item = json.loads(row['Action_Items_Description'])
                else:
                    action_item = row['Action_Items_Description']
                document_parts.append(f"Action Items Description: {json.dumps(action_item)}")
            except (json.JSONDecodeError, TypeError):
                document_parts.append(f"Action Items Description: {str(row['Action_Items_Description'])}")
        
        # Add action items names (using new field name 'Action_Items_Names')
        if row.get('Action_Items_Names'):
            try:
                if isinstance(row['Action_Items_Names'], str):
                    action_names = json.loads(row['Action_Items_Names'])
                else:
                    action_names = row['Action_Items_Names']
                document_parts.append(f"Action Items Names: {json.dumps(action_names)}")
            except (json.JSONDecodeError, TypeError):
                document_parts.append(f"Action Items Names: {str(row['Action_Items_Names'])}")
        
        # Add regulation subject (using new field name 'Reg_Subject')
        if row.get('Reg_Subject'):
            document_parts.append(f"Regulation Subject: {row['Reg_Subject']}")
        
        # Add previous regulation (using new field name 'Prev_Reg')
        if row.get('Prev_Reg'):
            document_parts.append(f"Previous Regulation: {row['Prev_Reg']}")
        
        return "\n\n".join(document_parts)
    
    @staticmethod
    def extract_key_fields(row: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive metadata fields from regulation row."""
        metadata = {
            'id': row.get('id'),
            'task_category': row.get('Task_Category'),
            'task_subcategory': row.get('Task_Subcategory'),
            'regulator': row.get('Regulator'),
            'regulation': row.get('Regulation', ''),
            'reg_number': row.get('Reg_Number', ''),
            'reg_date': row.get('Reg_Date'),
            'reg_category': row.get('Reg_Category', ''),
            'reg_subject': row.get('Reg_Subject', ''),
            'industry': row.get('Industry', ''),
            'sub_industry': row.get('Sub_Industry', ''),
            'activity_class': row.get('Activity_Class', ''),
            'sourced_from': row.get('Sourced_From', ''),
            'summary': row.get('Summary', ''),
            'action_items_description': row.get('Action_Items_Description', ''),
            'action_items_names': row.get('Action_Items_Names', ''),
            'prev_reg': row.get('Prev_Reg', ''),
            'due_date': row.get('Due_Date'),
            'frequency': row.get('Frequency', ''),
            'risk_category': row.get('Risk_Category'),
            'control_nature': row.get('Control_Nature'),
            'department': row.get('Department'),
            'date_created': row.get('date_created'),
            'date_modified': row.get('date_modified'),
            'effective_date': row.get('effective_date'),
            'end_date': row.get('end_date'),
            'risk_rating': row.get('risk_rating', ''),
            'active': row.get('active')
        }
        
        # Convert date objects to ISO format
        date_fields = ['due_date', 'reg_date', 'date_created', 'date_modified', 'effective_date', 'end_date']
        for field in date_fields:
            if metadata.get(field) and isinstance(metadata[field], date):
                metadata[field] = metadata[field].isoformat()
        
        # Remove None values and empty strings
        return {k: v for k, v in metadata.items() if v is not None and v != ''}
    
    @staticmethod
    def validate_regulation_data(row: Dict[str, Any]) -> bool:
        """Validate that regulation row has required fields."""
        required_fields = ['id', 'Regulation']
        
        for field in required_fields:
            if not row.get(field):
                logger.warning(f"Missing required field '{field}' in regulation row")
                return False
        
        return True


class DataPipeline:
    """Main data pipeline class that orchestrates data operations."""
    
    def __init__(self):
        self.db_connection = DatabaseConnection()
        self.data_fetcher = RegulationDataFetcher(self.db_connection)
        self.data_processor = RegulationDataProcessor()
    
    def connect(self) -> bool:
        """Connect to the database."""
        return self.db_connection.connect()
    
    def disconnect(self):
        """Disconnect from the database."""
        self.db_connection.disconnect()
    
    def get_all_regulations(self) -> List[Dict[str, Any]]:
        """Get all regulations from the database."""
        return self.data_fetcher.fetch_all_regulations()
    
    def get_regulation_by_id(self, regulation_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific regulation by ID."""
        return self.data_fetcher.fetch_regulation_by_id(regulation_id)
    
    def get_regulations_by_criteria(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get regulations based on specific criteria."""
        return self.data_fetcher.fetch_regulations_by_criteria(criteria)
    
    def get_regulation_count(self) -> int:
        """Get total count of regulations."""
        return self.data_fetcher.get_regulation_count()
    
    def process_regulation_for_embedding(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Process a regulation row for embedding creation."""
        if not self.data_processor.validate_regulation_data(row):
            raise ValueError(f"Invalid regulation data for row {row.get('id', 'unknown')}")
        
        document = self.data_processor.create_document_from_regulation(row)
        metadata = self.data_processor.extract_key_fields(row)
        print(f"Document: {document}")
        print(f"Metadata: {metadata}")
        print(f"Row ID: {row['id']}")   
        return {
            'document': document,
            'metadata': metadata,
            'row_id': row['id']
        }
    
    def get_processed_regulations(self) -> List[Dict[str, Any]]:
        """Get all regulations processed for embedding."""
        regulations = self.get_all_regulations()
        processed_regulations = []
        
        for row in regulations:
            try:
                processed = self.process_regulation_for_embedding(row)
                processed_regulations.append(processed)
            except ValueError as e:
                logger.warning(f"Skipping invalid regulation: {e}")
                continue
        
        logger.info(f"Processed {len(processed_regulations)} regulations for embedding")
        return processed_regulations


# Convenience functions for easy import
def create_data_pipeline() -> DataPipeline:
    """Create a new data pipeline instance."""
    return DataPipeline()


def get_database_connection() -> DatabaseConnection:
    """Create a new database connection instance."""
    return DatabaseConnection()


def get_data_fetcher(db_connection: DatabaseConnection = None) -> RegulationDataFetcher:
    """Create a new data fetcher instance."""
    return RegulationDataFetcher(db_connection)
