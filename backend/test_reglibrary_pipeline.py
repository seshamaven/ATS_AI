#!/usr/bin/env python3
"""
Test script to verify the updated datapipeline works with the new reglibrary table structure.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datapipeline import DataPipeline, DatabaseConnection
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_database_connection():
    """Test database connection."""
    logger.info("Testing database connection...")
    db_connection = DatabaseConnection()
    
    if db_connection.connect():
        logger.info("✓ Database connection successful")
        db_connection.disconnect()
        return True
    else:
        logger.error("✗ Database connection failed")
        return False

def test_data_pipeline():
    """Test the data pipeline with reglibrary table."""
    logger.info("Testing data pipeline...")
    
    try:
        pipeline = DataPipeline()
        
        # Test connection
        if not pipeline.connect():
            logger.error("✗ Failed to connect to database")
            return False
        
        logger.info("✓ Database connection successful")
        
        # Test getting regulation count
        try:
            count = pipeline.get_regulation_count()
            logger.info(f"✓ Total regulations count: {count}")
        except Exception as e:
            logger.error(f"✗ Error getting regulation count: {e}")
            return False
        
        # Test fetching a few regulations
        try:
            regulations = pipeline.get_all_regulations()
            logger.info(f"✓ Fetched {len(regulations)} regulations")
            
            if regulations:
                # Test processing first regulation
                first_reg = regulations[0]
                logger.info(f"✓ First regulation ID: {first_reg.get('id')}")
                logger.info(f"✓ First regulation has 'Regulation' field: {'Regulation' in first_reg}")
                logger.info(f"✓ First regulation has 'Summary' field: {'Summary' in first_reg}")
                
                # Test processing for embedding
                try:
                    processed = pipeline.process_regulation_for_embedding(first_reg)
                    logger.info("✓ Successfully processed regulation for embedding")
                    logger.info(f"✓ Document length: {len(processed['document'])}")
                    logger.info(f"✓ Metadata fields: {list(processed['metadata'].keys())}")
                except Exception as e:
                    logger.error(f"✗ Error processing regulation for embedding: {e}")
                    return False
            else:
                logger.warning("No regulations found in database")
                
        except Exception as e:
            logger.error(f"✗ Error fetching regulations: {e}")
            return False
        
        pipeline.disconnect()
        logger.info("✓ Data pipeline test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data pipeline test failed: {e}")
        return False

def test_field_mappings():
    """Test that field mappings are correct."""
    logger.info("Testing field mappings...")
    
    # Sample data structure matching the new reglibrary table
    sample_row = {
        'id': 1,
        'Task_Category': 1,
        'Task_Subcategory': 1,
        'Regulator': 1,
        'Regulation': 'Sample regulation text',
        'Reg_Number': 'REG-001',
        'Reg_Date': '2024-01-01',
        'Reg_Category': 'Compliance',
        'Reg_Subject': 'Sample subject',
        'Industry': 'Banking',
        'Sub_Industry': 'Commercial Banking',
        'Activity_Class': 'Operations',
        'Sourced_From': 'RBI',
        'Summary': 'Sample summary',
        'Action_Items_Description': 'Sample action items',
        'Action_Items_Names': 'Sample action names',
        'Prev_Reg': 'Previous regulation',
        'Due_Date': '2024-12-31',
        'Frequency': 'Monthly',
        'Risk_Category': 1,
        'Control_Nature': 1,
        'Department': 1,
        'date_created': '2024-01-01',
        'date_modified': '2024-01-01',
        'effective_date': '2024-01-01',
        'end_date': None,
        'risk_rating': 'Medium',
        'active': 1
    }
    
    try:
        from datapipeline import RegulationDataProcessor
        
        # Test document creation
        document = RegulationDataProcessor.create_document_from_regulation(sample_row)
        logger.info(f"✓ Document created successfully, length: {len(document)}")
        
        # Test metadata extraction
        metadata = RegulationDataProcessor.extract_key_fields(sample_row)
        logger.info(f"✓ Metadata extracted successfully, fields: {list(metadata.keys())}")
        
        # Test validation
        is_valid = RegulationDataProcessor.validate_regulation_data(sample_row)
        logger.info(f"✓ Validation result: {is_valid}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Field mapping test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting reglibrary pipeline tests...")
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Field Mappings", test_field_mappings),
        ("Data Pipeline", test_data_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} Test")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed! The reglibrary pipeline is working correctly.")
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")

if __name__ == "__main__":
    main()
