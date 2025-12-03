"""
Migration Script: Normalize .Net Profile Type Variations
This script fixes the mismatch between "Net" and ".Net" profile types in the database.

Usage:
    python migrate_profile_type_net.py

This will update all existing records with "Net", "net", ".net", or "dotnet" 
to the canonical form ".Net".
"""

import logging
from ats_database import create_ats_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_profile_type_net():
    """Migrate all .Net profile type variations to canonical '.Net' form."""
    try:
        with create_ats_database() as db:
            # Get count before migration
            db.cursor.execute("""
                SELECT profile_type, COUNT(*) as count
                FROM resume_metadata
                WHERE LOWER(profile_type) IN ('net', '.net', 'dotnet')
                   OR LOWER(REPLACE(profile_type, '.', '')) = 'net'
                GROUP BY profile_type
            """)
            before_results = db.cursor.fetchall()
            
            logger.info("Profile types before migration:")
            for row in before_results:
                logger.info(f"  {row['profile_type']}: {row['count']} records")
            
            # Perform migration
            db.cursor.execute("""
                UPDATE resume_metadata 
                SET profile_type = '.Net' 
                WHERE LOWER(profile_type) IN ('net', '.net', 'dotnet')
                   OR LOWER(REPLACE(profile_type, '.', '')) = 'net'
            """)
            
            rows_affected = db.cursor.rowcount
            db.connection.commit()
            
            logger.info(f"\nMigration completed: {rows_affected} records updated")
            
            # Verify after migration
            db.cursor.execute("""
                SELECT profile_type, COUNT(*) as count
                FROM resume_metadata
                WHERE LOWER(profile_type) LIKE '%net%'
                GROUP BY profile_type
            """)
            after_results = db.cursor.fetchall()
            
            logger.info("\nProfile types after migration:")
            for row in after_results:
                logger.info(f"  {row['profile_type']}: {row['count']} records")
            
            # Check if migration was successful
            non_canonical = [r for r in after_results if r['profile_type'].lower() not in ('.net', 'net')]
            if non_canonical:
                logger.warning(f"Warning: Found non-canonical .Net variations: {non_canonical}")
            else:
                logger.info("✓ Migration successful! All .Net variations normalized to '.Net'")
            
            return rows_affected
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("Profile Type Migration: Normalizing .Net Variations")
    print("=" * 60)
    print()
    
    try:
        rows_updated = migrate_profile_type_net()
        print(f"\n✓ Successfully updated {rows_updated} records")
        print("\nNext steps:")
        print("1. Test a search query: 'ASP.NET' should now find matching resumes")
        print("2. Verify new resume uploads use canonical '.Net' form")
        print("3. The canonicalize_profile_type function now handles variations correctly")
    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        exit(1)

