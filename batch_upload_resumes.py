"""
Batch Resume Upload Script
Recursively scans directories for resume files and uploads them to the ATS API.
"""

import os
import sys
import requests
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
import logging

# Configure logging with Windows console encoding fix
import sys

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    # Use UTF-8 encoding for console output
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_upload.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Resume file extensions
RESUME_EXTENSIONS = {'.pdf', '.doc', '.docx', '.docu'}

# Default API endpoint (can be overridden)
DEFAULT_API_URL = 'http://127.0.0.1:8000/api/upload_resume'
DEFAULT_API_URL_ALT = 'http://localhost:5002/api/processResume'  # Fallback to existing endpoint


def find_resume_files(root_dir: str) -> List[Path]:
    """
    Recursively find all resume files in the directory tree.
    
    Args:
        root_dir: Root directory to scan
        
    Returns:
        List of Path objects for resume files
    """
    resume_files = []
    root_path = Path(root_dir)
    
    if not root_path.exists():
        logger.error(f"Directory does not exist: {root_dir}")
        return resume_files
    
    logger.info(f"Scanning directory: {root_dir}")
    
    # Recursively walk through all subdirectories
    for file_path in root_path.rglob('*'):
        if file_path.is_file():
            # Check if file extension matches resume extensions
            if file_path.suffix.lower() in RESUME_EXTENSIONS:
                resume_files.append(file_path)
                logger.debug(f"Found resume: {file_path}")
    
    logger.info(f"Found {len(resume_files)} resume file(s)")
    return resume_files


def upload_resume(file_path: Path, api_url: str) -> Tuple[bool, str]:
    """
    Upload a single resume file to the API.
    
    Args:
        file_path: Path to the resume file
        api_url: API endpoint URL
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        logger.info(f"Uploading: {file_path}")
        
        # Open file in binary mode
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f, 'application/octet-stream')}
            
            # Make POST request
            response = requests.post(
                api_url,
                files=files,
                timeout=60  # 60 second timeout
            )
        
        # Check response
        if response.status_code == 200:
            try:
                result = response.json()
                candidate_id = result.get('candidate_id', 'N/A')
                logger.info(f"[OK] Success: {file_path.name} (Candidate ID: {candidate_id})")
                return True, f"Success - Candidate ID: {candidate_id}"
            except ValueError:
                logger.info(f"[OK] Success: {file_path.name} (Status: {response.status_code})")
                return True, f"Success - Status: {response.status_code}"
        else:
            error_msg = f"Failed with status {response.status_code}"
            try:
                error_data = response.json()
                error_msg = error_data.get('error', error_msg)
            except ValueError:
                error_msg = response.text[:200] if response.text else error_msg
            
            logger.error(f"[FAIL] Failed: {file_path.name} - {error_msg}")
            return False, error_msg
            
    except requests.exceptions.Timeout:
        error_msg = "Request timeout"
        logger.error(f"[FAIL] Failed: {file_path.name} - {error_msg}")
        return False, error_msg
    except requests.exceptions.ConnectionError:
        error_msg = "Connection error - API may not be running"
        logger.error(f"[FAIL] Failed: {file_path.name} - {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"[FAIL] Failed: {file_path.name} - {error_msg}")
        return False, error_msg


def check_api_health(api_url: str) -> bool:
    """
    Check if the API is running and accessible.
    
    Args:
        api_url: API endpoint URL
        
    Returns:
        True if API is accessible, False otherwise
    """
    try:
        # Try health endpoint (common pattern)
        base_url = api_url.rsplit('/api/', 1)[0]
        health_url = f"{base_url}/health"
        
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            logger.info("[OK] API is running and accessible")
            return True
    except Exception:
        pass
    
    # If health check fails, try the actual endpoint with a HEAD request
    try:
        response = requests.head(api_url, timeout=5)
        return True
    except Exception:
        logger.warning("Could not verify API health, proceeding anyway...")
        return True  # Proceed anyway - might be a different API structure


def batch_upload_resumes(
    root_dir: str,
    api_url: str = None,
    dry_run: bool = False
) -> dict:
    """
    Main function to batch upload resumes.
    
    Args:
        root_dir: Root directory to scan
        api_url: API endpoint URL (defaults to DEFAULT_API_URL)
        dry_run: If True, only scan files without uploading
        
    Returns:
        Dictionary with upload statistics
    """
    # Validate API URL format
    if api_url and not (api_url.startswith('http://') or api_url.startswith('https://')):
        logger.error(f"Invalid API URL format: {api_url}")
        logger.error("API URL must start with http:// or https://")
        logger.info(f"Using fallback API URL: {DEFAULT_API_URL_ALT}")
        api_url = DEFAULT_API_URL_ALT
    
    # Determine API URL
    if api_url is None:
        # Try default, then fallback
        api_url = DEFAULT_API_URL
        logger.info(f"Using API URL: {api_url}")
        
        # Check if default URL works, try fallback
        if not check_api_health(api_url):
            logger.info(f"Trying fallback API URL: {DEFAULT_API_URL_ALT}")
            api_url = DEFAULT_API_URL_ALT
    
    if not dry_run:
        logger.info(f"Checking API health at: {api_url}")
        if not check_api_health(api_url):
            logger.warning("API health check failed, but proceeding...")
    
    # Find all resume files
    resume_files = find_resume_files(root_dir)
    
    if not resume_files:
        logger.warning("No resume files found!")
        return {
            'total': 0,
            'success': 0,
            'failed': 0,
            'results': []
        }
    
    # Statistics
    stats = {
        'total': len(resume_files),
        'success': 0,
        'failed': 0,
        'results': []
    }
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting batch upload: {stats['total']} file(s)")
    logger.info(f"{'='*60}\n")
    
    # Upload each file
    for idx, file_path in enumerate(resume_files, 1):
        logger.info(f"[{idx}/{stats['total']}] Processing: {file_path}")
        
        if dry_run:
            logger.info(f"  [DRY RUN] Would upload: {file_path}")
            stats['results'].append({
                'file': str(file_path),
                'success': None,
                'message': 'Dry run - not uploaded'
            })
        else:
            success, message = upload_resume(file_path, api_url)
            
            stats['results'].append({
                'file': str(file_path),
                'success': success,
                'message': message
            })
            
            if success:
                stats['success'] += 1
            else:
                stats['failed'] += 1
        
        logger.info("")  # Blank line for readability
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("UPLOAD SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total files: {stats['total']}")
    if not dry_run:
        logger.info(f"Successful: {stats['success']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Success rate: {(stats['success']/stats['total']*100):.1f}%")
    logger.info(f"{'='*60}\n")
    
    return stats


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch upload resumes to ATS API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload all resumes from uploads/ directory
  python batch_upload_resumes.py uploads/
  
  # Upload with custom API URL
  python batch_upload_resumes.py uploads/ --api-url http://localhost:5002/api/processResume
  
  # Dry run (scan only, don't upload)
  python batch_upload_resumes.py uploads/ --dry-run
        """
    )
    
    parser.add_argument(
        'directory',
        type=str,
        help='Root directory to scan for resume files'
    )
    
    parser.add_argument(
        '--api-url',
        type=str,
        default=None,
        help=f'API endpoint URL (default: {DEFAULT_API_URL})'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Scan files without uploading (for testing)'
    )
    
    args = parser.parse_args()
    
    # Run batch upload
    stats = batch_upload_resumes(
        root_dir=args.directory,
        api_url=args.api_url,
        dry_run=args.dry_run
    )
    
    # Exit with error code if any uploads failed
    if stats['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()

