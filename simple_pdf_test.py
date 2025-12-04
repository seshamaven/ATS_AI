#!/usr/bin/env python3
"""
Simple test script to read PDF files from local profiles directory
"""

import os
import sys

def test_profiles_directory():
    """Test reading profiles from local profiles directory"""
    
    profiles_dir = "./profiles"
    
    print("=" * 80)
    print("TESTING LOCAL PROFILES DIRECTORY")
    print("=" * 80)
    print(f"Profiles Directory: {profiles_dir}")
    
    if not os.path.exists(profiles_dir):
        print(f"ERROR: Profiles directory {profiles_dir} does not exist!")
        return False
    
    files = os.listdir(profiles_dir)
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
    
    print(f"Found {len(files)} files in directory:")
    for file in files:
        print(f"  - {file}")
    
    print(f"\nPDF Files ({len(pdf_files)}):")
    for pdf_file in pdf_files:
        file_path = os.path.join(profiles_dir, pdf_file)
        file_size = os.path.getsize(file_path)
        print(f"  - {pdf_file} ({file_size:,} bytes)")
    
    return len(pdf_files) > 0

def test_simple_pdf_extraction():
    """Test simple PDF text extraction"""
    
    profiles_dir = "./profiles"
    pdf_files = [f for f in os.listdir(profiles_dir) if f.lower().endswith('.pdf')]
    
    print("\n" + "=" * 80)
    print("TESTING PDF TEXT EXTRACTION")
    print("=" * 80)
    
    for pdf_file in pdf_files[:2]:  # Test first 2 files
        file_path = os.path.join(profiles_dir, pdf_file)
        file_size = os.path.getsize(file_path)
        
        print(f"\nTesting: {pdf_file} ({file_size:,} bytes)")
        
        # Try to extract text using PyPDF2
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                if text.strip():
                    print(f"SUCCESS: Extracted {len(text)} characters")
                    print(f"Preview: {text[:100]}...")
                    
                    # Check for keywords
                    text_lower = text.lower()
                    if 'python' in text_lower:
                        print("FOUND: Python keyword")
                    if 'experience' in text_lower:
                        print("FOUND: Experience keyword")
                    if 'education' in text_lower or 'degree' in text_lower:
                        print("FOUND: Education keyword")
                else:
                    print("WARNING: No text extracted")
                    
        except ImportError:
            print("ERROR: PyPDF2 not available")
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    print("PDF Profiles Test")
    print()
    
    # Test profiles directory
    if test_profiles_directory():
        # Test PDF extraction
        test_simple_pdf_extraction()
    else:
        print("No PDF files found")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)
