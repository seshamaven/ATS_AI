#!/usr/bin/env python3
"""
Test PDF text extraction from local profiles directory
"""

import os
import sys

def extract_text_from_pdf(file_path):
    """Extract text from PDF file using available libraries"""
    try:
        # Try PyPDF2 first
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except ImportError:
            print("PyPDF2 not available, trying pdfplumber")
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                    return text.strip()
            except ImportError:
                print("pdfplumber not available, trying pymupdf")
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(file_path)
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    doc.close()
                    return text.strip()
                except ImportError:
                    print("PyMuPDF not available")
                    return f"PDF file content for {os.path.basename(file_path)} - text extraction libraries not available"
        
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return f"Error extracting text from {os.path.basename(file_path)}: {str(e)}"

def test_pdf_extraction():
    """Test PDF text extraction on all files in profiles directory"""
    
    profiles_dir = "./profiles"
    
    print("=" * 80)
    print("TESTING PDF TEXT EXTRACTION")
    print("=" * 80)
    print(f"Profiles Directory: {profiles_dir}")
    
    if not os.path.exists(profiles_dir):
        print(f"❌ Profiles directory {profiles_dir} does not exist!")
        return
    
    files = os.listdir(profiles_dir)
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
    
    print(f"📄 Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file}")
    print()
    
    for pdf_file in pdf_files:
        file_path = os.path.join(profiles_dir, pdf_file)
        file_size = os.path.getsize(file_path)
        
        print(f"Extracting text from {pdf_file} ({file_size:,} bytes)...")
        
        try:
            text = extract_text_from_pdf(file_path)
            
            if text and len(text) > 50:
                print(f"[OK] Successfully extracted {len(text)} characters")
                print(f"Preview (first 200 chars):")
                print(f"   {text[:200]}...")
                
                # Look for Python-related content
                if 'python' in text.lower():
                    print("[FOUND] Python-related content!")
                if 'experience' in text.lower():
                    print("[FOUND] Experience-related content!")
                if 'education' in text.lower() or 'degree' in text.lower():
                    print("[FOUND] Education-related content!")
                    
            else:
                print(f"[WARNING] Extracted text is short or empty: {len(text)} characters")
                print(f"Content: {text}")
                
        except Exception as e:
            print(f"Error extracting text: {e}")
        
        print()

def install_pdf_libraries():
    """Install PDF text extraction libraries"""
    
    print("=" * 80)
    print("INSTALLING PDF TEXT EXTRACTION LIBRARIES")
    print("=" * 80)
    
    libraries = [
        "PyPDF2",
        "pdfplumber", 
        "PyMuPDF",
        "python-docx"
    ]
    
    for lib in libraries:
        print(f"📦 Installing {lib}...")
        try:
            os.system(f"pip install {lib}")
            print(f"✅ {lib} installed successfully")
        except Exception as e:
            print(f"❌ Failed to install {lib}: {e}")
        print()

if __name__ == "__main__":
    print("PDF Text Extraction Test")
    print()
    
    # Check if we have PDF libraries
    try:
        import PyPDF2
        print("[OK] PyPDF2 is available")
    except ImportError:
        print("[NO] PyPDF2 not available")
    
    try:
        import pdfplumber
        print("[OK] pdfplumber is available")
    except ImportError:
        print("[NO] pdfplumber not available")
    
    try:
        import fitz
        print("[OK] PyMuPDF is available")
    except ImportError:
        print("[NO] PyMuPDF not available")
    
    print()
    
    # Test PDF extraction
    test_pdf_extraction()
    
    print("=" * 80)
    print("PDF EXTRACTION TEST COMPLETED")
    print("=" * 80)
