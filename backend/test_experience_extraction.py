#!/usr/bin/env python3
"""
Test experience extraction on sample PDF resumes
"""

import os
import sys
from typing import Dict, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def extract_text_from_pdf(file_path: str) -> str:
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
                    return f"PDF file content for {os.path.basename(file_path)} - text extraction libraries not available"
    except Exception as e:
        return f"Error extracting text from {os.path.basename(file_path)}: {str(e)}"

def test_experience_extraction():
    """Test experience extraction on provided PDF files"""
    
    # List of PDF files to test
    pdf_files = [
        "524538938.pdf",
        "524573810.pdf",
        "524691788.pdf",
        "524574773.pdf",
        "524695733.pdf",
        "524697389.pdf"
    ]
    
    print("=" * 80)
    print("TESTING EXPERIENCE EXTRACTION ON SAMPLE RESUMES")
    print("=" * 80)
    print()
    
    # Import the experience extractor
    try:
        from experience_extractor import ExperienceExtractor, extract_experience
    except ImportError as e:
        print(f"❌ Error importing ExperienceExtractor: {e}")
        return
    
    results = []
    
    for pdf_file in pdf_files:
        # Try multiple possible paths
        possible_paths = [
            pdf_file,  # Same directory
            os.path.join("backend", pdf_file),  # In backend subdirectory
            os.path.join(os.path.dirname(__file__), pdf_file),  # Relative to script
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        if not file_path:
            # Try to find the file in current directory or parent
            for root, dirs, files in os.walk('.'):
                if pdf_file in files:
                    file_path = os.path.join(root, pdf_file)
                    break
        
        if not os.path.exists(file_path):
            print(f"[WARNING] File not found: {file_path}")
            print()
            continue
        
        print(f"[TESTING] {pdf_file}")
        print("-" * 80)
        
        # Extract text from PDF
        print("Extracting text from PDF...")
        text = extract_text_from_pdf(file_path)
        
        if not text or len(text) < 50:
            print(f"[ERROR] Failed to extract text (got {len(text)} characters)")
            print()
            continue
        
        print(f"[OK] Extracted {len(text)} characters")
        
        # Extract experience
        print("Extracting experience...")
        try:
            result = extract_experience(text)
            
            total_exp = result.get('total_experience_years', 0.0)
            segments = result.get('segments', [])
            ignored = result.get('ignored', [])
            explicit_used = result.get('explicit_experience_used', False)
            
            print(f"\n[RESULTS]")
            print(f"   Total Experience: {total_exp} years")
            print(f"   Segments Found: {len(segments)}")
            if segments:
                print(f"   Date Ranges:")
                for seg in segments[:5]:  # Show first 5
                    print(f"      - {seg.get('start')} to {seg.get('end')}")
                if len(segments) > 5:
                    print(f"      ... and {len(segments) - 5} more")
            if ignored:
                print(f"   Ignored Entries: {len(ignored)}")
                for ign in ignored[:3]:  # Show first 3
                    print(f"      - {ign}")
                if len(ignored) > 3:
                    print(f"      ... and {len(ignored) - 3} more")
            if explicit_used:
                print(f"   [OK] Used explicit experience mention")
            
            # Show sample of resume text for context
            print(f"\n[Sample Resume Text] (first 300 chars):")
            try:
                sample_text = text[:300].encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                print(f"   {sample_text}...")
            except:
                print(f"   {repr(text[:300])}...")
            
            results.append({
                'file': pdf_file,
                'experience': total_exp,
                'segments': len(segments),
                'success': True
            })
            
        except Exception as e:
            print(f"[ERROR] Error extracting experience: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'file': pdf_file,
                'experience': 0.0,
                'segments': 0,
                'success': False,
                'error': str(e)
            })
        
        print()
        print("=" * 80)
        print()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'File':<30} {'Experience (years)':<20} {'Segments':<10} {'Status':<10}")
    print("-" * 80)
    for r in results:
        status = "[OK] Success" if r['success'] else f"[ERROR] {r.get('error', 'Unknown')}"
        print(f"{r['file']:<30} {r['experience']:<20.2f} {r['segments']:<10} {status:<10}")
    
    print()
    print(f"Total files tested: {len(results)}")
    print(f"Successful extractions: {sum(1 for r in results if r['success'])}")
    print(f"Average experience: {sum(r['experience'] for r in results) / len(results) if results else 0:.2f} years")

if __name__ == "__main__":
    test_experience_extraction()

