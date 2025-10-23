#!/usr/bin/env python3
"""
Test script to verify Azure OpenAI configuration.
Run this to check if your Azure OpenAI setup is working correctly.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_azure_openai_config():
    """Test Azure OpenAI configuration."""
    print("=" * 60)
    print("Azure OpenAI Configuration Test")
    print("=" * 60)
    
    # Check required environment variables
    required_vars = {
        'AZURE_OPENAI_API_KEY': os.getenv('AZURE_OPENAI_API_KEY'),
        'AZURE_OPENAI_ENDPOINT': os.getenv('AZURE_OPENAI_ENDPOINT'),
        'AZURE_OPENAI_API_VERSION': os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview'),
        'AZURE_OPENAI_DEPLOYMENT_NAME': os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
        'AZURE_OPENAI_MODEL': os.getenv('AZURE_OPENAI_MODEL'),
    }
    
    print("Environment Variables:")
    for var, value in required_vars.items():
        if value:
            if 'KEY' in var:
                print(f"  {var}: {'*' * 20} (configured)")
            else:
                print(f"  {var}: {value}")
        else:
            print(f"  {var}: NOT SET")
    
    print("\n" + "=" * 60)
    
    # Check if Azure OpenAI is configured
    has_azure_key = bool(required_vars['AZURE_OPENAI_API_KEY'])
    has_azure_endpoint = bool(required_vars['AZURE_OPENAI_ENDPOINT'])
    
    if has_azure_key and has_azure_endpoint:
        print("[OK] Azure OpenAI is CONFIGURED")
        print("[OK] The system will use Azure OpenAI instead of regular OpenAI")
        
        # Test imports
        try:
            from openai import AzureOpenAI
            print("[OK] AzureOpenAI import successful")
        except ImportError as e:
            print(f"[ERROR] AzureOpenAI import failed: {e}")
            return False
            
        # Test client creation
        try:
            client = AzureOpenAI(
                api_key=required_vars['AZURE_OPENAI_API_KEY'],
                api_version=required_vars['AZURE_OPENAI_API_VERSION'],
                azure_endpoint=required_vars['AZURE_OPENAI_ENDPOINT']
            )
            print("[OK] Azure OpenAI client created successfully")
        except Exception as e:
            print(f"[ERROR] Azure OpenAI client creation failed: {e}")
            return False
            
    else:
        print("[ERROR] Azure OpenAI is NOT FULLY CONFIGURED")
        print("Missing required variables:")
        if not has_azure_key:
            print("  - AZURE_OPENAI_API_KEY")
        if not has_azure_endpoint:
            print("  - AZURE_OPENAI_ENDPOINT")
        return False
    
    print("\n" + "=" * 60)
    print("Configuration Summary:")
    print("=" * 60)
    
    # Test ATS Config
    try:
        from ats_config import ATSConfig
        print("[OK] ATS Config loaded successfully")
        
        # Check if Azure is detected
        if ATSConfig.AZURE_OPENAI_ENDPOINT:
            print("[OK] ATS Config detects Azure OpenAI")
            print(f"  Endpoint: {ATSConfig.AZURE_OPENAI_ENDPOINT}")
            print(f"  Model: {ATSConfig.AZURE_OPENAI_MODEL}")
        else:
            print("[ERROR] ATS Config does not detect Azure OpenAI")
            
    except Exception as e:
        print(f"[ERROR] ATS Config test failed: {e}")
        return False
    
    # Test Config
    try:
        from config import Config
        print("[OK] Config loaded successfully")
        
        if Config.AZURE_OPENAI_ENDPOINT:
            print("[OK] Config detects Azure OpenAI")
        else:
            print("[ERROR] Config does not detect Azure OpenAI")
            
    except Exception as e:
        print(f"[ERROR] Config test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All tests passed! Azure OpenAI is ready to use.")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_azure_openai_config()
    sys.exit(0 if success else 1)
