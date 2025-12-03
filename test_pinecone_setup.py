"""
Diagnostic script to check Pinecone setup and identify issues.
Run this before creating indexes: python test_pinecone_setup.py
"""

import os
import sys

print("=" * 60)
print("Pinecone Setup Diagnostic Tool")
print("=" * 60)
print()

# Check 1: Python version
print("1. Checking Python version...")
print(f"   Python: {sys.version}")
print()

# Check 2: Pinecone package
print("2. Checking Pinecone package installation...")
try:
    import pinecone
    print(f"   ✅ Pinecone package installed: {pinecone.__version__}")
except ImportError as e:
    print(f"   ❌ Pinecone package not installed: {e}")
    print("   Fix: pip install pinecone-client")
    sys.exit(1)
print()

# Check 3: Environment variables
print("3. Checking environment variables...")
api_key_env = os.getenv('PINECONE_API_KEY')
api_key_ats = os.getenv('ATS_PINECONE_API_KEY')

if api_key_env:
    print(f"   ✅ PINECONE_API_KEY found (length: {len(api_key_env)})")
else:
    print("   ❌ PINECONE_API_KEY not set")

if api_key_ats:
    print(f"   ✅ ATS_PINECONE_API_KEY found (length: {len(api_key_ats)})")
else:
    print("   ⚠️  ATS_PINECONE_API_KEY not set (optional)")

if not api_key_env and not api_key_ats:
    print("\n   💡 To set API key:")
    print("      Windows PowerShell: $env:PINECONE_API_KEY='your-key'")
    print("      Windows CMD: set PINECONE_API_KEY=your-key")
    print("      Linux/Mac: export PINECONE_API_KEY=your-key")
print()

# Check 4: ATSConfig
print("4. Checking ATSConfig...")
try:
    from ats_config import ATSConfig
    if hasattr(ATSConfig, 'PINECONE_API_KEY'):
        config_key = ATSConfig.PINECONE_API_KEY
        if config_key:
            print(f"   ✅ ATSConfig.PINECONE_API_KEY found (length: {len(config_key)})")
        else:
            print("   ⚠️  ATSConfig.PINECONE_API_KEY is None or empty")
    else:
        print("   ⚠️  ATSConfig.PINECONE_API_KEY attribute not found")
except Exception as e:
    print(f"   ⚠️  Could not import ATSConfig: {e}")
print()

# Check 5: .env file
print("5. Checking .env file...")
env_file = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_file):
    print(f"   ✅ .env file found: {env_file}")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("   ✅ python-dotenv installed, .env loaded")
    except ImportError:
        print("   ⚠️  python-dotenv not installed (optional)")
        print("   Fix: pip install python-dotenv")
else:
    print(f"   ⚠️  .env file not found: {env_file}")
    print("   💡 You can create .env file with: PINECONE_API_KEY=your-key")
print()

# Check 6: Test Pinecone connection
print("6. Testing Pinecone connection...")
api_key = api_key_env or api_key_ats
if api_key:
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=api_key)
        print("   ✅ Pinecone client initialized successfully")
        
        # Try to list indexes
        try:
            indexes = pc.list_indexes()
            print(f"   ✅ Successfully connected! Found {len(indexes)} existing indexes")
            if indexes:
                print("   Existing indexes:")
                for idx in indexes:
                    print(f"      - {idx.name}")
        except Exception as e:
            print(f"   ⚠️  Connected but error listing indexes: {e}")
    except Exception as e:
        print(f"   ❌ Failed to connect to Pinecone: {e}")
        print("   💡 Check your API key is correct")
else:
    print("   ⚠️  Cannot test connection - no API key found")
print()

# Summary
print("=" * 60)
print("Summary")
print("=" * 60)

issues = []
if not api_key_env and not api_key_ats:
    issues.append("❌ PINECONE_API_KEY not set")

if issues:
    print("\n⚠️  Issues found:")
    for issue in issues:
        print(f"   {issue}")
    print("\n💡 Quick fix:")
    print("   python create_java_index.py --api-key YOUR_API_KEY")
else:
    print("\n✅ All checks passed! You should be able to create indexes.")
    print("\n💡 Try running:")
    print("   python create_java_index.py")

print()

