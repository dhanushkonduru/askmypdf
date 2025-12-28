#!/usr/bin/env python3
"""
Quick setup verification script for RAGChat
Checks if all required dependencies and configurations are in place.
"""
import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed."""
    print("🔍 Checking dependencies...")
    required_packages = {
        'streamlit': 'streamlit',
        'groq': 'groq',
        'sentence_transformers': 'sentence_transformers',
        'qdrant_client': 'qdrant_client',
        'langchain_community': 'langchain_community',
        'langchain_text_splitters': 'langchain_text_splitters',
        'python-dotenv': 'dotenv'  # Package name vs import name
    }
    
    missing = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            print(f"  ❌ {package_name} - MISSING")
            missing.append(package_name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    return True

def check_env_file():
    """Check if .env file exists and has required variables."""
    print("\n🔍 Checking .env file...")
    env_path = Path(__file__).parent / '.env'
    
    if not env_path.exists():
        print("  ⚠️  .env file not found")
        print("  📝 Create a .env file with:")
        print("     GROQ_API_KEY=your_api_key_here")
        print("     QDRANT_URL=http://localhost:6333")
        return False
    
    print(f"  ✅ .env file found at {env_path}")
    
    # Load and check variables
    from dotenv import load_dotenv
    load_dotenv()
    
    groq_key = os.getenv("GROQ_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    
    print(f"  - GROQ_API_KEY: {'✅ Set' if groq_key else '❌ Missing'}")
    print(f"  - QDRANT_URL: {qdrant_url}")
    
    if not groq_key:
        print("\n  ⚠️  GROQ_API_KEY is required!")
        return False
    
    return True

def check_qdrant_connection():
    """Check if Qdrant is accessible."""
    print("\n🔍 Checking Qdrant connection...")
    try:
        from qdrant_client import QdrantClient
        from dotenv import load_dotenv
        load_dotenv()
        
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        collections = client.get_collections()
        print(f"  ✅ Connected to Qdrant at {qdrant_url}")
        print(f"  📚 Found {len(collections.collections)} collections")
        return True
    except Exception as e:
        print(f"  ⚠️  Could not connect to Qdrant: {str(e)}")
        print("  💡 Make sure Qdrant is running if using local instance")
        return False

def main():
    print("🚀 RAGChat Setup Verification\n")
    print("=" * 50)
    
    deps_ok = check_dependencies()
    env_ok = check_env_file()
    qdrant_ok = check_qdrant_connection()
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"  Dependencies: {'✅' if deps_ok else '❌'}")
    print(f"  Environment: {'✅' if env_ok else '❌'}")
    print(f"  Qdrant: {'✅' if qdrant_ok else '⚠️'}")
    
    if deps_ok and env_ok:
        print("\n✅ Setup looks good! You can run the app with:")
        print("   python -m streamlit run app.py")
        return 0
    else:
        print("\n❌ Please fix the issues above before running the app.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

