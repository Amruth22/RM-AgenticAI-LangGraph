#!/usr/bin/env python3
"""Test script to verify configuration and imports are working."""

import os
import sys
from pathlib import Path

def test_environment_setup():
    """Test if environment variables are properly set."""
    print("🔍 Testing Environment Setup")
    print("-" * 40)
    
    # Check for .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
    else:
        print("⚠️ .env file not found - creating from example")
        example_file = Path(".env.example")
        if example_file.exists():
            import shutil
            shutil.copy(".env.example", ".env")
            print("✅ Created .env from .env.example")
        else:
            print("❌ .env.example not found")
            return False
    
    # Check critical environment variables
    gemini_key = os.getenv("GEMINI_API_KEY_1")
    if gemini_key and len(gemini_key) > 10:
        print(f"✅ GEMINI_API_KEY_1 is set ({gemini_key[:10]}...)")
    else:
        print("⚠️ GEMINI_API_KEY_1 not set or invalid")
        print("   Please set your Gemini API key in the .env file")
    
    return True

def test_imports():
    """Test critical imports."""
    print("\n🧪 Testing Critical Imports")
    print("-" * 40)
    
    imports_to_test = [
        ("os", "Built-in OS module"),
        ("sys", "Built-in sys module"),
        ("pathlib", "Built-in pathlib module"),
        ("pydantic", "Pydantic for data validation"),
        ("streamlit", "Streamlit web framework"),
        ("pandas", "Pandas for data manipulation"),
        ("numpy", "NumPy for numerical computing"),
    ]
    
    failed_imports = []
    
    for module_name, description in imports_to_test:
        try:
            __import__(module_name)
            print(f"✅ {module_name}: {description}")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            failed_imports.append(module_name)
    
    if failed_imports:
        print(f"\n⚠️ {len(failed_imports)} imports failed")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print(f"\n✅ All {len(imports_to_test)} imports successful")
        return True

def test_missing_imports():
    """Test for missing import issues."""
    print("\n🔍 Testing Import Issues")
    print("-" * 40)
    
    try:
        # Test the specific import that was failing
        sys.path.insert(0, '.')
        from langraph_agents.workflows.prospect_analysis_workflow import ProspectAnalysisWorkflow
        print("✅ ProspectAnalysisWorkflow imported successfully")
        
        # Test other critical imports
        from langraph_agents.agents.risk_assessment_agent import RiskAssessmentAgent
        from langraph_agents.agents.product_specialist_agent import ProductSpecialistAgent
        print("✅ Agent imports successful")
        
        return True
        
    except ModuleNotFoundError as e:
        print(f"❌ Missing module: {e}")
        if "product_recommendation_workflow" in str(e):
            print("   This is a known issue - workflow files are missing")
            print("   Solution: Run python quick_fix.py")
        return False
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_pydantic_settings():
    """Test Pydantic settings configuration."""
    print("\n⚙️ Testing Pydantic Settings")
    print("-" * 40)
    
    try:
        # Try to import pydantic_settings
        try:
            from pydantic_settings import BaseSettings
            print("✅ pydantic_settings imported successfully")
        except ImportError:
            print("⚠️ pydantic_settings not found, trying pydantic BaseSettings")
            from pydantic import BaseSettings
        
        # Try to load our settings
        sys.path.insert(0, '.')
        from config.settings import Settings, get_settings
        
        # Test settings instantiation
        settings = get_settings()
        print("✅ Settings loaded successfully")
        print(f"   - Gemini API Key: {'Set' if settings.gemini_api_key else 'Not set'}")
        print(f"   - Log Level: {settings.log_level}")
        print(f"   - Debug Mode: {settings.debug_mode}")
        
        return True
        
    except Exception as e:
        print(f"❌ Settings loading failed: {e}")
        print("   This is the error you were experiencing")
        
        # Try to provide specific guidance
        if "extra_forbidden" in str(e):
            print("\n🔧 SOLUTION:")
            print("   The Pydantic model needs to allow extra fields.")
            print("   Run: python quick_fix.py")
            print("   Or manually add 'extra = \"ignore\"' to the Config class")
        
        return False

def test_streamlit_compatibility():
    """Test Streamlit compatibility."""
    print("\n🌐 Testing Streamlit Compatibility")
    print("-" * 40)
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        # Test if we can import our main modules
        sys.path.insert(0, '.')
        from config.settings import get_settings
        from config.logging_config import setup_logging, get_logger
        
        print("✅ Main application modules imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Streamlit compatibility test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Configuration and Import Test Suite")
    print("=" * 50)
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Critical Imports", test_imports),
        ("Missing Import Issues", test_missing_imports),
        ("Pydantic Settings", test_pydantic_settings),
        ("Streamlit Compatibility", test_streamlit_compatibility),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("You can now run: streamlit run main.py")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("1. Run: python quick_fix.py")
        print("2. Run: pip install -r requirements.txt")
        print("3. Set GEMINI_API_KEY_1 in your .env file")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)