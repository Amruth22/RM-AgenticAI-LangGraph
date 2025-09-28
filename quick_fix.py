"""Quick fix script for common import and dependency issues."""

import subprocess
import sys
import importlib

def install_package(package_name):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        return False

def create_env_file():
    """Create .env file from example if it doesn't exist."""
    print("📝 Creating .env file...")
    
    if os.path.exists(".env"):
        print("✅ .env file already exists")
        return True
    
    if os.path.exists(".env.example"):
        try:
            import shutil
            shutil.copy(".env.example", ".env")
            print("✅ Created .env from .env.example")
            print("⚠️ Please edit .env and add your GEMINI_API_KEY_1")
            return True
        except Exception as e:
            print(f"❌ Failed to copy .env.example: {e}")
    
    # Create basic .env file
    try:
        with open(".env", "w") as f:
            f.write("""# Google AI API Key (Required)
GEMINI_API_KEY_1=your_gemini_api_key_here

# LangSmith (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=rm-agentic-ai

# Application Settings
LOG_LEVEL=INFO
ENABLE_MONITORING=true
DEBUG_MODE=false
""")
        print("✅ Created basic .env file")
        print("⚠️ Please edit .env and add your GEMINI_API_KEY_1")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def fix_missing_imports():
    """Fix missing import issues in workflow __init__.py."""
    print("🔧 Fixing missing imports...")
    
    workflow_init = "langraph_agents/workflows/__init__.py"
    try:
        with open(workflow_init, 'r') as f:
            content = f.read()
        
        # Check if already fixed
        if "# TODO: Implement these workflows" in content:
            print("✅ Workflow imports already fixed")
            return True
        
        # Fix the imports
        if "from .product_recommendation_workflow import" in content:
            content = content.replace(
                "from .product_recommendation_workflow import ProductRecommendationWorkflow",
                "# from .product_recommendation_workflow import ProductRecommendationWorkflow"
            )
            content = content.replace(
                "from .interactive_chat_workflow import InteractiveChatWorkflow",
                "# from .interactive_chat_workflow import InteractiveChatWorkflow"
            )
            content = content.replace(
                '    "ProductRecommendationWorkflow",',
                '    # "ProductRecommendationWorkflow",'
            )
            content = content.replace(
                '    "InteractiveChatWorkflow"',
                '    # "InteractiveChatWorkflow"'
            )
        
        with open(workflow_init, 'w') as f:
            f.write(content)
        
        print("✅ Fixed workflow import issues")
        return True
        
    except Exception as e:
        print(f"❌ Failed to fix imports: {e}")
        return False

def fix_pydantic_config():
    """Fix Pydantic configuration issues."""
    print("🔧 Fixing Pydantic configuration...")
    
    config_file = "config/settings.py"
    try:
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Check if the fix is already applied
        if 'extra = "ignore"' in content or '"extra": "ignore"' in content:
            print("✅ Pydantic configuration already fixed")
            return True
        
        # Apply the fix
        if 'class Config:' in content:
            content = content.replace(
                'class Config:',
                'class Config:\n        extra = "ignore"  # Allow extra fields'
            )
        
        with open(config_file, 'w') as f:
            f.write(content)
        
        print("✅ Applied Pydantic configuration fix")
        return True
        
    except Exception as e:
        print(f"❌ Failed to fix Pydantic config: {e}")
        return False

def check_and_fix_imports():
    """Check and fix common import issues."""
    print("🔍 Checking and fixing common import issues...")
    
    # First fix Pydantic config
    fix_pydantic_config()
    
    # Check pydantic-settings
    try:
        import pydantic_settings
        print("✅ pydantic-settings is available")
    except ImportError:
        print("⚠️ pydantic-settings not found, installing...")
        install_package("pydantic-settings")
    
    # Check other critical packages
    critical_packages = {
        "streamlit": "streamlit",
        "pandas": "pandas", 
        "numpy": "numpy",
        "google.generativeai": "google-generativeai",
        "pydantic": "pydantic",
        "loguru": "loguru",
        "dotenv": "python-dotenv"
    }
    
    missing_packages = []
    
    for import_name, package_name in critical_packages.items():
        try:
            importlib.import_module(import_name)
            print(f"✅ {import_name} is available")
        except ImportError:
            print(f"❌ {import_name} not found")
            missing_packages.append(package_name)
    
    # Install missing packages
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        for package in missing_packages:
            install_package(package)
    
    return len(missing_packages) == 0

def create_fallback_config():
    """Create a fallback configuration that doesn't use BaseSettings."""
    fallback_config = '''"""Fallback settings configuration without BaseSettings dependency."""

import os
from functools import lru_cache
from typing import Optional


class Settings:
    """Simple settings class without BaseSettings dependency."""
    
    def __init__(self):
        # Load from environment variables
        self.gemini_api_key = os.getenv("GEMINI_API_KEY_1", "")
        self.langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
        
        # LangSmith Configuration
        self.langchain_tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true"
        self.langchain_project = os.getenv("LANGCHAIN_PROJECT", "rm-agentic-ai")
        
        # Application Settings
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.enable_monitoring = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
        self.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        
        # Performance Settings
        self.max_concurrent_agents = int(os.getenv("MAX_CONCURRENT_AGENTS", "5"))
        self.agent_timeout = int(os.getenv("AGENT_TIMEOUT", "300"))
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
        
        # File Paths
        self.data_dir = os.getenv("DATA_DIR", "data")
        self.models_dir = os.getenv("MODELS_DIR", "models")
        self.output_dir = os.getenv("OUTPUT_DIR", "output")
        
        # Model Settings
        self.risk_model_path = "models/risk_profile_model.pkl"
        self.goal_model_path = "models/goal_success_model.pkl"
        self.risk_encoders_path = "models/label_encoders.pkl"
        self.goal_encoders_path = "models/goal_success_label_encoders.pkl"
        
        # Data Files
        self.prospects_csv = "data/input_data/prospects.csv"
        self.products_csv = "data/input_data/products.csv"
        
        # Streamlit Configuration
        self.page_title = "AI-Powered Investment Analyzer"
        self.page_icon = "🤖"
        self.layout = "wide"
        
        # Agent Configuration
        self.default_temperature = 0.1
        self.max_tokens = 4000


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Global settings instance
settings = get_settings()
'''
    
    try:
        with open("config/settings_fallback.py", "w") as f:
            f.write(fallback_config)
        print("✅ Created fallback settings configuration")
        return True
    except Exception as e:
        print(f"❌ Failed to create fallback config: {e}")
        return False

def test_basic_imports():
    """Test basic imports to see what's working."""
    print("\n🧪 Testing basic imports...")
    
    test_imports = [
        "streamlit",
        "pandas", 
        "numpy",
        "os",
        "sys",
        "json"
    ]
    
    working_imports = []
    failed_imports = []
    
    for module in test_imports:
        try:
            importlib.import_module(module)
            working_imports.append(module)
            print(f"✅ {module}")
        except ImportError as e:
            failed_imports.append((module, str(e)))
            print(f"❌ {module}: {e}")
    
    print(f"\n📊 Results: {len(working_imports)}/{len(test_imports)} imports working")
    
    if len(working_imports) >= 4:  # At least basic modules work
        print("✅ Basic Python environment is functional")
        return True
    else:
        print("❌ Basic Python environment has issues")
        return False

def main():
    """Main fix function."""
    print("🛠️ Quick Fix Script for RM-AgenticAI-LangGraph")
    print("=" * 60)
    
    # Step 1: Create .env file
    print("\n📝 Step 1: Creating .env file")
    create_env_file()
    
    # Step 2: Fix missing imports
    print("\n🔧 Step 2: Fixing missing imports")
    fix_missing_imports()
    
    # Step 3: Fix Pydantic configuration
    print("\n🔧 Step 3: Fixing Pydantic Configuration")
    fix_pydantic_config()
    
    # Test basic imports first
    if not test_basic_imports():
        print("❌ Basic Python environment issues detected")
        return False
    
    # Check and fix imports
    imports_ok = check_and_fix_imports()
    
    # Create fallback config if needed
    if not imports_ok:
        print("\n🔄 Creating fallback configuration...")
        create_fallback_config()
    
    print("\n" + "=" * 60)
    print("📝 NEXT STEPS:")
    print("=" * 60)
    
    if imports_ok:
        print("✅ All critical packages are available")
        print("1. Make sure your .env file has the GEMINI_API_KEY_1 set")
        print("2. Try running: streamlit run main.py")
    else:
        print("⚠️ Some packages are missing, but basic functionality should work")
        print("1. Install missing packages manually if needed")
        print("2. Use the fallback configuration if imports fail")
        print("3. Try running: streamlit run main.py")
    
    print("\n🆘 If you still have issues:")
    print("- Run: pip install pydantic-settings")
    print("- Run: pip install streamlit pandas numpy")
    print("- Check your Python version (3.9+ required)")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Quick fix script failed: {e}")
        print("Please install packages manually:")
        print("pip install pydantic-settings streamlit pandas numpy google-generativeai")