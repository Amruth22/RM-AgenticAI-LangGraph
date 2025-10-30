"""
Step-by-step installation script for RM-AgenticAI-LangGraph
This script helps install dependencies in the correct order to avoid conflicts.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 9:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.9+ required")
        return False

def install_core_dependencies():
    """Install core dependencies first."""
    core_deps = [
        "pip>=23.0",
        "setuptools>=65.0",
        "wheel>=0.38.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "python-dotenv>=1.0.0",
        "requests>=2.28.0",
        "typing-extensions>=4.5.0"
    ]
    
    print("\n📦 Installing core dependencies...")
    for dep in core_deps:
        if not run_command(f"pip install \"{dep}\"", f"Installing {dep.split('>=')[0]}"):
            return False
    return True

def install_ml_dependencies():
    """Install ML dependencies."""
    ml_deps = [
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0"
    ]
    
    print("\n🤖 Installing ML dependencies...")
    for dep in ml_deps:
        if not run_command(f"pip install \"{dep}\"", f"Installing {dep.split('>=')[0]}"):
            return False
    return True

def install_streamlit():
    """Install Streamlit."""
    print("\n🎨 Installing Streamlit...")
    return run_command("pip install streamlit>=1.29.0", "Installing Streamlit")

def install_google_ai():
    """Install Google AI dependencies."""
    print("\n🧠 Installing Google AI dependencies...")
    return run_command("pip install google-generativeai>=0.3.0", "Installing Google Generative AI")

def install_pydantic():
    """Install Pydantic with proper version."""
    print("\n📋 Installing Pydantic...")
    # Try pydantic v2 first, fallback to v1 if needed
    if not run_command("pip install \"pydantic>=2.4.0\"", "Installing Pydantic v2"):
        print("⚠️ Pydantic v2 failed, trying v1...")
        return run_command("pip install \"pydantic>=1.10.0,<2.0.0\"", "Installing Pydantic v1")
    return True

def install_langchain():
    """Install LangChain dependencies."""
    print("\n🔗 Installing LangChain dependencies...")
    
    langchain_deps = [
        "langchain-core>=0.1.0",
        "langchain>=0.1.0",
        "langchain-google-genai>=1.0.0"
    ]
    
    for dep in langchain_deps:
        if not run_command(f"pip install \"{dep}\"", f"Installing {dep.split('>=')[0]}"):
            print(f"⚠️ Failed to install {dep}, continuing...")
    
    return True

def install_langgraph():
    """Install LangGraph."""
    print("\n🕸️ Installing LangGraph...")
    return run_command("pip install langgraph>=0.0.40", "Installing LangGraph")

def install_optional_dependencies():
    """Install optional dependencies."""
    print("\n🔧 Installing optional dependencies...")
    
    optional_deps = [
        "loguru>=0.7.0",
        "aiohttp>=3.8.0",
        "markdown>=3.4.0"
    ]
    
    for dep in optional_deps:
        run_command(f"pip install \"{dep}\"", f"Installing {dep.split('>=')[0]} (optional)")
    
    return True

def create_directories():
    """Create required directories."""
    print("\n📁 Creating required directories...")
    
    directories = [
        "logs",
        "models", 
        "output",
        "data/training_data",
        "data/evaluation_data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def setup_environment():
    """Set up environment file."""
    print("\n⚙️ Setting up environment...")
    
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            try:
                with open(".env.example", "r") as src, open(".env", "w") as dst:
                    dst.write(src.read())
                print("✅ Created .env file from .env.example")
            except Exception as e:
                print(f"⚠️ Could not create .env file: {e}")
        else:
            # Create basic .env file
            env_content = """# Google AI API Key (Required)
GEMINI_API_KEY_1=your_gemini_api_key_here

# Application Settings
LOG_LEVEL=INFO
ENABLE_MONITORING=true
DEBUG_MODE=false

# Performance Settings
MAX_CONCURRENT_AGENTS=3
AGENT_TIMEOUT=300
"""
            try:
                with open(".env", "w") as f:
                    f.write(env_content)
                print("✅ Created basic .env file")
            except Exception as e:
                print(f"⚠️ Could not create .env file: {e}")
    else:
        print("✅ .env file already exists")
    
    return True

def test_installation():
    """Test the installation."""
    print("\n🧪 Testing installation...")
    
    test_imports = [
        ("streamlit", "Streamlit"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("google.generativeai", "Google Generative AI"),
        ("pydantic", "Pydantic")
    ]
    
    success_count = 0
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name} imported successfully")
            success_count += 1
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
    
    print(f"\n📊 Import test results: {success_count}/{len(test_imports)} successful")
    
    if success_count >= len(test_imports) - 1:  # Allow 1 failure
        print("🎉 Installation appears successful!")
        return True
    else:
        print("⚠️ Some imports failed. You may need to install missing dependencies manually.")
        return False

def main():
    """Main installation function."""
    print("🚀 RM-AgenticAI-LangGraph Installation Script")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Installation aborted due to Python version incompatibility")
        return False
    
    # Installation steps
    steps = [
        ("Upgrade pip and core tools", lambda: run_command("pip install --upgrade pip setuptools wheel", "Upgrading pip")),
        ("Install core dependencies", install_core_dependencies),
        ("Install ML dependencies", install_ml_dependencies),
        ("Install Streamlit", install_streamlit),
        ("Install Google AI", install_google_ai),
        ("Install Pydantic", install_pydantic),
        ("Install LangChain", install_langchain),
        ("Install LangGraph", install_langgraph),
        ("Install optional dependencies", install_optional_dependencies),
        ("Create directories", create_directories),
        ("Setup environment", setup_environment),
        ("Test installation", test_installation)
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        try:
            if not step_func():
                failed_steps.append(step_name)
                print(f"⚠️ Step '{step_name}' had issues but continuing...")
        except Exception as e:
            print(f"❌ Step '{step_name}' failed with error: {e}")
            failed_steps.append(step_name)
    
    # Summary
    print("\n" + "="*60)
    print("📊 INSTALLATION SUMMARY")
    print("="*60)
    
    if failed_steps:
        print(f"⚠️ {len(failed_steps)} steps had issues:")
        for step in failed_steps:
            print(f"  - {step}")
        print("\nYou may need to install some dependencies manually.")
    else:
        print("🎉 All installation steps completed successfully!")
    
    print("\n📝 Next Steps:")
    print("1. Edit the .env file and add your Gemini API key")
    print("2. Run: python test_system.py")
    print("3. Run: streamlit run main.py")
    
    return len(failed_steps) == 0

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Installation script failed: {e}")
        sys.exit(1)