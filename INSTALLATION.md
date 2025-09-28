# 🚀 Installation & Setup Guide

## RM-AgenticAI-LangGraph Installation

This guide will help you set up the advanced AI-powered investment analyzer with LangGraph multi-agent system.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 2GB free space
- **OS**: Windows 10+, macOS 10.14+, or Linux

### Required API Keys
- **Google Gemini API Key**: For AI-powered analysis
- **LangSmith API Key** (Optional): For monitoring and debugging

## 🛠️ Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Amruth22/RM-AgenticAI-LangGraph.git
cd RM-AgenticAI-LangGraph
```

### 2. Create Virtual Environment

**Using venv (Recommended):**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

**Using conda:**
```bash
conda create -n rm-agentic python=3.9
conda activate rm-agentic
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**If you encounter issues, try upgrading pip first:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

**Copy the environment template:**
```bash
cp .env.example .env
```

**Edit the `.env` file with your API keys:**
```env
# Required: Google AI API Key
GEMINI_API_KEY_1=your_gemini_api_key_here

# Optional: LangSmith for monitoring
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=rm-agentic-ai

# Application Settings
LOG_LEVEL=INFO
ENABLE_MONITORING=true
DEBUG_MODE=false
```

### 5. Create Required Directories

```bash
mkdir -p logs
mkdir -p models
mkdir -p output
mkdir -p data/training_data
mkdir -p data/evaluation_data
```

## 🔑 API Key Setup

### Google Gemini API Key

1. **Visit Google AI Studio**: https://makersuite.google.com/app/apikey
2. **Create a new API key** or use an existing one
3. **Copy the API key** and paste it in your `.env` file
4. **Test the connection** (optional):

```python
import google.generativeai as genai
genai.configure(api_key="your_api_key_here")
model = genai.GenerativeModel("gemini-2.0-flash")
response = model.generate_content("Hello, world!")
print(response.text)
```

### LangSmith API Key (Optional)

1. **Visit LangSmith**: https://smith.langchain.com/
2. **Create an account** and get your API key
3. **Add to `.env` file** for advanced monitoring

## 🧪 Verify Installation

### 1. Test Basic Setup

```bash
python -c "
import streamlit as st
import pandas as pd
from langraph_agents.workflows.prospect_analysis_workflow import ProspectAnalysisWorkflow
print('✅ All imports successful!')
"
```

### 2. Test API Connection

```bash
python -c "
from config.settings import get_settings
settings = get_settings()
print(f'✅ Settings loaded: {settings.gemini_api_key[:10]}...')
"
```

### 3. Run the Application

```bash
streamlit run main.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

## 🎯 First Run

### 1. Access the Application
Open your browser and navigate to `http://localhost:8501`

### 2. Test with Sample Data
- Select a prospect from the dropdown
- Click "🚀 Start AI Analysis"
- Wait for the multi-agent analysis to complete
- Review the results in different tabs

### 3. Verify Agent Performance
- Check the "🤖 Agent Performance" tab
- Ensure all agents executed successfully
- Review execution times and success rates

## 🔧 Configuration Options

### Application Settings

Edit `config/settings.py` or use environment variables:

```env
# Performance Settings
MAX_CONCURRENT_AGENTS=5
AGENT_TIMEOUT=300
CACHE_TTL=3600

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# Model Settings
DEFAULT_TEMPERATURE=0.1
MAX_TOKENS=4000
```

### Streamlit Configuration

Create `.streamlit/config.toml`:

```toml
[server]
port = 8501
headless = true

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

## 📊 Adding Your Own Data

### 1. Prospects Data

Replace `data/input_data/prospects.csv` with your data:

```csv
prospect_id,name,age,annual_income,current_savings,target_goal_amount,investment_horizon_years,number_of_dependents,investment_experience_level,investment_goal
P001,John Doe,35,800000,500000,2000000,10,2,Intermediate,Retirement Planning
```

### 2. Products Catalog

Update `data/input_data/products.csv`:

```csv
product_id,product_name,product_type,risk_level,min_investment,expected_return,expense_ratio,category,description
MF001,Growth Fund,Mutual Fund,High,5000,12-15%,1.2%,Equity,High-growth equity fund
```

### 3. ML Models (Optional)

If you have trained models, place them in the `models/` directory:
- `risk_profile_model.pkl`
- `goal_success_model.pkl`
- `label_encoders.pkl`
- `goal_success_label_encoders.pkl`

## 🐛 Troubleshooting

### Common Issues

**1. Pydantic Validation Error (extra_forbidden)**
```bash
# Error: pydantic_core._pydantic_core.ValidationError: Extra inputs are not permitted
# Solution: Run the quick fix script
python quick_fix.py

# Or run the test script to diagnose
python test_config.py

# Manual fix: Update config/settings.py to allow extra fields
# Add 'extra = "ignore"' to the Config class
```

**2. Import Errors**
```bash
# Solution: Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

**2. API Key Issues**
```bash
# Check if API key is loaded
python -c "from config.settings import get_settings; print(get_settings().gemini_api_key)"
```

**3. Streamlit Port Issues**
```bash
# Use different port
streamlit run main.py --server.port 8502
```

**4. Memory Issues**
```bash
# Reduce concurrent agents
export MAX_CONCURRENT_AGENTS=2
```

### Debug Mode

Enable debug mode for detailed logging:

```env
DEBUG_MODE=true
LOG_LEVEL=DEBUG
```

### Check Logs

```bash
# View application logs
tail -f logs/app.log

# View agent-specific logs
tail -f logs/agents.log
```

## 🔄 Updates and Maintenance

### Update Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Update Application

```bash
git pull origin main
pip install -r requirements.txt
streamlit run main.py
```

### Clear Cache

```bash
# Clear Streamlit cache
streamlit cache clear

# Clear Python cache
find . -type d -name "__pycache__" -delete
```

## 📈 Performance Optimization

### 1. Enable Caching

Ensure caching is enabled in your `.env`:
```env
CACHE_TTL=3600
ENABLE_MONITORING=true
```

### 2. Optimize Agent Settings

```env
MAX_CONCURRENT_AGENTS=3  # Reduce for lower memory usage
AGENT_TIMEOUT=180        # Reduce timeout for faster failures
```

### 3. Use Production Settings

```env
DEBUG_MODE=false
LOG_LEVEL=WARNING
```

## 🚀 Production Deployment

### Docker Deployment (Recommended)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Environment Variables for Production

```env
# Production settings
DEBUG_MODE=false
LOG_LEVEL=INFO
ENABLE_MONITORING=true

# Security
SECRET_KEY=your_production_secret_key

# Performance
MAX_CONCURRENT_AGENTS=5
AGENT_TIMEOUT=300
```

## 📞 Support

### Getting Help

1. **Check the logs** in `logs/` directory
2. **Review the documentation** in this repository
3. **Create an issue** on GitHub with:
   - Error messages
   - System information
   - Steps to reproduce

### Useful Commands

```bash
# Check system info
python --version
pip list | grep -E "(streamlit|langchain|langgraph)"

# Test specific components
python -m pytest tests/ -v

# Generate requirements
pip freeze > requirements_current.txt
```

---

**🎉 Congratulations!** You've successfully set up the RM-AgenticAI-LangGraph system. 

For advanced configuration and customization, refer to the main [README.md](README.md) file.