# 🤖 RM-AgenticAI-LangGraph

## Advanced AI-Powered Investment Analyzer with Multi-Agent System

A sophisticated financial advisory platform that combines machine learning models with LangGraph's multi-agent architecture to provide intelligent investment recommendations and relationship management tools.

## 🎯 Overview

This system transforms traditional financial advisory through:
- **Multi-Agent Architecture**: Specialized AI agents for different aspects of financial analysis
- **Intelligent Workflows**: Dynamic decision trees based on client profiles
- **Real-time Analysis**: Instant risk assessment and goal prediction
- **Personalized Recommendations**: AI-driven product matching and justification
- **Interactive RM Assistant**: Context-aware chat system for relationship managers

## 🏗️ Architecture

### Multi-Agent System
- **🔍 Data Analyst Agent**: Input validation and data processing
- **📊 Risk Assessment Agent**: ML-based risk profiling
- **🎯 Goal Planning Agent**: Investment goal success prediction
- **👤 Persona Agent**: Client behavioral classification
- **💼 Product Specialist Agent**: Intelligent product recommendations
- **📋 Meeting Coordinator Agent**: Automated meeting guide generation
- **🤖 RM Assistant Agent**: Interactive query handling
- **📈 Portfolio Optimizer Agent**: Investment allocation optimization
- **⚠️ Compliance Agent**: Regulatory compliance checks

### Workflow Engine
```
Data Input → Validation → Risk & Goal Analysis → Persona Classification
                                ↓
Portfolio Optimization ← Product Recommendation ← Compliance Check
                                ↓
Meeting Guide Generation → Interactive RM Assistant
```

## 🚀 Features

### Core Capabilities
- **Automated Risk Profiling**: ML-based classification (Low/Moderate/High)
- **Goal Success Prediction**: Probability analysis for investment objectives
- **AI Persona Classification**: Behavioral categorization using GenAI
- **Dynamic Product Matching**: Context-aware recommendation engine
- **Intelligent Meeting Preparation**: Auto-generated discussion guides
- **Interactive Chat Assistant**: Real-time RM support system

### Advanced Features
- **Parallel Processing**: Multiple prospect analysis simultaneously
- **Conditional Workflows**: Adaptive paths based on client profiles
- **Memory Management**: Context retention across sessions
- **Performance Monitoring**: Real-time agent analytics
- **Audit Trails**: Complete decision tracking

## 📁 Project Structure

```
RM-AgenticAI-LangGraph/
├── README.md
├── requirements.txt
├── .env.example
├── main.py                          # Streamlit application
├── config/
│   ├── __init__.py
│   ├── settings.py                  # Configuration management
│   └── logging_config.py            # Logging setup
├── langraph_agents/
│   ├── __init__.py
│   ├── base_agent.py               # Base agent class
│   ├── state_models.py             # Pydantic state models
│   ├── agents/                     # Individual agent implementations
│   │   ├── __init__.py
│   │   ├── data_analyst_agent.py
│   │   ├── risk_assessment_agent.py
│   │   ├── goal_planning_agent.py
│   │   ├── persona_agent.py
│   │   ├── product_specialist_agent.py
│   │   ├── meeting_coordinator_agent.py
│   │   ├── rm_assistant_agent.py
│   │   ├── portfolio_optimizer_agent.py
│   │   └── compliance_agent.py
│   ├── tools/                      # Agent tools and utilities
│   │   ├── __init__.py
│   │   ├── ml_model_tools.py
│   │   ├── data_processing_tools.py
│   │   ├── genai_tools.py
│   │   └── calculation_tools.py
│   ├── workflows/                  # LangGraph workflows
│   │   ├── __init__.py
│   │   ├── prospect_analysis_workflow.py
│   │   ├── product_recommendation_workflow.py
│   │   └── interactive_chat_workflow.py
│   └── utils/                      # Graph utilities
│       ├── __init__.py
│       ├── graph_builder.py
│       ├── state_manager.py
│       └── monitoring.py
├── models/                         # ML models and encoders
│   ├── __init__.py
│   ├── risk_profile_model.pkl
│   ├── goal_success_model.pkl
│   ├── label_encoders.pkl
│   └── goal_success_label_encoders.pkl
├── data/
│   ├── input_data/
│   │   ├── prospects.csv
│   │   └── products.csv
│   ├── training_data/
│   └── evaluation_data/
├── legacy/                         # Original implementation
│   ├── genAI/
│   ├── train_model/
│   └── utils/
└── tests/
    ├── __init__.py
    ├── test_agents.py
    ├── test_workflows.py
    └── test_integration.py
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Amruth22/RM-AgenticAI-LangGraph.git
   cd RM-AgenticAI-LangGraph
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run the application**
   ```bash
   streamlit run main.py
   ```

## ⚙️ Configuration

### Environment Variables
```env
# Google AI API Key
GEMINI_API_KEY_1=your_gemini_api_key_here

# LangSmith (Optional - for monitoring)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key

# Application Settings
LOG_LEVEL=INFO
ENABLE_MONITORING=true
```

## 🔧 Usage

### Basic Workflow
1. **Select Prospect**: Choose from available prospects
2. **Automated Analysis**: Multi-agent system processes client data
3. **Review Results**: Risk profile, goal success, and persona classification
4. **Product Recommendations**: AI-generated product justifications
5. **Meeting Preparation**: Auto-generated meeting guides
6. **Interactive Chat**: Query the RM assistant for insights

### Advanced Features
- **Batch Processing**: Analyze multiple prospects simultaneously
- **Custom Workflows**: Define specialized analysis paths
- **Performance Monitoring**: Track agent performance and decisions
- **Audit Trails**: Review complete analysis history

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_agents.py
pytest tests/test_workflows.py
pytest tests/test_integration.py

# Run with coverage
pytest --cov=langraph_agents
```

## 📊 Monitoring

The system includes comprehensive monitoring:
- **Agent Performance**: Execution times and success rates
- **Workflow Analytics**: Path analysis and bottlenecks
- **Decision Tracking**: Complete audit trails
- **Error Monitoring**: Exception tracking and recovery

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on LangGraph for multi-agent orchestration
- Powered by Google Gemini for generative AI capabilities
- Uses scikit-learn for machine learning models
- Streamlit for the user interface

## 📞 Support

For support and questions:
- Create an issue in this repository
- Contact the development team
- Check the documentation in the `/docs` folder

---

**Made with ❤️ for the future of financial advisory**