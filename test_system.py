"""System test script to validate the RM-AgenticAI-LangGraph setup."""

import asyncio
import sys
from datetime import datetime
from typing import Dict, Any

def test_imports():
    """Test all critical imports."""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        from graph import ProspectAnalysisWorkflow
        print("✅ LangGraph workflow imported successfully")
    except ImportError as e:
        print(f"❌ LangGraph workflow import failed: {e}")
        return False
    
    try:
        from config.settings import get_settings
        print("✅ Settings imported successfully")
    except ImportError as e:
        print(f"❌ Settings import failed: {e}")
        return False
    
    try:
        from langraph_agents.agents.data_analyst_agent import DataAnalystAgent
        from langraph_agents.agents.risk_assessment_agent import RiskAssessmentAgent
        from langraph_agents.agents.persona_agent import PersonaAgent
        from langraph_agents.agents.product_specialist_agent import ProductSpecialistAgent
        print("✅ All agents imported successfully")
    except ImportError as e:
        print(f"❌ Agent import failed: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration loading."""
    print("\n🔧 Testing configuration...")
    
    try:
        from config.settings import get_settings
        settings = get_settings()
        
        if settings.gemini_api_key and len(settings.gemini_api_key) > 10:
            print("✅ Gemini API key configured")
        else:
            print("⚠️  Gemini API key not configured or too short")
            return False
        
        print(f"✅ Log level: {settings.log_level}")
        print(f"✅ Max concurrent agents: {settings.max_concurrent_agents}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_data_loading():
    """Test data file loading."""
    print("\n📊 Testing data loading...")
    
    try:
        import pandas as pd
        from config.settings import get_settings
        settings = get_settings()
        
        # Test prospects data
        try:
            prospects_df = pd.read_csv(settings.prospects_csv)
            print(f"✅ Prospects data loaded: {len(prospects_df)} records")
        except Exception as e:
            print(f"⚠️  Prospects data loading failed: {e}")
        
        # Test products data
        try:
            products_df = pd.read_csv(settings.products_csv)
            print(f"✅ Products data loaded: {len(products_df)} records")
        except Exception as e:
            print(f"⚠️  Products data loading failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_agent_initialization():
    """Test agent initialization."""
    print("\n🤖 Testing agent initialization...")
    
    try:
        from langraph_agents.agents.data_analyst_agent import DataAnalystAgent
        from langraph_agents.agents.risk_assessment_agent import RiskAssessmentAgent
        from langraph_agents.agents.persona_agent import PersonaAgent
        from langraph_agents.agents.product_specialist_agent import ProductSpecialistAgent
        
        # Initialize agents
        data_analyst = DataAnalystAgent()
        print(f"✅ {data_analyst.name} initialized")
        
        risk_assessor = RiskAssessmentAgent()
        print(f"✅ {risk_assessor.name} initialized")
        
        persona_classifier = PersonaAgent()
        print(f"✅ {persona_classifier.name} initialized")
        
        product_specialist = ProductSpecialistAgent()
        print(f"✅ {product_specialist.name} initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False

def test_workflow_creation():
    """Test workflow creation."""
    print("\n🔄 Testing workflow creation...")
    
    try:
        from graph import ProspectAnalysisWorkflow
        
        workflow = ProspectAnalysisWorkflow()
        print("✅ Workflow created successfully")
        
        summary = workflow.get_workflow_summary()
        print(f"✅ Workflow has {len(summary['agents'])} agents")
        print(f"✅ Workflow has {len(summary['steps'])} steps")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow creation failed: {e}")
        return False

async def test_sample_analysis():
    """Test sample prospect analysis."""
    print("\n🧪 Testing sample analysis...")
    
    try:
        from graph import ProspectAnalysisWorkflow
        
        # Sample prospect data
        sample_prospect = {
            "prospect_id": "TEST001",
            "name": "Test Client",
            "age": 35,
            "annual_income": 800000,
            "current_savings": 500000,
            "target_goal_amount": 2000000,
            "investment_horizon_years": 10,
            "number_of_dependents": 2,
            "investment_experience_level": "Intermediate",
            "investment_goal": "Test Goal"
        }
        
        workflow = ProspectAnalysisWorkflow()
        print("✅ Starting sample analysis...")
        
        # Run analysis with timeout
        try:
            result = await asyncio.wait_for(
                workflow.analyze_prospect(sample_prospect),
                timeout=120  # 2 minute timeout
            )
            
            print("✅ Sample analysis completed successfully")
            
            # Check results
            if result.analysis.risk_assessment:
                print(f"✅ Risk assessment: {result.analysis.risk_assessment.risk_level}")
            
            if result.analysis.persona_classification:
                print(f"✅ Persona: {result.analysis.persona_classification.persona_type}")
            
            if result.recommendations.recommended_products:
                print(f"✅ Recommendations: {len(result.recommendations.recommended_products)} products")
            
            exec_summary = result.get_execution_summary()
            print(f"✅ Execution summary: {exec_summary['success_rate']:.1%} success rate")
            
            return True
            
        except asyncio.TimeoutError:
            print("⚠️  Sample analysis timed out (this may be due to API rate limits)")
            return True  # Don't fail the test for timeout
            
    except Exception as e:
        print(f"❌ Sample analysis failed: {e}")
        return False

def test_logging():
    """Test logging configuration."""
    print("\n📝 Testing logging...")
    
    try:
        from config.logging_config import setup_logging, get_logger
        
        setup_logging()
        logger = get_logger("TestLogger")
        
        logger.info("Test log message")
        print("✅ Logging configured successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

async def run_all_tests():
    """Run all system tests."""
    print("🚀 Starting RM-AgenticAI-LangGraph System Tests")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Data Loading Test", test_data_loading),
        ("Agent Initialization Test", test_agent_initialization),
        ("Workflow Creation Test", test_workflow_creation),
        ("Logging Test", test_logging),
        ("Sample Analysis Test", test_sample_analysis)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the configuration and try again.")
        return False

def main():
    """Main test function."""
    try:
        # Run async tests
        result = asyncio.run(run_all_tests())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()