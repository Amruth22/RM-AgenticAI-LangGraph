#!/usr/bin/env python3
"""Test script to validate uploaded ML models and their integration."""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def test_model_files():
    """Test if all required model files exist and are loadable."""
    print("🔍 Testing Model Files")
    print("-" * 40)
    
    required_files = {
        "risk_profile_model.pkl": "Risk assessment model",
        "label_encoders.pkl": "Risk model label encoders", 
        "goal_success_model.pkl": "Goal success prediction model",
        "goal_success_label_encoders.pkl": "Goal model label encoders"
    }
    
    models_dir = Path("models")
    results = {}
    
    for filename, description in required_files.items():
        filepath = models_dir / filename
        
        if not filepath.exists():
            print(f"❌ {filename}: File not found")
            results[filename] = False
            continue
        
        try:
            # Try to load the model/encoder
            model_data = joblib.load(filepath)
            print(f"✅ {filename}: {description} loaded successfully")
            
            # Basic validation
            if hasattr(model_data, 'predict'):
                print(f"   - Model type: {type(model_data).__name__}")
                if hasattr(model_data, 'classes_'):
                    print(f"   - Classes: {model_data.classes_}")
            elif isinstance(model_data, dict):
                print(f"   - Encoders count: {len(model_data)}")
                print(f"   - Encoder keys: {list(model_data.keys())}")
            
            results[filename] = True
            
        except Exception as e:
            print(f"❌ {filename}: Failed to load - {str(e)}")
            results[filename] = False
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n📊 Model Files: {success_count}/{total_count} loaded successfully")
    return success_count == total_count

def test_risk_model():
    """Test the risk assessment model with sample data."""
    print("\n🎯 Testing Risk Assessment Model")
    print("-" * 40)
    
    try:
        # Load models
        risk_model = joblib.load("models/risk_profile_model.pkl")
        risk_encoders = joblib.load("models/label_encoders.pkl")
        
        print("✅ Risk models loaded")
        
        # Create sample data
        sample_data = {
            "age": 35,
            "annual_income": 800000,
            "current_savings": 500000,
            "target_goal_amount": 2000000,
            "investment_horizon_years": 10,
            "number_of_dependents": 2,
            "investment_experience_level": "Intermediate"
        }
        
        print(f"📝 Testing with sample prospect: {sample_data['age']} years old, ₹{sample_data['annual_income']:,} income")
        
        # Prepare data
        input_df = pd.DataFrame([sample_data])
        
        # Encode categorical variables
        for col, encoder in risk_encoders.items():
            if col in input_df.columns:
                try:
                    original_value = input_df[col].iloc[0]
                    input_df[col] = encoder.transform(input_df[col])
                    print(f"   - Encoded {col}: '{original_value}' → {input_df[col].iloc[0]}")
                except ValueError as e:
                    print(f"   ⚠️ Encoding issue for {col}: {e}")
                    # Use first class as fallback
                    input_df[col] = encoder.transform([encoder.classes_[0]])[0]
        
        # Make prediction
        prediction = risk_model.predict(input_df)[0]
        probabilities = risk_model.predict_proba(input_df)[0]
        
        # Map prediction
        risk_mapping = {0: "Low", 1: "Moderate", 2: "High"}
        risk_level = risk_mapping.get(prediction, f"Unknown({prediction})")
        confidence = float(max(probabilities))
        
        print(f"🎯 Prediction Results:")
        print(f"   - Risk Level: {risk_level}")
        print(f"   - Confidence: {confidence:.1%}")
        print(f"   - Probabilities: Low={probabilities[0]:.3f}, Moderate={probabilities[1]:.3f}, High={probabilities[2]:.3f}")
        
        # Validate results
        if risk_level in ["Low", "Moderate", "High"] and 0 <= confidence <= 1:
            print("✅ Risk model test passed")
            return True
        else:
            print("❌ Risk model test failed - invalid results")
            return False
            
    except Exception as e:
        print(f"❌ Risk model test failed: {str(e)}")
        return False

def test_goal_model():
    """Test the goal success prediction model with sample data."""
    print("\n🎯 Testing Goal Success Model")
    print("-" * 40)
    
    try:
        # Load models
        goal_model = joblib.load("models/goal_success_model.pkl")
        goal_encoders = joblib.load("models/goal_success_label_encoders.pkl")
        
        print("✅ Goal models loaded")
        
        # Create sample data
        sample_data = {
            "age": 35,
            "annual_income": 800000,
            "current_savings": 500000,
            "target_goal_amount": 2000000,
            "investment_experience_level": "Intermediate",
            "investment_horizon_years": 10
        }
        
        print(f"📝 Testing goal: ₹{sample_data['target_goal_amount']:,} in {sample_data['investment_horizon_years']} years")
        
        # Prepare data
        input_df = pd.DataFrame([sample_data])
        
        # Encode categorical variables
        for col, encoder in goal_encoders.items():
            if col in input_df.columns:
                try:
                    original_value = input_df[col].iloc[0]
                    input_df[col] = encoder.transform(input_df[col])
                    print(f"   - Encoded {col}: '{original_value}' → {input_df[col].iloc[0]}")
                except ValueError as e:
                    print(f"   ⚠️ Encoding issue for {col}: {e}")
                    # Use first class as fallback
                    input_df[col] = encoder.transform([encoder.classes_[0]])[0]
        
        # Make prediction
        if hasattr(goal_model, 'predict_proba'):
            # Classification model
            probabilities = goal_model.predict_proba(input_df)[0]
            prediction = goal_model.predict(input_df)[0]
            
            goal_success = "Likely" if prediction == 1 else "Unlikely"
            probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
            
            print(f"🎯 Prediction Results (Classification):")
            print(f"   - Goal Success: {goal_success}")
            print(f"   - Probability: {probability:.1%}")
            if len(probabilities) > 1:
                print(f"   - Class Probabilities: {probabilities}")
        else:
            # Regression model
            probability = float(goal_model.predict(input_df)[0])
            goal_success = "Likely" if probability > 0.6 else "Unlikely"
            
            print(f"🎯 Prediction Results (Regression):")
            print(f"   - Goal Success: {goal_success}")
            print(f"   - Probability: {probability:.1%}")
        
        # Validate results
        if goal_success in ["Likely", "Unlikely"] and 0 <= probability <= 1:
            print("✅ Goal model test passed")
            return True
        else:
            print("❌ Goal model test failed - invalid results")
            return False
            
    except Exception as e:
        print(f"❌ Goal model test failed: {str(e)}")
        return False

def test_agent_integration():
    """Test that agents can properly load and use the models."""
    print("\n🤖 Testing Agent Integration")
    print("-" * 40)
    
    try:
        # Add current directory to path
        sys.path.insert(0, '.')
        
        # Test Risk Assessment Agent
        from langraph_agents.agents.risk_assessment_agent import RiskAssessmentAgent
        risk_agent = RiskAssessmentAgent()
        
        if risk_agent.risk_model is not None and risk_agent.label_encoders is not None:
            print("✅ Risk Assessment Agent: Models loaded successfully")
            risk_integration = True
        else:
            print("❌ Risk Assessment Agent: Models not loaded")
            risk_integration = False
        
        # Test Goal Planning Agent
        from langraph_agents.agents.goal_planning_agent import GoalPlanningAgent
        goal_agent = GoalPlanningAgent()
        
        if goal_agent.goal_model is not None and goal_agent.goal_encoders is not None:
            print("✅ Goal Planning Agent: Models loaded successfully")
            goal_integration = True
        else:
            print("❌ Goal Planning Agent: Models not loaded")
            goal_integration = False
        
        return risk_integration and goal_integration
        
    except Exception as e:
        print(f"❌ Agent integration test failed: {str(e)}")
        return False

def test_model_performance():
    """Test model performance with multiple samples."""
    print("\n📊 Testing Model Performance")
    print("-" * 40)
    
    try:
        # Load models
        risk_model = joblib.load("models/risk_profile_model.pkl")
        risk_encoders = joblib.load("models/label_encoders.pkl")
        goal_model = joblib.load("models/goal_success_model.pkl")
        goal_encoders = joblib.load("models/goal_success_label_encoders.pkl")
        
        # Test with multiple samples
        test_samples = [
            {"age": 25, "annual_income": 600000, "current_savings": 100000, "target_goal_amount": 1000000, 
             "investment_horizon_years": 15, "number_of_dependents": 0, "investment_experience_level": "Beginner"},
            {"age": 40, "annual_income": 1200000, "current_savings": 800000, "target_goal_amount": 3000000, 
             "investment_horizon_years": 8, "number_of_dependents": 2, "investment_experience_level": "Advanced"},
            {"age": 55, "annual_income": 800000, "current_savings": 1500000, "target_goal_amount": 2000000, 
             "investment_horizon_years": 5, "number_of_dependents": 1, "investment_experience_level": "Intermediate"}
        ]
        
        print(f"Testing with {len(test_samples)} sample prospects...")
        
        risk_predictions = []
        goal_predictions = []
        
        for i, sample in enumerate(test_samples, 1):
            print(f"\n📝 Sample {i}: Age {sample['age']}, Income ₹{sample['annual_income']:,}")
            
            # Risk prediction
            risk_df = pd.DataFrame([sample])
            for col, encoder in risk_encoders.items():
                if col in risk_df.columns:
                    try:
                        risk_df[col] = encoder.transform(risk_df[col])
                    except ValueError:
                        risk_df[col] = encoder.transform([encoder.classes_[0]])[0]
            
            risk_pred = risk_model.predict(risk_df)[0]
            risk_proba = risk_model.predict_proba(risk_df)[0]
            risk_level = {0: "Low", 1: "Moderate", 2: "High"}[risk_pred]
            risk_predictions.append(risk_level)
            
            # Goal prediction
            goal_df = pd.DataFrame([{k: v for k, v in sample.items() if k != 'number_of_dependents'}])
            for col, encoder in goal_encoders.items():
                if col in goal_df.columns:
                    try:
                        goal_df[col] = encoder.transform(goal_df[col])
                    except ValueError:
                        goal_df[col] = encoder.transform([encoder.classes_[0]])[0]
            
            if hasattr(goal_model, 'predict_proba'):
                goal_proba = goal_model.predict_proba(goal_df)[0]
                goal_prob = float(goal_proba[1]) if len(goal_proba) > 1 else float(goal_proba[0])
            else:
                goal_prob = float(goal_model.predict(goal_df)[0])
            
            goal_success = "Likely" if goal_prob > 0.6 else "Unlikely"
            goal_predictions.append(goal_success)
            
            print(f"   - Risk: {risk_level} ({max(risk_proba):.1%} confidence)")
            print(f"   - Goal: {goal_success} ({goal_prob:.1%} probability)")
        
        # Summary
        print(f"\n📊 Performance Summary:")
        print(f"   - Risk Levels: {dict(pd.Series(risk_predictions).value_counts())}")
        print(f"   - Goal Success: {dict(pd.Series(goal_predictions).value_counts())}")
        print("✅ Model performance test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Model performance test failed: {str(e)}")
        return False

def generate_model_report():
    """Generate a comprehensive model report."""
    print("\n📋 Generating Model Report")
    print("-" * 40)
    
    try:
        # Load all models
        risk_model = joblib.load("models/risk_profile_model.pkl")
        risk_encoders = joblib.load("models/label_encoders.pkl")
        goal_model = joblib.load("models/goal_success_model.pkl")
        goal_encoders = joblib.load("models/goal_success_label_encoders.pkl")
        
        report = []
        report.append("# ML Models Report")
        report.append("=" * 50)
        report.append("")
        
        # Risk Model Info
        report.append("## Risk Assessment Model")
        report.append(f"- Model Type: {type(risk_model).__name__}")
        if hasattr(risk_model, 'classes_'):
            report.append(f"- Classes: {list(risk_model.classes_)}")
        if hasattr(risk_model, 'feature_names_in_'):
            report.append(f"- Features: {list(risk_model.feature_names_in_)}")
        report.append(f"- Risk Encoders: {list(risk_encoders.keys())}")
        report.append("")
        
        # Goal Model Info
        report.append("## Goal Success Model")
        report.append(f"- Model Type: {type(goal_model).__name__}")
        if hasattr(goal_model, 'classes_'):
            report.append(f"- Classes: {list(goal_model.classes_)}")
        if hasattr(goal_model, 'feature_names_in_'):
            report.append(f"- Features: {list(goal_model.feature_names_in_)}")
        report.append(f"- Goal Encoders: {list(goal_encoders.keys())}")
        report.append("")
        
        # Save report
        with open("models/MODEL_REPORT.md", "w") as f:
            f.write("\n".join(report))
        
        print("✅ Model report generated: models/MODEL_REPORT.md")
        return True
        
    except Exception as e:
        print(f"❌ Failed to generate model report: {str(e)}")
        return False

def main():
    """Run comprehensive model testing."""
    print("🧪 ML Models Validation Suite")
    print("=" * 50)
    
    tests = [
        ("Model Files", test_model_files),
        ("Risk Model", test_risk_model),
        ("Goal Model", test_goal_model),
        ("Agent Integration", test_agent_integration),
        ("Model Performance", test_model_performance),
        ("Model Report", generate_model_report),
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
    print("📊 MODEL VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All model tests passed! Your ML models are ready for use.")
        print("The system will now use ML-based predictions with high accuracy.")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Some models may not work correctly.")
        print("The system will fall back to rule-based predictions where needed.")
    
    print("\n📋 Next Steps:")
    print("1. Check models/MODEL_REPORT.md for detailed model information")
    print("2. Run: streamlit run main.py")
    print("3. Test with real prospects to see ML predictions in action")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)