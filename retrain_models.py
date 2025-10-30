#!/usr/bin/env python3
"""Script to retrain ML models with current environment versions."""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_synthetic_training_data():
    """Create synthetic training data for model training."""
    print("📊 Creating synthetic training data...")
    
    # Generate synthetic prospect data
    np.random.seed(42)
    n_samples = 1000
    
    # Features
    ages = np.random.randint(25, 65, n_samples)
    annual_incomes = np.random.randint(300000, 2000000, n_samples)
    current_savings = np.random.randint(50000, 1500000, n_samples)
    target_amounts = current_savings + np.random.randint(500000, 5000000, n_samples)
    investment_horizons = np.random.randint(1, 20, n_samples)
    dependents = np.random.randint(0, 5, n_samples)
    
    # Experience levels
    experience_levels = np.random.choice(['Beginner', 'Intermediate', 'Advanced'], n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': ages,
        'annual_income': annual_incomes,
        'current_savings': current_savings,
        'target_goal_amount': target_amounts,
        'investment_horizon_years': investment_horizons,
        'number_of_dependents': dependents,
        'investment_experience_level': experience_levels
    })
    
    # Generate risk labels based on business logic
    risk_labels = []
    for _, row in data.iterrows():
        risk_score = 0
        
        # Age factor
        if row['age'] < 35:
            risk_score += 2
        elif row['age'] < 50:
            risk_score += 1
        
        # Income factor
        if row['annual_income'] > 1000000:
            risk_score += 2
        elif row['annual_income'] > 600000:
            risk_score += 1
        
        # Investment horizon
        if row['investment_horizon_years'] > 10:
            risk_score += 2
        elif row['investment_horizon_years'] > 5:
            risk_score += 1
        
        # Experience
        if row['investment_experience_level'] == 'Advanced':
            risk_score += 2
        elif row['investment_experience_level'] == 'Intermediate':
            risk_score += 1
        
        # Dependents reduce risk tolerance
        if row['number_of_dependents'] > 2:
            risk_score -= 1
        
        # Map to risk levels
        if risk_score >= 6:
            risk_labels.append('High')
        elif risk_score >= 3:
            risk_labels.append('Moderate')
        else:
            risk_labels.append('Low')
    
    data['risk_level'] = risk_labels
    
    # Generate goal success probabilities
    goal_success_probs = []
    for _, row in data.iterrows():
        # Calculate required monthly investment
        required_amount = row['target_goal_amount'] - row['current_savings']
        monthly_income = row['annual_income'] / 12
        affordable_investment = monthly_income * 0.2  # 20% of income
        
        # Simple calculation assuming 8% annual return
        if row['investment_horizon_years'] > 0:
            monthly_rate = 0.08 / 12
            months = row['investment_horizon_years'] * 12
            if monthly_rate > 0:
                required_monthly = required_amount * monthly_rate / ((1 + monthly_rate) ** months - 1)
            else:
                required_monthly = required_amount / months
        else:
            required_monthly = float('inf')
        
        # Determine success probability
        if required_monthly <= affordable_investment * 0.5:
            prob = np.random.uniform(0.8, 0.95)
        elif required_monthly <= affordable_investment:
            prob = np.random.uniform(0.6, 0.8)
        elif required_monthly <= affordable_investment * 1.5:
            prob = np.random.uniform(0.3, 0.6)
        else:
            prob = np.random.uniform(0.1, 0.3)
        
        goal_success_probs.append(prob)
    
    data['goal_success_probability'] = goal_success_probs
    
    print(f"✅ Created {len(data)} training samples")
    return data

def train_risk_model(data):
    """Train risk assessment model."""
    print("🎯 Training risk assessment model...")
    
    # Prepare features
    feature_cols = ['age', 'annual_income', 'current_savings', 'target_goal_amount', 
                   'investment_horizon_years', 'number_of_dependents', 'investment_experience_level']
    
    X = data[feature_cols].copy()
    y = data['risk_level']
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("Risk Model Performance:")
    print(classification_report(y_test, y_pred))
    
    # Save model and encoders
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    joblib.dump(model, models_dir / "risk_profile_model.pkl")
    joblib.dump(label_encoders, models_dir / "label_encoders.pkl")
    
    print("✅ Risk model saved successfully")
    return model, label_encoders

def train_goal_model(data):
    """Train goal success model."""
    print("🎯 Training goal success model...")
    
    # Prepare features (exclude number_of_dependents for goal model)
    feature_cols = ['age', 'annual_income', 'current_savings', 'target_goal_amount', 
                   'investment_horizon_years', 'investment_experience_level']
    
    X = data[feature_cols].copy()
    y = data['goal_success_probability']
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Goal Model Performance - MSE: {mse:.4f}")
    
    # Save model and encoders
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    joblib.dump(model, models_dir / "goal_success_model.pkl")
    joblib.dump(label_encoders, models_dir / "goal_success_label_encoders.pkl")
    
    print("✅ Goal model saved successfully")
    return model, label_encoders

def test_models():
    """Test the trained models."""
    print("🧪 Testing trained models...")
    
    try:
        # Load models
        risk_model = joblib.load("ml/models/risk_profile_model.pkl")
        risk_encoders = joblib.load("ml/models/label_encoders.pkl")
        goal_model = joblib.load("ml/models/goal_success_model.pkl")
        goal_encoders = joblib.load("ml/models/goal_success_label_encoders.pkl")
        
        # Test with sample data
        sample_data = {
            "age": 35,
            "annual_income": 800000,
            "current_savings": 500000,
            "target_goal_amount": 2000000,
            "investment_horizon_years": 10,
            "number_of_dependents": 2,
            "investment_experience_level": "Intermediate"
        }
        
        print(f"📝 Testing with sample: {sample_data['age']} years old, ₹{sample_data['annual_income']:,} income")
        
        # Test risk model
        risk_input = pd.DataFrame([sample_data])
        for col, encoder in risk_encoders.items():
            if col in risk_input.columns:
                risk_input[col] = encoder.transform(risk_input[col])
        
        risk_pred = risk_model.predict(risk_input)[0]
        risk_proba = risk_model.predict_proba(risk_input)[0]
        risk_mapping = {0: "Low", 1: "Moderate", 2: "High"}
        risk_level = risk_mapping.get(risk_pred, f"Unknown({risk_pred})")
        
        print(f"🎯 Risk Assessment: {risk_level} (confidence: {max(risk_proba):.1%})")
        
        # Test goal model
        goal_input = pd.DataFrame([{k: v for k, v in sample_data.items() if k != 'number_of_dependents'}])
        for col, encoder in goal_encoders.items():
            if col in goal_input.columns:
                goal_input[col] = encoder.transform(goal_input[col])
        
        goal_prob = goal_model.predict(goal_input)[0]
        goal_success = "Likely" if goal_prob > 0.6 else "Unlikely"
        
        print(f"🎯 Goal Success: {goal_success} ({goal_prob:.1%} probability)")
        
        print("✅ All models tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        return False

def main():
    """Main retraining function."""
    print("🔄 ML Model Retraining Script")
    print("=" * 50)
    
    try:
        # Create training data
        data = create_synthetic_training_data()
        
        # Train models
        risk_model, risk_encoders = train_risk_model(data)
        goal_model, goal_encoders = train_goal_model(data)
        
        # Test models
        test_success = test_models()
        
        if test_success:
            print("\n" + "=" * 50)
            print("🎉 MODEL RETRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print("✅ Risk assessment model retrained and saved")
            print("✅ Goal success model retrained and saved")
            print("✅ All encoders updated for current environment")
            print("✅ Models tested and working correctly")
            print("\n📋 Next Steps:")
            print("1. Restart your Streamlit application")
            print("2. The models should now load without version errors")
            print("3. You'll see 🤖 ML Model Predictions in the results")
        else:
            print("❌ Model testing failed. Please check the error messages above.")
            
    except Exception as e:
        print(f"❌ Retraining failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)