"""
Risk Profile Prediction Node
Predicts risk profile and risk scores using ML model.
"""

import joblib
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
from state import WorkflowState


def train_risk_model():
    """Train risk profile model using the available dataset"""

    model_path = "ml/models/risk_profile_model.pkl"
    encoder_path = "ml/models/label_encoders.pkl"

    if os.path.exists(model_path):
        print(f" Risk model already exists at {model_path}")
        return True

    try:
        # Load prospects data
        data_path = "data/input_data/prospects.csv"
        if not os.path.exists(data_path):
            print(" Risk profile model training data not available")
            print("Please provide training dataset in data/input_data/prospects.csv")
            return False

        df = pd.read_csv(data_path)
        print(f"[TRAINING] Loaded {len(df)} prospect records")

        # Create target variable: risk_level based on investment experience and age
        def assign_risk_level(row):
            if row['investment_experience_level'] == 'Beginner':
                return 'Low'
            elif row['investment_experience_level'] == 'Advanced':
                return 'High'
            else:  # Intermediate
                return 'Medium'

        df['risk_level'] = df.apply(assign_risk_level, axis=1)

        # Feature selection
        feature_cols = ['age', 'annual_income', 'current_savings', 'investment_horizon_years',
                       'number_of_dependents', 'investment_experience_level']
        X = df[feature_cols].copy()
        y = df['risk_level']

        # Encode categorical variables
        label_encoders = {}
        X_encoded = X.copy()

        for col in ['investment_experience_level']:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
            label_encoders[col] = le

        # Train Random Forest model
        print("[TRAINING] Training Risk Assessment model...")
        model = RandomForestClassifier(
            n_estimators=10,
            max_depth=5,
            random_state=42,
            min_samples_split=2,
            min_samples_leaf=1
        )
        model.fit(X_encoded, y)

        # Create models directory if it doesn't exist
        os.makedirs("ml/models", exist_ok=True)

        # Save model and encoders
        joblib.dump(model, model_path)
        joblib.dump(label_encoders, encoder_path)

        print(f"[SUCCESS] Risk model trained and saved to {model_path}")
        print(f"[INFO] Model features: {feature_cols}")
        print(f"[INFO] Model classes: {model.classes_}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to train risk model: {str(e)}")
        return False


async def predict_risk_profile(state: WorkflowState) -> WorkflowState:
    """Predict risk profile using ML model or rule-based fallback"""

    try:
        model_path = "ml/models/risk_profile_model.pkl"
        encoder_path = "ml/models/label_encoders.pkl"

        # Try to load trained model
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            model = joblib.load(model_path)
            label_encoders = joblib.load(encoder_path)

            print("[ML] Using trained Risk Assessment model")

            # Prepare features for prediction
            features = pd.DataFrame({
                'age': [state.prospect.age],
                'annual_income': [state.prospect.annual_income],
                'current_savings': [state.prospect.current_savings],
                'investment_horizon_years': [state.prospect.investment_horizon_years],
                'number_of_dependents': [state.prospect.number_of_dependents],
                'investment_experience_level': [state.prospect.investment_experience_level]
            })

            # Encode categorical features
            X_encoded = features.copy()
            for col in ['investment_experience_level']:
                if col in label_encoders:
                    try:
                        X_encoded[col] = label_encoders[col].transform(features[col])
                    except ValueError:
                        # Handle unseen categories
                        X_encoded[col] = 0

            # Make prediction
            risk_pred = model.predict(X_encoded)[0]
            risk_proba = model.predict_proba(X_encoded)[0]

            # Map to risk score (0-100)
            risk_mapping = {'Low': 25, 'Medium': 50, 'High': 75}
            state.prospect.risk_level = risk_pred
            state.prospect.risk_score = risk_mapping.get(risk_pred, 50)

            print(f"[ML] Predicted risk level: {risk_pred} (confidence: {max(risk_proba):.2%})")

        else:
            # Train model if it doesn't exist
            print("[MODEL] Model not found, attempting to train...")
            if train_risk_model():
                # Retry prediction after training
                return await predict_risk_profile(state)
            else:
                # Fallback to rule-based prediction
                print("[FALLBACK] Using rule-based fallback for risk prediction")

                # Simple rule-based risk assessment
                risk_score = 50  # Default medium risk
                if state.prospect.age < 30:
                    risk_score += 20
                elif state.prospect.age > 60:
                    risk_score -= 20

                if state.prospect.investment_experience_level == "Beginner":
                    risk_score -= 15
                elif state.prospect.investment_experience_level == "Expert":
                    risk_score += 15

                state.prospect.risk_score = max(0, min(100, risk_score))
                state.prospect.risk_level = "High" if risk_score > 70 else "Low" if risk_score < 30 else "Medium"

        state.current_step = "risk_assessed"

    except Exception as e:
        print(f"[ERROR] Risk prediction error: {str(e)}")
        # Fallback prediction
        state.prospect.risk_score = 50
        state.prospect.risk_level = "Medium"
        state.current_step = "risk_assessed"

    return state


# Auto-train model if this file is run directly
if __name__ == "__main__":
    train_risk_model()
