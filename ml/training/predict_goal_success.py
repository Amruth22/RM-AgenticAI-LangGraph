"""
Goal Success Prediction Node
Predicts probability of goal success using ML model.
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


def train_goal_model():
    """Train goal success prediction model using the available dataset"""

    model_path = "ml/models/goal_success_model.pkl"
    encoder_path = "ml/models/goal_success_label_encoders.pkl"

    if os.path.exists(model_path):
        print(f" Goal success model already exists at {model_path}")
        return True

    try:
        # Load prospects data
        data_path = "data/input_data/prospects.csv"
        if not os.path.exists(data_path):
            print(" Goal success model training data not available")
            print("Please provide training dataset in data/input_data/prospects.csv")
            return False

        df = pd.read_csv(data_path)
        print(f"[TRAINING] Loaded {len(df)} prospect records for goal success model")

        # Create target variable: goal_success based on savings capacity and timeline
        def assign_goal_success(row):
            # Calculate annual savings rate
            annual_savings = row['current_savings'] / max(1, row['investment_horizon_years'])
            savings_rate = annual_savings / max(1, row['annual_income'])

            # If good savings rate and reasonable timeline, likely to succeed
            if row['investment_horizon_years'] >= 10 and savings_rate > 0.1:
                return 'Likely'
            elif row['investment_horizon_years'] < 3 and row['target_goal_amount'] > row['current_savings'] * 3:
                return 'Unlikely'
            else:
                return 'Moderate'

        df['goal_success'] = df.apply(assign_goal_success, axis=1)

        # Feature selection
        feature_cols = ['age', 'annual_income', 'current_savings', 'investment_horizon_years',
                       'target_goal_amount', 'number_of_dependents', 'investment_experience_level']
        X = df[feature_cols].copy()
        y = df['goal_success']

        # Encode categorical variables
        label_encoders = {}
        X_encoded = X.copy()

        for col in ['investment_experience_level']:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
            label_encoders[col] = le

        # Train Random Forest model
        print("[TRAINING] Training Goal Success model...")
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

        print(f"[SUCCESS] Goal success model trained and saved to {model_path}")
        print(f"[INFO] Model features: {feature_cols}")
        print(f"[INFO] Model classes: {model.classes_}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to train goal success model: {str(e)}")
        return False


async def predict_goal_success(state: WorkflowState) -> WorkflowState:
    """Predict probability of achieving investment goal using ML model or rule-based fallback"""

    try:
        model_path = "ml/models/goal_success_model.pkl"
        encoder_path = "ml/models/goal_success_label_encoders.pkl"

        # Try to load trained model
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            model = joblib.load(model_path)
            label_encoders = joblib.load(encoder_path)

            print("[ML] Using trained Goal Success model")

            # Prepare features for prediction
            features = pd.DataFrame({
                'age': [state.prospect.age],
                'annual_income': [state.prospect.annual_income],
                'current_savings': [state.prospect.current_savings],
                'investment_horizon_years': [state.prospect.investment_horizon_years],
                'target_goal_amount': [state.prospect.target_goal_amount],
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
            goal_pred = model.predict(X_encoded)[0]
            goal_proba = model.predict_proba(X_encoded)[0]

            # Map prediction to probability
            success_mapping = {'Unlikely': 0.3, 'Moderate': 0.6, 'Likely': 0.85}
            state.prospect.goal_success_probability = success_mapping.get(goal_pred, 0.5)

            print(f"[ML] Predicted goal success: {goal_pred} (probability: {state.prospect.goal_success_probability:.1%})")

        else:
            # Train model if it doesn't exist
            print("[MODEL] Model not found, attempting to train...")
            if train_goal_model():
                # Retry prediction after training
                return await predict_goal_success(state)
            else:
                # Fallback to rule-based prediction
                print("[FALLBACK] Using rule-based fallback for goal success prediction")

                # Simple rule-based goal success assessment
                success_probability = 0.5  # Default 50%

                # Adjust based on investment horizon
                if state.prospect.investment_horizon_years >= 10:
                    success_probability += 0.25
                elif state.prospect.investment_horizon_years >= 5:
                    success_probability += 0.15
                elif state.prospect.investment_horizon_years < 2:
                    success_probability -= 0.20

                # Adjust based on savings rate
                if state.prospect.annual_income > 0:
                    savings_rate = state.prospect.current_savings / (state.prospect.annual_income * state.prospect.investment_horizon_years + 0.01)
                    if savings_rate > 0.3:
                        success_probability += 0.15
                    elif savings_rate < 0.1:
                        success_probability -= 0.15

                state.prospect.goal_success_probability = max(0.0, min(1.0, success_probability))

        state.current_step = "goal_assessed"

    except Exception as e:
        print(f"[ERROR] Goal success prediction error: {str(e)}")
        # Fallback prediction
        state.prospect.goal_success_probability = 0.5
        state.current_step = "goal_assessed"

    return state


# Auto-train model if this file is run directly
if __name__ == "__main__":
    train_goal_model()
