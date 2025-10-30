"""
Model Training Orchestration
Main entry point for training all ML models for the Prospect Analysis workflow.
"""

from ml.training.predict_risk_profile import train_risk_model
from ml.training.predict_goal_success import train_goal_model


def main():
    """Train all models for the workflow"""
    print("=" * 60)
    print("[START] Starting ML Model Training...")
    print("=" * 60)

    models_to_train = [
        ("Risk Profile Model", train_risk_model),
        ("Goal Success Model", train_goal_model),
    ]

    results = {}
    for model_name, train_func in models_to_train:
        print(f"\n[TRAINING] {model_name}...")
        try:
            success = train_func()
            results[model_name] = success
            status = "[SUCCESS]" if success else "[SKIPPED]"
            print(f"{status}: {model_name}")
        except Exception as e:
            print(f"[ERROR] training {model_name}: {str(e)}")
            results[model_name] = False

    print("\n" + "=" * 60)
    print("[SUMMARY] Training Summary:")
    print("=" * 60)
    for model_name, success in results.items():
        status = "[OK]" if success else "[FAILED]"
        print(f"{status} {model_name}")

    print("\n" + "=" * 60)
    print("[DONE] Model training completed!")
    print("=" * 60)

    return all(results.values())


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
