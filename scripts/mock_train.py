import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "cv_cattle" / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Generating mock lameness model for pipeline verification...")
    # 72 features (var=24, mean=24, disp=24)
    # create synthetic data
    X = np.random.rand(100, 72)
    y = np.random.randint(0, 2, 100)
    
    clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    clf.fit(X, y)
    
    model_path = MODEL_DIR / "lameness_model.pkl"
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
