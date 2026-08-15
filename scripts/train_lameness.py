import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "cv_cattle" / "outputs" / "dataset_features"
MODEL_DIR = PROJECT_ROOT / "cv_cattle" / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    try:
        X = np.load(OUTPUTS_DIR / "features_X.npy")
        y = np.load(OUTPUTS_DIR / "features_y.npy")
    except FileNotFoundError:
        print("Features not found. Please run extract_features.py first.")
        return
        
    print(f"Loaded features: X={X.shape}, y={y.shape}")
    
    if len(X) < 2:
        print("Not enough sequences to train.")
        return
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["Normal", "Lame"]))
    
    model_path = MODEL_DIR / "lameness_model.pkl"
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
