import numpy as np
import joblib
from pathlib import Path

class LamenessPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        if Path(model_path).exists():
            self.model = joblib.load(model_path)
            print(f"Lameness model loaded from {model_path}")
        else:
            print(f"Lameness model not found at {model_path}. Predictions will be UNKNOWN.")
            
    def is_ready(self):
        return self.model is not None
        
    def predict_lameness(self, keypoint_sequence):
        if not self.is_ready() or len(keypoint_sequence) < 30:
            return "UNKNOWN", 0.0
            
        seq = np.array(keypoint_sequence[-30:]) 
        
        variance_feat = np.var(seq, axis=0).flatten()
        mean_feat = np.mean(seq, axis=0).flatten()
        disp_feat = (seq[-1] - seq[0]).flatten()
        
        feature_vector = np.concatenate([variance_feat, mean_feat, disp_feat]).reshape(1, -1)
        
        preds = self.model.predict_proba(feature_vector)
        lame_prob = preds[0][1]
        
        if lame_prob >= 0.5:
            return "POSSIBLE LAMENESS", lame_prob
        else:
            return "NORMAL", 1.0 - lame_prob
