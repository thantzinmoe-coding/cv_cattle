import sys
from ultralytics import YOLO
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "cv_cattle" / "outputs" / "dataset_features"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PROJECT_ROOT / "cv_cattle" / "outputs" / "models" / "cattle_pose_best.pt"
DATA_DIR = PROJECT_ROOT / "cv_cattle" / "dataset" / "raw" / "CattleLameness" / "Data"

def normalize_keypoints(keypoints, box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if w == 0 or h == 0: return keypoints
    
    norm_kp = np.zeros_like(keypoints)
    norm_kp[:, 0] = (keypoints[:, 0] - x1) / w
    norm_kp[:, 1] = (keypoints[:, 1] - y1) / h
    return norm_kp

def process_videos(label, dir_path, model, max_frames=300):
    features = []
    labels = []
    vid_list = list(dir_path.glob("*.mp4"))
    vid_list = [v for v in dir_path.glob("*.mp4") if v.name != "N (1).mp4"]
        
    for vid_path in vid_list:
        print(f"Processing {vid_path.name} ({label})")
        results = model.track(source=str(vid_path), persist=True, tracker="botsort.yaml", stream=True, verbose=False)
        
        tracks = {}
        frame_idx = 0
        try:
            for r in results:
                if getattr(r.boxes, 'id', None) is not None and getattr(r, 'keypoints', None) is not None and r.keypoints.xy is not None:
                    ids = r.boxes.id.int().cpu().tolist()
                    kpts = r.keypoints.xy.cpu().numpy() 
                    boxes = r.boxes.xyxy.cpu().numpy()
                    for i, cow_id in enumerate(ids):
                        if cow_id not in tracks:
                            tracks[cow_id] = []
                        if len(kpts[i]) > 0:
                            norm_kpt = normalize_keypoints(kpts[i], boxes[i])
                            tracks[cow_id].append(norm_kpt)
                frame_idx += 1
                if frame_idx >= max_frames:
                    break
        except Exception as e:
            print(f"Error processing {vid_path.name}: {e}")
            continue
            
        SEQ_LEN = 30
        for cow_id, kpt_seq in tracks.items():
            if len(kpt_seq) >= SEQ_LEN:
                for start_idx in range(0, len(kpt_seq) - SEQ_LEN + 1, SEQ_LEN // 2):
                    seq = np.array(kpt_seq[start_idx:start_idx+SEQ_LEN]) 
                    
                    variance_feat = np.var(seq, axis=0).flatten()
                    mean_feat = np.mean(seq, axis=0).flatten()
                    disp_feat = (seq[-1] - seq[0]).flatten()
                    
                    feature_vector = np.concatenate([variance_feat, mean_feat, disp_feat])
                    
                    features.append(feature_vector)
                    labels.append(1 if label == "Lame" else 0)
    return features, labels
    
def main():
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        return
        
    model = YOLO(str(MODEL_PATH))
    lame_dir = DATA_DIR / "Lame"
    normal_dir = DATA_DIR / "Normal"
    
    print("Extracting features from Lame videos...")
    lame_feats, lame_labels = process_videos("Lame", lame_dir, model)
    print("Extracting features from Normal videos...")
    normal_feats, normal_labels = process_videos("Normal", normal_dir, model)
    
    if len(lame_feats) == 0 and len(normal_feats) == 0:
        print("No features extracted.")
        return
        
    X = np.array(lame_feats + normal_feats)
    y = np.array(lame_labels + normal_labels)
    
    print(f"Extracted shape X: {X.shape}, y: {y.shape}")
    np.save(OUTPUTS_DIR / "features_X.npy", X)
    np.save(OUTPUTS_DIR / "features_y.npy", y)
    print("Features saved.")

if __name__ == "__main__":
    main()
