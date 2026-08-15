import os
import shutil
import random
from pathlib import Path

def reorganize():
    # Paths configured specifically for the workspace
    source_dir = Path(r"c:\Users\Asus\OneDrive - University of Information Technology\Desktop\cv-cattle-project\Cow Pose Estimation")
    dest_dir = Path(r"c:\Users\Asus\OneDrive - University of Information Technology\Desktop\cv-cattle-project\dataset")
    
    # Collect all image/label combination pairs across all existing subsets
    pairs = []
    for split in ['train', 'val']:
        img_split_dir = source_dir / 'images' / split
        lbl_split_dir = source_dir / 'labels' / split
        
        if not img_split_dir.exists(): continue
        
        for img_path in img_split_dir.glob("*"):
            if not img_path.is_file(): continue
            lbl_path = lbl_split_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                pairs.append((img_path, lbl_path))
                
    # Strict 42 seeded split as specified
    random.seed(42)
    random.shuffle(pairs)
    
    total = len(pairs)
    if total == 0:
        print("No image/label pairs found in 'Cow Pose Estimation'. Please ensure dataset is correctly structured.")
        return
        
    train_end = int(0.7 * total)
    val_end = int(0.9 * total)
    
    splits = {
        'train': pairs[:train_end],
        'val': pairs[train_end:val_end],
        'test': pairs[val_end:]
    }
    
    print(f"Total matched pairs found: {total}")
    for split_name, split_pairs in splits.items():
        print(f"Copying {len(split_pairs)} files to {split_name} split...")
        img_out = dest_dir / 'images' / split_name
        lbl_out = dest_dir / 'labels' / split_name
        
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        
        for img_src, lbl_src in split_pairs:
            shutil.copy(img_src, img_out / img_src.name)
            shutil.copy(lbl_src, lbl_out / lbl_src.name)
            
    print("Dataset strictly restructured and securely transferred to 'dataset' folder successfully!")

if __name__ == "__main__":
    reorganize()
