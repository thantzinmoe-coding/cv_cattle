import json
import shutil
import random
from pathlib import Path
import argparse

random.seed(42)

def convert_coco_to_yolo(coco_json_path, image_dir, output_dir):
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)
        
    images_info = {img['id']: img for img in coco['images']}
    
    # Map image id to annotations
    img_to_anns = {img_id: [] for img_id in images_info.keys()}
    for ann in coco['annotations']:
        img_to_anns[ann['image_id']].append(ann)
        
    image_ids = list(images_info.keys())
    random.shuffle(image_ids)
    
    n = len(image_ids)
    train_split = int(0.7 * n)
    val_split = int(0.9 * n)
    
    splits = {
        'train': image_ids[:train_split],
        'val': image_ids[train_split:val_split],
        'test': image_ids[val_split:]
    }

    images_out_dir = Path(output_dir) / 'images'
    labels_out_dir = Path(output_dir) / 'labels'
    
    for split_name in splits.keys():
        (images_out_dir / split_name).mkdir(parents=True, exist_ok=True)
        (labels_out_dir / split_name).mkdir(parents=True, exist_ok=True)
    
    for split_name, ids in splits.items():
        print(f"Processing {split_name} split: {len(ids)} images")
        
        for img_id in ids:
            img_info = images_info[img_id]
            file_name = img_info['file_name']
            width = img_info['width']
            height = img_info['height']
            
            src_img_path = Path(image_dir) / file_name
            if not src_img_path.exists():
                print(f"Warning: {src_img_path} not found. Skipping...")
                continue
                
            dst_img_path = images_out_dir / split_name / file_name
            shutil.copy(src_img_path, dst_img_path)
            
            label_name = Path(file_name).stem + '.txt'
            label_path = labels_out_dir / split_name / label_name
            
            with open(label_path, 'w') as f_out:
                for ann in img_to_anns[img_id]:
                    # COCO bbox: [x_min, y_min, width, height]
                    bbox = ann['bbox']
                    # YOLO: x_center, y_center, width, height (normalized)
                    x_c = (bbox[0] + bbox[2] / 2) / width
                    y_c = (bbox[1] + bbox[3] / 2) / height
                    w = bbox[2] / width
                    h = bbox[3] / height
                    
                    # Keypoints: [x1, y1, v1, x2, y2, v2, ...]
                    kpts = ann.get('keypoints', [])
                    yolo_kpts = []
                    for i in range(0, len(kpts), 3):
                        kx = kpts[i] / width
                        ky = kpts[i+1] / height
                        kv = kpts[i+2]
                        yolo_kpts.extend([kx, ky, kv])
                        
                    kpts_str = " ".join([f"{k:.5f}" if isinstance(k, float) else str(k) for k in yolo_kpts])
                    line = f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} {kpts_str}\n"
                    f_out.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO JSON annotations to YOLOv8 Pose txt format.")
    parser.add_argument("--json", required=True, help="Path to COCO JSON file")
    parser.add_argument("--images", required=True, help="Path to source image directory containing images")
    parser.add_argument("--output", default="dataset", help="Output directory for YOLO dataset")
    args = parser.parse_args()
    convert_coco_to_yolo(args.json, args.images, args.output)
    print("Conversion complete!")
