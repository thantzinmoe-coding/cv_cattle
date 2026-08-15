import json
import argparse

def inspect_coco(json_path):
    print(f"Inspecting {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    print("Keys in JSON:", list(data.keys()))
    if 'info' in data:
        print("\nDataset Info:", data['info'])
    print(f"\nNumber of images: {len(data['images'])}")
    print(f"Number of annotations: {len(data['annotations'])}")
    
    if 'categories' in data:
        cat = data['categories'][0]
        print(f"\nCategory: {cat['name']}")
        if 'keypoints' in cat:
            print(f"Number of keypoints: {len(cat['keypoints'])}")
            print("Keypoints:", cat['keypoints'])
            
    if data['annotations']:
        print("\nAnnotation example:")
        print(json.dumps(data['annotations'][0], indent=2))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect COCO annotation JSON.")
    parser.add_argument("--json", required=True, help="Path to COCO JSON annotation file")
    args = parser.parse_args()
    inspect_coco(args.json)
