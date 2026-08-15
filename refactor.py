import os
import shutil
from pathlib import Path

root = Path(__file__).parent

# Setup dirs
(root / "weights").mkdir(exist_ok=True)
(root / "core").mkdir(exist_ok=True)
(root / "dataset" / "raw").mkdir(parents=True, exist_ok=True)
(root / "core" / "__init__.py").touch(exist_ok=True)

# Move Weights
for f in os.listdir(root):
    if (f.startswith('yolo') and f.endswith('.pt')) or (f.startswith('.yolo') and f.endswith('.part')):
        shutil.move(str(root / f), str(root / "weights" / f))

# Move dataset
cow_pose = root / "Cow Pose Estimation"
if cow_pose.exists():
    shutil.move(str(cow_pose), str(root / "dataset" / "raw" / "Cow Pose Estimation"))

# Move Core files
for core_file in ['predict.py', 'train.py', 'evaluate.py', 'lameness_classifier.py']:
    f_path = root / core_file
    if f_path.exists():
        shutil.move(str(f_path), str(root / "core" / core_file))

# Move test api
if (root / "test_api.py").exists():
    shutil.move(str(root / "test_api.py"), str(root / "scripts" / "test_api.py"))

print("Refactoring complete.")
