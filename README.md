# Cattle Body Pose Estimation and Cow Counting System

A complete end-to-end YOLOv8-based system designed to detect cattle, estimate 12 critical body keypoints, and count individual cows in static images and dynamic video streams using ByteTrack object tracking.

## Project Structure
```text
cattle-pose-estimation/
├── dataset/
│   ├── data.yaml
│   ├── images/  (train, val, test)
│   └── labels/  (train, val, test)
├── scripts/
│   ├── inspect_dataset.py
│   ├── convert_coco_to_yolo.py
│   └── visualize_results.py
├── train.py
├── evaluate.py
├── predict.py
├── outputs/
│   ├── models/
│   ├── metrics/
│   ├── plots/
│   ├── predictions/
│   └── comparison/
└── requirements.txt
```

## Setup Instructions

1. **Install dependencies**:
   Ensure you have Python 3.11+ installed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**:
   Download the [Kaggle Cow Pose Estimation Dataset](https://www.kaggle.com/datasets/zaidworks0508/cow-pose-estimation-dataset).
   - Convert COCO labels to YOLO format and automatically split into Train (70%), Val (20%), and Test (10%):
     ```bash
     python scripts/convert_coco_to_yolo.py --json <path-to-annotations.json> --images <path-to-raw-images>
     ```
   - Optionally inspect the COCO dataset constraints before processing:
     ```bash
     python scripts/inspect_dataset.py --json <path-to-annotations.json>
     ```

## Usage Pipeline

### 1. Training
Fine-tune the YOLOv8 nano pose model (`yolov8n-pose.pt`) for 50 epochs globally tracking training bounds.
```bash
python train.py
```
*Outputs are saved symmetrically in `outputs/models/cattle_pose_best.pt`, metrics in `outputs/metrics/`, and plots in `outputs/plots/`.*

### 2. Evaluation
Validate the final model's keypoint and bounding box Mean Average Precision (`mAP@50` & `mAP@50-95`) exclusively on the 10% test split:
```bash
python evaluate.py
```
*Outputs JSON metrics and CSV payload in `outputs/metrics/test_metrics.json` and `.csv`.*

### 3. Prediction & Cow Counting
Run static frame or temporally tracked tracking via ByteTrack across any source. The system ensures robust persistence avoiding duplicate cow-counting in videos.
```bash
# Images
python predict.py --source image.jpg

# Video
python predict.py --source video.mp4
```
*Stores predicted overlays dynamically mapped to `outputs/predictions/` and tracking logs to `outputs/metrics/counting_results.json`.*

### 4. Visualization
Compare Ground Truth predictions with generated model estimates efficiently.
```bash
python scripts/visualize_results.py
```

## Metrics & Configurations
* **Architecture:** YOLOv8 Pose.
* **Evaluation Core:** Dataset predictions are graded on `Pose mAP@50` and `Pose mAP@50-95` exclusively as requested.
* **Seed Lock:** Hardcoded random seed (`42`) mapping exact 70/20/10 splits for consistent reproducibility.
"# cv_cattle" 
