# HerdWatch Cattle Monitoring System

Camera-based cattle detection, counting, tracking, and movement observation for
images, recorded videos, and a live browser camera.

## Setup

```powershell
pip install -r requirements.txt
cd frontend
npm install
```

## Detection model

The deployed cattle detection model must exist at:

```text
outputs/models/cattle_count_best.pt
```

Model development is intentionally kept outside the monitoring application.
The web UI and public API do not expose training or evaluation controls.

## Predict and count

```powershell
python -m core.predict --source path\to\image.jpg
python -m core.predict --source path\to\video.mp4
```

Annotated output is stored in `outputs/predictions`. The latest count and
per-cow movement conditions are stored in `outputs/metrics/counting_results.json`.

## Run the web app

In one terminal:

```powershell
uvicorn backend.main:app --reload
```

In another terminal:

```powershell
cd frontend
npm run dev
```

Open `http://localhost:5173`. The interface provides:

- Image and recorded-video analysis
- Live browser-camera monitoring
- Directional virtual-line counting with stable tracking IDs
- Configurable counting-line orientation, position, confidence, and camera ROI
- Separate cumulative crossing, forward/reverse, and visible-cattle counts
- Normal movement, low movement, high activity, and observing signals
- Detection confidence and observation duration for each tracked cow

For walkway counting, place the virtual line across the direction of travel:
use a vertical line for left/right movement or a horizontal line for up/down
movement. A tracked animal is counted once per session after it moves fully
through the line's hysteresis zone. For general recorded footage, the reported
"cattle detected" value is the larger of the repeatable peak-visible estimate
and the line-crossing count, so cattle that remain on one side of the line are
not reported as zero. Uploaded analyses use isolated job IDs and write their
metrics under `outputs/metrics/jobs`.

Movement conditions are visual observations only. They are not veterinary
diagnoses and must not be used to determine illness, pain, pregnancy, or
lameness without a qualified veterinarian.

## Retrain the lameness classifier

The labeled raw clips are read from `dataset/raw/CattleLameness/Data/Normal`
and `dataset/raw/CattleLameness/Data/Lame`. Extract tracked motion features and
train the deployed classifier with:

```powershell
python scripts/extract_features.py
python scripts/train_lameness.py
```

The extractor normalizes clips to 10 samples per second and keeps all windows
from one source video in the same evaluation split. The previous checkpoint is
backed up before `outputs/models/lame_checkpoint.joblib` is replaced. Held-out
metrics are written to `outputs/models/lame_checkpoint.metrics.json`.
