# Getting a real mAP/IoU number (not run in this sandbox)

This needs two things this sandbox doesn't have: your actual warehouse
footage, and time spent drawing boxes. Here's the real path, no shortcuts.

## 1. Label ~30-50 frames

Pull 30-50 representative frames from your Warehouse Auditor footage
(varied angles, lighting, occlusion). Use a free tool:

- **Roboflow** (roboflow.com, free tier) -- browser-based, exports directly
  to YOLO format, easiest if you've never labeled before.
- **LabelImg** (`pip install labelImg`) -- desktop app, also exports YOLO
  format, fully offline if you'd rather not upload footage anywhere.

Draw a box around every person/box/shelf-item you care about detecting.
20-30 minutes of clicking for 30-50 frames is a realistic estimate.

## 2. Point YOLOv8's validation pipeline at your labels

Export gives you a folder structure like:

```
my_validation_set/
  images/
    frame_001.jpg
    ...
  labels/
    frame_001.txt
    ...
  data.yaml
```

Then, on your own machine:

```bash
pip install ultralytics
```

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # or your own fine-tuned weights, if you have them
metrics = model.val(data="my_validation_set/data.yaml")

print("mAP@0.5:", metrics.box.map50)
print("mAP@0.5:0.95:", metrics.box.map)
print("precision:", metrics.box.mp)
print("recall:", metrics.box.mr)
```

This prints real numbers computed against boxes you actually drew, on
footage you actually have. That's the only honest way to get a
project-specific mAP -- a generic COCO benchmark number would describe
stock YOLOv8, not your project.

## Why this isn't already done

No hand-labeled validation set exists yet for the Warehouse Auditor
footage, and building one needs the actual video files and manual
labeling time neither of which are available in the sandbox that built
Aperture. This is real, sequenced work, not something to fake with an
invented percentage.
