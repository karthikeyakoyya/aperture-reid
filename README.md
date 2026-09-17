# Aperture: Cross-Camera Identity Resolution & Footfall Console

Two cameras watching adjacent space, each running its own independent
tracker with its own local ID numbering. Aperture answers the question
neither camera can answer alone: **is the person Camera B just picked up
the same one Camera A just lost, or someone new?** It uses that answer
to build one continuous footfall map instead of two disconnected ones.

This is a from-scratch project built to close a specific gap: most
fresher computer-vision portfolios stop at single-camera detection plus
tracking. Cross-camera re-identification is the part almost nobody
attempts, because there's no off-the-shelf labelled dataset and no
shared ground-truth ID to check answers against. So this project builds
the smallest honest version of it, end to end, with a clear documented
path to a stronger v2.

## What's actually in this repo

```
backend/
  main.py                     FastAPI app + live WebSocket feed
  simulation.py               synthetic two-camera pedestrian world (demo mode)
  reid_engine.py               the actual cross-camera matching logic
  anomaly.py                   loitering / reappearance event-correlation rules
  evaluate.py                  ground-truth precision/recall/F1 harness (also CI-gates)
  heatmap.py                   merged floor-plan footfall accumulator
  vision_pipeline.py           real-video mode: YOLOv8 + ByteTrack + HSV descriptor
                                (+ SegmentingCameraStream variant, untested, see below)
  onnx_latency_benchmark.py    ONNX Runtime vs PyTorch latency script (run on your own machine)
  REAL_MAP_EVAL.md             how to get a real mAP/IoU number on your own footage
  requirements.txt             demo-mode dependencies (no torch, no GPU needed)
  requirements-real-video.txt  extra deps only needed for real footage
frontend/
  index.html                   single-file live console (no build step)
.github/workflows/
  reid-eval.yml                CI regression gate: runs evaluate.py on every push
```

## Run the demo (2 minutes, no camera hardware needed)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** in a browser. You'll see two synthetic
camera panels sharing a corridor with an overlapping "handoff zone."
Watch a colored figure walk out of Camera A's frame and reappear in
Camera B's; when the engine matches them, a copper ring pulses once
across the panel boundary and the resolution ledger logs the match with
a confidence score. The footfall panel below accumulates a single merged
heat map across both cameras' floor space.

## How the matching actually works (v1, honestly scoped)

Each tracked person gets an **HSV color-histogram descriptor** of their
bounding box: cheap to compute, and more robust to camera-to-camera
exposure/white-balance differences than a raw RGB comparison. A new
sighting in one camera is compared against the pool of recent sightings
from *other* cameras inside a short time window (`reid_engine.py`,
`TEMPORAL_WINDOW_SECONDS = 6.0`); if the best histogram correlation
clears `MATCH_THRESHOLD = 0.55`, it's accepted as the same person.

This is **not** a learned re-ID embedding network (no OSNet, no
triplet-loss CNN). That needs a labelled multi-camera person dataset
and a GPU to train on, which this project doesn't have. Color histograms
are the strongest truthful thing to build without those. They work well
when people wear visually distinct clothing and poorly when two people
are dressed alike or lighting shifts hard between cameras, which is the
real, defensible limitation to walk an interviewer through. A subtler
version of the same limitation: the histogram is bucketed into 16x16
hue/saturation bins (`HIST_BINS` in `reid_engine.py`), so two genuinely
different but visually close shades can quantize into the same bin and
read as a perfect match; you'll occasionally see this in the demo's
own resolution ledger. Finer binning trades that away for more
sensitivity to real lighting noise, which is the actual tradeoff a v2
embedding model is meant to resolve properly.

**Documented v2 path:** swap `histogram_from_patch()` for a pretrained
lightweight re-ID embedding (e.g. OSNet via `torchreid`), replace
`HISTCMP_CORREL` with cosine distance on the embedding, and switch the
greedy match in `ReIDEngine.observe()` for a proper Hungarian assignment
(`scipy.optimize.linear_sum_assignment`) once more than a couple of
people can be in the handoff zone at once.

## Running it on real video instead of the simulation

`vision_pipeline.py` runs the identical `Sighting` to `ReIDEngine` to
`FootfallHeatmap` pipeline against real footage:

```bash
pip install -r requirements-real-video.txt
```

```python
from vision_pipeline import CameraStream
from reid_engine import ReIDEngine

engine = ReIDEngine()
cam_a = CameraStream("cam_a", "videos/entrance.mp4")
cam_b = CameraStream("cam_b", "videos/lobby.mp4")

for sighting in cam_a.sightings():
    engine.observe(sighting)
```

It uses `ultralytics` YOLOv8 for detection and its built-in ByteTrack
integration for per-camera tracking, then crops each tracked person and
runs the same HSV descriptor extraction as the simulator. For a real
deployment, calibrate `FloorCalibration` per camera (four point
correspondences from pixel space to a shared floor-plan sketch, solved
with `cv2.findHomography`) so both cameras' footfall lands on the same
coordinate system. That step is stubbed with an identity passthrough
until you do it.

## Where this maps onto the JD

- **Object detection & multi-object tracking**: YOLOv8 + ByteTrack per
  camera (`vision_pipeline.py`), same stack as the Warehouse Auditor
  project, now doing tracking handoff across views instead of within one.
- **Re-identification**: the actual subject of this project.
- **Anomaly / event correlation across detections, temporal & spatial
  rules**: the temporal window gate and the handoff-zone geometry in
  `simulation.py` are exactly this, correlating detections and filtering
  which candidates are even eligible to match.
- **Dataset/tooling mindset**: `FloorCalibration` and the clean
  `Sighting` interface are built so a real deployment's calibration step
  and dataset are the only things that change; the matching and heatmap
  code doesn't.

## Measured accuracy (not just a demo)

Every claim above was checkable, but none of it was a *number* until
`evaluate.py` existed. The simulator secretly knows each pedestrian's true
identity (`Sighting.ground_truth_id`) but never gives it to `ReIDEngine`,
the engine matches blind, exactly as it would on real unlabelled footage.
`evaluate.py` is the only place that ground truth gets used, and only
after the fact, to grade the engine's own decisions with standard
pairwise precision/recall/F1 over every cross-camera track pair.

```bash
cd backend
python evaluate.py --ticks 400 --seed 1 --runs 20
```

Result over 20 independent runs (current defaults: 64x64 HSV bins,
match threshold 0.75):

| metric | value |
|---|---|
| precision | 0.978 |
| recall | 1.000 |
| F1 | 0.989 |
| throughput | ~3,600 sightings/sec (single CPU core, no GPU) |

Getting here was a real tuning pass, not a lucky first run. The original
16x16-bin, 0.55-threshold config scored 0.635 precision / 0.998 recall
(F1 0.776), recall was already excellent, but coarse histogram bins were
letting visually-similar-but-different outfit colors collide into false
merges, the same failure mode named earlier in this README. Sweeping bin
resolution and threshold against this harness found 64x64 bins / 0.75
threshold as a stable middle point, not the top score on the sweep
(80x80 bins hit 0.987 precision), but deliberately not pushed that far,
because finer bins fit the simulator's flat, noise-free colors better
than they would fit real camera pixels with lighting variation. That
tradeoff, and why it matters, is the actual point of building the
harness.

**Caveat, stated plainly:** this is precision/recall against a synthetic
simulator's ground truth, not against labelled real video, there's no
public multi-camera re-ID dataset lying around to test against instead.
The throughput number is the matching engine's raw processing speed
(sightings per second through `ReIDEngine.observe`), not video frames
per second end to end, an honest distinction, since this project has no
GPU-accelerated detection benchmark to report yet (see gaps below).

## Loitering / event-correlation rules (`anomaly.py`)

Re-ID resolves *who*; `anomaly.py` is the "so what" layer the JD calls
"applying temporal/spatial rules... to generate reliable business
insights." Two rules, both tested against the simulator:

- **Loitering**: a resolved global identity present continuously (either
  camera) for longer than `LOITER_SECONDS` (default 6.0s) gets flagged
  once, not every tick.
- **Reappearance**: a global identity that had gone quiet for longer than
  `GAP_SECONDS` (default 10.0s) shows up again.

Tested against 200 simulated ticks (seed 5): 32 people passed through,
2 were flagged loitering, 0 false reappearance flags. The honest caveat:
the loitering rule currently measures *total presence duration*, not
*stillness* -- it can't yet tell a slow walker from someone standing
still, so a couple of naturally slow walkers get flagged alongside any
genuine loiterers. A real v2 would check position variance over the
dwell window, not just elapsed time. Results show up live as "Loitering
alerts" in the frontend.

## CI regression gate (`.github/workflows/reid-eval.yml`)

`evaluate.py` now takes `--min-precision` / `--min-recall` and exits with
status 1 if the tuned matcher's measured performance drops below them --
tested directly: exit 0 when thresholds are achievable, exit 1 with a
clear message when they aren't. The workflow runs this on every push
that touches `backend/`, so a change that quietly breaks the matching
logic fails CI instead of shipping silently. This is the smallest honest
version of an "automated evaluation pipeline": a regression gate on
measured behavior, not a training pipeline.

## What's real vs. what's a documented next step

Being direct about verification status, because it matters more than
looking complete:

| Capability | Status |
|---|---|
| Cross-camera re-ID, tuned + measured | **Tested**: 97.8% precision, 100% recall, 20 runs |
| Loitering / reappearance rules | **Tested**: against the simulator, caveat noted above |
| CI regression gate | **Tested**: exit codes verified for both pass and fail |
| RTSP as a video source | Documented `ultralytics` capability, **not exercised** against a live stream here |
| Segmentation (`SegmentingCameraStream`) | Written, syntax-checked, **not run** against real or synthetic video |
| ONNX Runtime latency benchmark | Script provided (`onnx_latency_benchmark.py`), **not run** here (no torch install fit in this sandbox) |
| mAP/IoU on real detections | Not done. See `REAL_MAP_EVAL.md` for the actual labeling + validation steps |
| OCR | Not attempted. Doesn't fit this project's actual scenario (no text-bearing objects in a pedestrian corridor) -- a shelf-label OCR pass belongs on the Warehouse Auditor project instead |

Nothing in the "not run" rows appears as a resume claim or a reported
number anywhere in this repo. A script that exists is not the same as a
result -- run it yourself before citing it.

## Honest gaps this project does not close

- No mAP/IoU/MOTA on real footage yet. `REAL_MAP_EVAL.md` has the actual
  labeling + validation steps; the script above (`onnx_latency_benchmark.py`)
  and the segmentation variant are similarly written but unrun.
- No TensorRT/OpenVINO. ONNX export is scripted (`onnx_latency_benchmark.py`)
  but not executed here -- the sandbox that built this project ran out of
  disk space installing `torch`/`ultralytics`.
