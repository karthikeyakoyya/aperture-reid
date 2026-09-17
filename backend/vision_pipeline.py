"""
vision_pipeline.py
-------------------
The real-camera counterpart to simulation.py. Point this at two actual
video files (or RTSP/webcam streams) that cover adjacent or overlapping
physical space, and it emits the same Sighting objects that reid_engine.py
consumes -- so everything downstream (matching, heatmap, UI) is identical
whether the source is the synthetic demo or real footage.

Requires: pip install ultralytics opencv-python
(kept out of the default requirements.txt so the demo mode has zero
heavy dependencies; see requirements-real-video.txt)

Usage:
    from vision_pipeline import CameraStream
    cam_a = CameraStream("cam_a", "videos/entrance.mp4", floor_homography=H_A)
    cam_b = CameraStream("cam_b", "videos/lobby.mp4", floor_homography=H_B)
    for sighting in cam_a.sightings():
        engine.observe(sighting)

floor_homography: a 3x3 matrix mapping this camera's pixel coordinates to
the shared normalized floor-plan (0..1, 0..1) used by heatmap.py. In a
real deployment you compute this once per camera with cv2.findHomography
against four marked reference points visible in both the frame and a
floor-plan sketch. It defaults to an identity-like passthrough so the
pipeline still runs (with a rough approximation) before that calibration
step is done.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from reid_engine import Sighting, histogram_from_patch

try:
    from ultralytics import YOLO
    import cv2
    _HAS_VISION_DEPS = True
except ImportError:
    _HAS_VISION_DEPS = False

PERSON_CLASS_ID = 0  # COCO class index for "person" in the stock YOLOv8 weights
CONFIDENCE_THRESHOLD = 0.4


@dataclass
class FloorCalibration:
    """Four point-correspondences: pixel (x, y) -> floor-plan (x, y) in 0..1."""
    pixel_points: list
    floor_points: list

    def homography(self):
        src = np.array(self.pixel_points, dtype=np.float32)
        dst = np.array(self.floor_points, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        return H


class CameraStream:
    """
    Wraps one video source: runs YOLOv8 detection + the built-in ByteTrack
    tracker (via ultralytics' `model.track(..., tracker="bytetrack.yaml")`),
    crops each tracked person, computes an HSV appearance descriptor, and
    projects their foot-point into shared floor-plan coordinates.
    """

    def __init__(
        self,
        camera_id: str,
        source: str,
        model_weights: str = "yolov8n.pt",
        floor_homography: np.ndarray | None = None,
    ):
        if not _HAS_VISION_DEPS:
            raise RuntimeError(
                "Real-video mode needs `pip install ultralytics opencv-python`. "
                "Demo mode (simulation.py) has no such dependency."
            )
        self.camera_id = camera_id
        self.source = source
        self.model = YOLO(model_weights)
        self.homography = floor_homography

    def _project_to_floor(self, foot_x: float, foot_y: float) -> tuple:
        if self.homography is None:
            # no calibration yet: fall back to raw normalized pixel position
            return foot_x, foot_y
        pt = np.array([foot_x, foot_y, 1.0])
        mapped = self.homography @ pt
        mapped /= mapped[2]
        return float(mapped[0]), float(mapped[1])

    def sightings(self):
        """Generator of Sighting objects, one batch per processed frame."""
        results = self.model.track(
            source=self.source,
            classes=[PERSON_CLASS_ID],
            conf=CONFIDENCE_THRESHOLD,
            tracker="bytetrack.yaml",
            stream=True,
            verbose=False,
        )
        for frame_result in results:
            frame = frame_result.orig_img
            h, w = frame.shape[:2]
            boxes = frame_result.boxes
            if boxes is None or boxes.id is None:
                continue
            now = time.time()
            for box, track_id in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy()):
                x1, y1, x2, y2 = box
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0:
                    continue
                histogram = histogram_from_patch(crop)
                foot_x_px, foot_y_px = (x1 + x2) / 2.0 / w, y2 / h
                floor_x, floor_y = self._project_to_floor(foot_x_px, foot_y_px)
                mean_color = crop.reshape(-1, 3).mean(axis=0)[::-1]  # bgr -> rgb
                yield Sighting(
                    camera_id=self.camera_id,
                    local_track_id=int(track_id),
                    histogram=histogram,
                    bbox=(x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h),
                    color_swatch=tuple(int(c) for c in mean_color),
                    timestamp=now,
                )


# ---------------------------------------------------------------------------
# RTSP / live streams
# ---------------------------------------------------------------------------
# `source` above is passed straight through to ultralytics' `model.track()`,
# which natively accepts an RTSP URL exactly like a file path -- no code
# change needed:
#
#     cam_a = CameraStream("cam_a", "rtsp://192.168.1.50:554/stream1")
#
# Verification status, stated plainly: this is a documented ultralytics
# capability (https://docs.ultralytics.com/modes/predict/#inference-sources),
# not something exercised against a live camera in this sandbox -- there is
# no RTSP source available here to test against. Point it at a real stream
# (a phone RTSP-server app works for a quick check) before relying on it.


# ---------------------------------------------------------------------------
# Segmentation variant (untested in this sandbox, code-only)
# ---------------------------------------------------------------------------
class SegmentingCameraStream(CameraStream):
    """
    Same interface as CameraStream, but loads a YOLOv8 *segmentation*
    checkpoint (e.g. "yolov8n-seg.pt") instead of a detection one, and
    keeps each person's pixel mask alongside their box. Re-ID matching
    itself is unchanged (still HSV histogram over the crop) -- what a
    mask adds is a tighter crop (foreground pixels only, background
    excluded) for the histogram, which should reduce background-color
    contamination in cluttered scenes.

    Verification status, stated plainly: this class has not been run
    against real or synthetic video in this sandbox (no segmentation
    weights were downloaded or exercised here). It is a documented,
    reasonable extension of CameraStream, not a claimed, tested feature --
    treat it as a starting point, not a result.
    """

    def sightings(self):
        results = self.model.track(
            source=self.source,
            classes=[PERSON_CLASS_ID],
            conf=CONFIDENCE_THRESHOLD,
            tracker="bytetrack.yaml",
            stream=True,
            verbose=False,
        )
        for frame_result in results:
            frame = frame_result.orig_img
            h, w = frame.shape[:2]
            boxes = frame_result.boxes
            masks = frame_result.masks
            if boxes is None or boxes.id is None:
                continue
            now = time.time()
            for i, (box, track_id) in enumerate(zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy())):
                x1, y1, x2, y2 = box
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0:
                    continue
                if masks is not None:
                    # zero out background pixels inside the box using the
                    # instance mask, so the histogram only sees the person
                    mask = masks.data[i].cpu().numpy()
                    mask_crop = mask[int(y1):int(y2), int(x1):int(x2)]
                    mask_crop = (mask_crop > 0.5).astype("uint8")[..., None]
                    crop = crop * mask_crop
                histogram = histogram_from_patch(crop)
                foot_x_px, foot_y_px = (x1 + x2) / 2.0 / w, y2 / h
                floor_x, floor_y = self._project_to_floor(foot_x_px, foot_y_px)
                mean_color = crop.reshape(-1, 3).mean(axis=0)[::-1]
                yield Sighting(
                    camera_id=self.camera_id,
                    local_track_id=int(track_id),
                    histogram=histogram,
                    bbox=(x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h),
                    color_swatch=tuple(int(c) for c in mean_color),
                    timestamp=now,
                )

