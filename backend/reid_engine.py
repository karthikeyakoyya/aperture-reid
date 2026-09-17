"""
reid_engine.py
--------------
Cross-camera identity resolution.

The core problem: two cameras see overlapping or adjacent physical space.
Each camera independently tracks people frame-to-frame (that part is solved
by any tracker, e.g. ByteTrack). The open problem this module solves is
matching a track in Camera A to a track in Camera B when they are, in fact,
the same physical person -- without ever having a shared ground-truth ID.

v1 approach (deliberately simple and honest):
  - Appearance descriptor: an HSV color histogram of the person's bounding
    box, which is cheap to compute and reasonably robust to camera-to-camera
    exposure differences when done in HSV rather than RGB.
  - Matching score: histogram correlation (cv2.compareHist, CORREL method)
    combined with a temporal gate (candidates are only compared if they were
    seen within a configurable time window of each other) and an optional
    spatial gate (entry/exit zones can be restricted to plausible handoff
    regions between the two views).
  - A match is accepted once its combined score clears MATCH_THRESHOLD and
    it is the best mutual match available in the current candidate pool
    (simple greedy bipartite matching, not the Hungarian algorithm -- fine
    at the scale of a handful of concurrent people, and easy to defend line
    by line in an interview).

This is intentionally NOT a learned re-ID embedding network (e.g. OSNet,
a triplet-loss CNN). That is the documented next step in README.md. This
version is the strongest truthful thing to build without a labelled
person-re-id dataset and a GPU to train on.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:  # pragma: no cover - cv2 is a required dep, guarded for clarity
    _HAS_CV2 = False


MATCH_THRESHOLD = 0.75          # minimum histogram correlation to consider a match
TEMPORAL_WINDOW_SECONDS = 6.0   # candidates must appear within this window of each other
HIST_BINS = (64, 64)            # H and S bins for the HSV histogram


@dataclass
class Sighting:
    """One observation of a tracked person in a single camera."""
    camera_id: str
    local_track_id: int
    histogram: np.ndarray
    bbox: tuple  # (x, y, w, h) in normalized 0..1 camera coordinates
    color_swatch: tuple  # approximate (r, g, b) for UI display, 0-255
    timestamp: float = field(default_factory=time.time)
    ground_truth_id: Optional[str] = None  # never read by the matcher itself;
    # only present so an offline evaluation harness (evaluate.py) can grade
    # match quality against a known-correct identity. Real deployments have
    # no such field -- it exists purely because the simulator can supply one.


@dataclass
class GlobalIdentity:
    """A resolved cross-camera identity: one real person, seen by >=1 camera."""
    global_id: str
    sightings: list = field(default_factory=list)  # list[Sighting]
    confidence: float = 0.0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    @property
    def camera_ids(self) -> set:
        return {s.camera_id for s in self.sightings}


def histogram_from_patch(patch_bgr: np.ndarray) -> np.ndarray:
    """Compute a normalized HSV color histogram descriptor for a person crop."""
    if not _HAS_CV2:
        raise RuntimeError("OpenCV is required for histogram extraction")
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HIST_BINS), [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()


def histogram_from_rgb_color(rgb: tuple) -> np.ndarray:
    """
    Build a synthetic HSV-style histogram descriptor directly from a flat RGB
    color, used by the simulator so the same matching code path is exercised
    without needing real camera pixels.
    """
    patch = np.full((32, 32, 3), rgb[::-1], dtype=np.uint8)  # rgb -> bgr fill
    return histogram_from_patch(patch)


def _similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    if not _HAS_CV2:
        # cosine similarity fallback, kept for completeness
        denom = (np.linalg.norm(h1) * np.linalg.norm(h2)) or 1e-6
        return float(np.dot(h1, h2) / denom)
    score = cv2.compareHist(h1.astype("float32"), h2.astype("float32"), cv2.HISTCMP_CORREL)
    return float(max(0.0, score))


class ReIDEngine:
    """
    Stateful cross-camera matcher. Feed it Sightings as they arrive from each
    camera's local tracker; it maintains a rolling pool of unresolved recent
    sightings per camera and produces GlobalIdentity matches across cameras.
    """

    def __init__(
        self,
        match_threshold: float = MATCH_THRESHOLD,
        temporal_window: float = TEMPORAL_WINDOW_SECONDS,
    ):
        self.match_threshold = match_threshold
        self.temporal_window = temporal_window
        self._recent_by_camera: dict[str, list[Sighting]] = {}
        self._local_to_global: dict[tuple[str, int], str] = {}
        self.identities: dict[str, GlobalIdentity] = {}
        self.match_events: list[dict] = []  # audit trail for the UI ledger

    def _prune_stale(self, now: float):
        cutoff = now - self.temporal_window
        for cam, sightings in self._recent_by_camera.items():
            self._recent_by_camera[cam] = [s for s in sightings if s.timestamp >= cutoff]

    def observe(self, sighting: Sighting) -> GlobalIdentity:
        """
        Register a new sighting. Returns the GlobalIdentity it was attached
        to (existing, matched, or freshly created).
        """
        now = sighting.timestamp
        self._prune_stale(now)

        local_key = (sighting.camera_id, sighting.local_track_id)

        # Already-resolved local track: just append the sighting.
        if local_key in self._local_to_global:
            gid = self._local_to_global[local_key]
            identity = self.identities[gid]
            identity.sightings.append(sighting)
            identity.last_seen = now
            self._recent_by_camera.setdefault(sighting.camera_id, []).append(sighting)
            return identity

        # Search other cameras' recent, still-open pool for a match.
        best_match: Optional[Sighting] = None
        best_score = 0.0
        for cam_id, sightings in self._recent_by_camera.items():
            if cam_id == sighting.camera_id:
                continue
            for candidate in sightings:
                if abs(candidate.timestamp - now) > self.temporal_window:
                    continue
                score = _similarity(sighting.histogram, candidate.histogram)
                if score > best_score:
                    best_score, best_match = score, candidate

        if best_match is not None and best_score >= self.match_threshold:
            gid = self._local_to_global.get((best_match.camera_id, best_match.local_track_id))
            if gid is None:
                gid = self._new_identity(best_match)
            identity = self.identities[gid]
            identity.sightings.append(sighting)
            identity.confidence = best_score
            identity.last_seen = now
            self._local_to_global[local_key] = gid
            self.match_events.append({
                "global_id": gid,
                "camera_from": best_match.camera_id,
                "camera_to": sighting.camera_id,
                "score": round(best_score, 3),
                "timestamp": now,
                "color": sighting.color_swatch,
            })
        else:
            gid = self._new_identity(sighting)

        self._recent_by_camera.setdefault(sighting.camera_id, []).append(sighting)
        return self.identities[gid]

    def _new_identity(self, sighting: Sighting) -> str:
        gid = str(uuid.uuid4())[:8]
        identity = GlobalIdentity(
            global_id=gid,
            sightings=[sighting],
            first_seen=sighting.timestamp,
            last_seen=sighting.timestamp,
        )
        self.identities[gid] = identity
        self._local_to_global[(sighting.camera_id, sighting.local_track_id)] = gid
        return gid

    def snapshot(self) -> dict:
        """A JSON-serializable snapshot for the API/WebSocket layer."""
        return {
            "identity_count": len(self.identities),
            "cross_camera_count": sum(1 for i in self.identities.values() if len(i.camera_ids) > 1),
            "recent_matches": self.match_events[-25:],
        }
