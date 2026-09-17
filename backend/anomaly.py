"""
anomaly.py
----------
Anomaly/event-correlation rules built on top of what ReIDEngine already
knows: how long a resolved identity has been present, and how long it's
been gone. Two rules, both genuinely simple and both genuinely useful,
which is the honest way to introduce "anomaly detection" rather than
reaching for something more exotic than the data supports:

1. Loitering: a global identity has been continuously present (any
   camera, any sighting) for longer than LOITER_SECONDS without leaving.
2. Reappearance-after-gap: a global identity that had gone quiet for
   longer than GAP_SECONDS shows up again. In a real deployment this is
   the flag for "someone came back after a suspiciously long absence",
   e.g. a person re-entering a restricted corridor.

Both rules only need what GlobalIdentity already tracks (first_seen,
last_seen, the sightings list) -- no new detector, no new model, just
correlation logic over identities the re-ID engine already resolved.
This is deliberately the same kind of rule-based event correlation the
JD asks for ("applying temporal/spatial rules... to generate reliable
business insights"), not a learned anomaly-detection model, which would
need labelled anomalous examples this project doesn't have.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from reid_engine import GlobalIdentity, ReIDEngine

LOITER_SECONDS = 6.0
GAP_SECONDS = 10.0


@dataclass
class AnomalyEvent:
    kind: str            # "loitering" | "reappearance"
    global_id: str
    detail: str
    timestamp: float


class AnomalyDetector:
    """
    Call `scan(engine)` once per tick (same cadence as the main loop).
    Stateful: remembers which identities have already been flagged for
    loitering so it doesn't repeat the same alert every tick, and
    remembers each identity's last-seen time from the previous scan so
    it can detect a gap followed by a reappearance.
    """

    def __init__(self, loiter_seconds: float = LOITER_SECONDS, gap_seconds: float = GAP_SECONDS):
        self.loiter_seconds = loiter_seconds
        self.gap_seconds = gap_seconds
        self._already_flagged_loiter: set[str] = set()
        self._last_seen_at_prev_scan: dict[str, float] = {}
        self.events: list[AnomalyEvent] = []

    def scan(self, engine: ReIDEngine, now: float | None = None) -> list[AnomalyEvent]:
        now = now if now is not None else time.time()
        new_events: list[AnomalyEvent] = []

        for gid, identity in engine.identities.items():
            present_duration = identity.last_seen - identity.first_seen

            if (
                present_duration >= self.loiter_seconds
                and gid not in self._already_flagged_loiter
            ):
                self._already_flagged_loiter.add(gid)
                new_events.append(AnomalyEvent(
                    kind="loitering",
                    global_id=gid,
                    detail=f"present for {present_duration:.1f}s without leaving either camera's view",
                    timestamp=now,
                ))

            prev_last_seen = self._last_seen_at_prev_scan.get(gid)
            if prev_last_seen is not None:
                gap = now - prev_last_seen
                # a reappearance is detected when we see a NEW sighting
                # after a gap that had already exceeded the threshold
                if gap >= self.gap_seconds and identity.last_seen > prev_last_seen:
                    new_events.append(AnomalyEvent(
                        kind="reappearance",
                        global_id=gid,
                        detail=f"reappeared after a {gap:.1f}s absence",
                        timestamp=now,
                    ))

            self._last_seen_at_prev_scan[gid] = identity.last_seen

        self.events.extend(new_events)
        return new_events

    def reset(self):
        self._already_flagged_loiter.clear()
        self._last_seen_at_prev_scan.clear()
        self.events.clear()
