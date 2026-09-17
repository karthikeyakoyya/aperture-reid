"""
simulation.py
--------------
A synthetic two-camera pedestrian world, used as the default "demo mode"
so the whole system (re-ID engine + heatmap + live UI) is runnable and
demonstrable without needing two synced real cameras and a labelled
dataset on hand.

The physical layout: a single shared corridor, x in [0, 1]. Camera A
frames x in [0.0, 0.62]. Camera B frames x in [0.38, 1.0]. The overlap
band (x in [0.38, 0.62]) is the "handoff zone" -- exactly the region
where a real deployment would need cross-camera re-ID to avoid double
counting a person as they walk from one camera's field of view into
the other's.

Each simulated person is assigned a fixed RGB "outfit color" for their
whole walk. That color is the ground-truth appearance signal that
reid_engine.py has to recover independently in each camera -- the
simulator does NOT tell the engine which local tracks belong together;
it only ever emits per-camera Sightings, exactly as two real cameras
running independent trackers would.

Swap this module out for vision_pipeline.py to run against real video
files -- both feed the same ReIDEngine / FootfallHeatmap interfaces.
"""

from __future__ import annotations

import colorsys
import random
import time
import uuid
from dataclasses import dataclass, field

from reid_engine import Sighting, histogram_from_rgb_color

CAMERA_A_RANGE = (0.0, 0.62)
CAMERA_B_RANGE = (0.38, 1.0)
CORRIDOR_Y_JITTER = 0.06
SPAWN_INTERVAL = (0.7, 2.2)   # seconds between new pedestrians spawning
WALK_SPEED = (0.035, 0.07)    # normalized units per tick


def _random_outfit_color(rng: random.Random) -> tuple:
    """
    A wide, continuous space of plausible clothing colors rather than a
    small fixed palette. A small fixed palette causes unrelated simulated
    pedestrians to collide on the exact same appearance by chance -- which
    the re-ID engine would then (correctly, by its own logic) merge into
    one identity. Real people rarely wear pixel-identical outfits, so the
    simulator should reflect that instead of manufacturing false merges.
    Genuine appearance collisions are a real, documented failure mode of
    color-histogram re-ID (see README) -- this just keeps them rare rather
    than making them the common case in a basic demo run.
    """
    hue = rng.random()
    saturation = rng.uniform(0.45, 0.85)
    value = rng.uniform(0.55, 0.92)
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(r * 255), int(g * 255), int(b * 255))


@dataclass
class _Pedestrian:
    person_uid: str
    color: tuple
    y: float
    x: float
    direction: int  # +1 left-to-right, -1 right-to-left
    speed: float
    cam_a_track_id: int | None = None
    cam_b_track_id: int | None = None


class TwoCameraSimulation:
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._pedestrians: list[_Pedestrian] = []
        self._next_local_id = {"cam_a": 1, "cam_b": 1}
        self._next_spawn_at = 0.0
        self._t = 0.0

    def _maybe_spawn(self):
        if self._t >= self._next_spawn_at:
            direction = self._rng.choice([1, -1])
            x0 = 0.0 if direction == 1 else 1.0
            ped = _Pedestrian(
                person_uid=str(uuid.uuid4())[:6],
                color=_random_outfit_color(self._rng),
                y=0.5 + self._rng.uniform(-CORRIDOR_Y_JITTER, CORRIDOR_Y_JITTER),
                x=x0,
                direction=direction,
                speed=self._rng.uniform(*WALK_SPEED),
            )
            self._pedestrians.append(ped)
            self._next_spawn_at = self._t + self._rng.uniform(*SPAWN_INTERVAL)

    def step(self, dt: float = 0.25) -> dict:
        """
        Advance the simulation by dt seconds. Returns a dict describing
        current camera views and any new Sightings that should be fed to
        the ReIDEngine and FootfallHeatmap by the caller.
        """
        self._t += dt
        self._maybe_spawn()

        sightings: list[Sighting] = []
        floor_positions: list[dict] = []
        alive: list[_Pedestrian] = []
        sim_now = self._t

        for ped in self._pedestrians:
            ped.x += ped.direction * ped.speed * dt * 4
            if -0.05 <= ped.x <= 1.05:
                alive.append(ped)
            floor_positions.append({"x": ped.x, "y": ped.y, "color": ped.color})

            in_a = CAMERA_A_RANGE[0] <= ped.x <= CAMERA_A_RANGE[1]
            in_b = CAMERA_B_RANGE[0] <= ped.x <= CAMERA_B_RANGE[1]

            if in_a:
                if ped.cam_a_track_id is None:
                    ped.cam_a_track_id = self._next_local_id["cam_a"]
                    self._next_local_id["cam_a"] += 1
                sightings.append(self._make_sighting("cam_a", ped, CAMERA_A_RANGE, sim_now))
            else:
                ped.cam_a_track_id = None  # left the view; a re-entry gets a NEW local id

            if in_b:
                if ped.cam_b_track_id is None:
                    ped.cam_b_track_id = self._next_local_id["cam_b"]
                    self._next_local_id["cam_b"] += 1
                sightings.append(self._make_sighting("cam_b", ped, CAMERA_B_RANGE, sim_now))
            else:
                ped.cam_b_track_id = None

        self._pedestrians = alive

        return {
            "sightings": sightings,
            "floor_positions": floor_positions,
            "sim_time": self._t,
        }

    @staticmethod
    def _make_sighting(camera_id: str, ped: "_Pedestrian", cam_range: tuple, sim_now: float) -> Sighting:
        local_id = ped.cam_a_track_id if camera_id == "cam_a" else ped.cam_b_track_id
        span = cam_range[1] - cam_range[0]
        local_x = (ped.x - cam_range[0]) / span
        return Sighting(
            camera_id=camera_id,
            local_track_id=local_id,
            histogram=histogram_from_rgb_color(ped.color),
            bbox=(local_x, ped.y, 0.05, 0.14),
            color_swatch=ped.color,
            timestamp=sim_now,
            ground_truth_id=ped.person_uid,
        )
