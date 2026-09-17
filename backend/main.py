"""
main.py
-------
FastAPI app tying together the simulation (or, in real-video mode,
vision_pipeline.CameraStream), the ReIDEngine, and the FootfallHeatmap,
and streaming the combined state to the frontend over a WebSocket.

Run:
    cd backend
    uvicorn main:app --reload --port 8000

Then open frontend/index.html (it connects to ws://localhost:8000/ws/live).
"""

from __future__ import annotations

import asyncio
import time

import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from heatmap import FootfallHeatmap
from reid_engine import ReIDEngine
from simulation import TwoCameraSimulation
from anomaly import AnomalyDetector

app = FastAPI(title="Aperture -- Cross-Camera Re-ID Console")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


@app.get("/")
def index():
    return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))

engine = ReIDEngine()
heatmap = FootfallHeatmap(grid_size=44)
sim = TwoCameraSimulation()
anomaly_detector = AnomalyDetector()
_start_time = time.time()


@app.get("/api/status")
def status():
    return {
        "uptime_seconds": round(time.time() - _start_time, 1),
        "reid": engine.snapshot(),
        "footfall_events": heatmap.total_events,
    }


@app.post("/api/reset")
def reset():
    global engine, heatmap, sim, anomaly_detector
    engine = ReIDEngine()
    heatmap = FootfallHeatmap(grid_size=44)
    sim = TwoCameraSimulation()
    anomaly_detector = AnomalyDetector()
    return {"reset": True}


@app.websocket("/ws/live")
async def live_feed(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            step = sim.step(dt=0.25)

            for sighting in step["sightings"]:
                identity = engine.observe(sighting)
                heatmap.record(sighting.bbox[0], sighting.bbox[1])

            new_anomalies = anomaly_detector.scan(engine, now=step["sim_time"])

            payload = {
                "sim_time": round(step["sim_time"], 1),
                "floor_positions": step["floor_positions"],
                "reid": engine.snapshot(),
                "heatmap": heatmap.as_list(),
                "footfall_events": heatmap.total_events,
                "anomalies": [
                    {"kind": a.kind, "global_id": a.global_id, "detail": a.detail}
                    for a in new_anomalies
                ],
            }
            await websocket.send_json(payload)
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        pass
