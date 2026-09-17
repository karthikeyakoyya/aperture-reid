"""
onnx_latency_benchmark.py
--------------------------
NOT RUN IN THE BUILD SANDBOX. This script needs `torch` + `ultralytics`,
which could not be installed here (ran out of disk space mid-session).
It is real, standard code you can run on your own machine to get a real
number -- do not put a result on your resume until you've actually run
this and seen the printed table yourself.

What it does: loads a YOLOv8 detection model, exports it to ONNX, and
times single-frame CPU inference three ways (PyTorch eager, TorchScript,
ONNX Runtime) so you can report a real, defensible latency/FPS comparison
instead of an estimate.

Run on your own machine:
    pip install ultralytics onnxruntime
    python onnx_latency_benchmark.py

Swap MODEL_PATH for your own trained weights (e.g. the Cervical Cell
Classifier's MobileNetV2, if you export that to ONNX the same way) to get
numbers specific to your own model rather than stock YOLOv8n.
"""

import time

import numpy as np

MODEL_PATH = "yolov8n.pt"   # swap for your own trained weights if you have an ONNX-exportable model
N_WARMUP = 5
N_TIMED = 50
INPUT_SIZE = 640  # YOLOv8's default input resolution


def main():
    from ultralytics import YOLO

    print(f"Loading {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)

    dummy_image = np.random.randint(0, 255, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)

    # ---- PyTorch eager ----
    for _ in range(N_WARMUP):
        model.predict(dummy_image, verbose=False)
    t0 = time.perf_counter()
    for _ in range(N_TIMED):
        model.predict(dummy_image, verbose=False)
    eager_ms = (time.perf_counter() - t0) / N_TIMED * 1000

    # ---- TorchScript ----
    ts_path = model.export(format="torchscript")
    ts_model = YOLO(ts_path)
    for _ in range(N_WARMUP):
        ts_model.predict(dummy_image, verbose=False)
    t0 = time.perf_counter()
    for _ in range(N_TIMED):
        ts_model.predict(dummy_image, verbose=False)
    torchscript_ms = (time.perf_counter() - t0) / N_TIMED * 1000

    # ---- ONNX Runtime ----
    onnx_path = model.export(format="onnx")
    onnx_model = YOLO(onnx_path)
    for _ in range(N_WARMUP):
        onnx_model.predict(dummy_image, verbose=False)
    t0 = time.perf_counter()
    for _ in range(N_TIMED):
        onnx_model.predict(dummy_image, verbose=False)
    onnx_ms = (time.perf_counter() - t0) / N_TIMED * 1000

    print(f"\n{'path':<15}{'mean latency (ms)':>20}{'FPS':>10}")
    for label, ms in [("PyTorch eager", eager_ms), ("TorchScript", torchscript_ms), ("ONNX Runtime", onnx_ms)]:
        print(f"{label:<15}{ms:>20.2f}{1000/ms:>10.1f}")

    print(f"\nONNX Runtime vs PyTorch eager: {eager_ms / onnx_ms:.2f}x")
    print("These numbers are specific to the machine you ran this on -- report")
    print("them as such, not as universal constants.")


if __name__ == "__main__":
    main()
