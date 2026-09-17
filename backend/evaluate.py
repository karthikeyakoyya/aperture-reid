"""
evaluate.py
-----------
An honest answer to "what's your actual accuracy number?" -- the one
thing this project didn't have until now.

The simulator (simulation.py) knows the true identity of every synthetic
pedestrian (Sighting.ground_truth_id) but NEVER passes it to the ReIDEngine
-- the engine only ever sees camera_id, local_track_id, and an HSV
histogram, exactly like it would with real, unlabelled camera footage.
This script is the only place the ground truth is used: after the engine
has made all its matching decisions blind, we grade them.

Metric: pairwise precision / recall / F1 over every pair of cross-camera
local tracks.
  - A pair "should link" if both local tracks belong to the same
    ground-truth person.
  - A pair "did link" if the engine assigned them the same global_id.
  - True positive:  should link AND did link
  - False positive: should NOT link BUT did link   (a false merge --
                     exactly the color-collision failure mode documented
                     in README.md)
  - False negative: should link BUT did NOT link   (a missed handoff)

This is standard pairwise evaluation methodology for entity resolution /
re-identification tasks, not something invented for this project.

Run:
    python evaluate.py                  # default: 400 ticks, seed 3
    python evaluate.py --ticks 800 --seed 11 --runs 5
"""

from __future__ import annotations

import argparse
import statistics
import time
from itertools import combinations

from reid_engine import ReIDEngine
from simulation import TwoCameraSimulation


def run_once(ticks: int, seed: int, dt: float = 0.25) -> dict:
    sim = TwoCameraSimulation(seed=seed)
    engine = ReIDEngine()

    sightings_processed = 0
    t0 = time.perf_counter()
    for _ in range(ticks):
        step = sim.step(dt=dt)
        for sighting in step["sightings"]:
            engine.observe(sighting)
            sightings_processed += 1
    elapsed = time.perf_counter() - t0

    # Build one row per distinct (camera_id, local_track_id) with the
    # global_id the engine settled on and the ground truth it never saw.
    rows = []
    for gid, identity in engine.identities.items():
        for s in identity.sightings:
            rows.append((s.camera_id, s.local_track_id, gid, s.ground_truth_id))
    # de-duplicate to one row per local track (first sighting is enough,
    # the global_id assignment for a local track never changes after
    # first resolution)
    seen = {}
    for camera_id, local_id, gid, truth_id in rows:
        seen[(camera_id, local_id)] = (gid, truth_id)

    tracks = list(seen.items())
    tp = fp = fn = 0
    for (a_key, (a_gid, a_truth)), (b_key, (b_gid, b_truth)) in combinations(tracks, 2):
        if a_key[0] == b_key[0]:
            continue  # same camera: not a cross-camera re-ID decision
        should_link = a_truth == b_truth
        did_link = a_gid == b_gid
        if should_link and did_link:
            tp += 1
        elif did_link and not should_link:
            fp += 1
        elif should_link and not did_link:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and (precision + recall) > 0
          else float("nan"))

    return {
        "ticks": ticks,
        "seed": seed,
        "sightings_processed": sightings_processed,
        "elapsed_seconds": elapsed,
        "throughput_sightings_per_sec": sightings_processed / elapsed if elapsed > 0 else float("inf"),
        "cross_camera_track_pairs_evaluated": tp + fp + fn,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=400)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--runs", type=int, default=5, help="repeat with consecutive seeds and average")
    parser.add_argument("--min-precision", type=float, default=None,
                         help="exit with status 1 if average precision falls below this (for CI gating)")
    parser.add_argument("--min-recall", type=float, default=None,
                         help="exit with status 1 if average recall falls below this (for CI gating)")
    args = parser.parse_args()

    results = [run_once(args.ticks, args.seed + i) for i in range(args.runs)]

    print(f"\n{'seed':>6} {'pairs':>8} {'tp':>5} {'fp':>5} {'fn':>5} {'precision':>10} {'recall':>8} {'f1':>8} {'sightings/sec':>14}")
    for r in results:
        print(f"{r['seed']:>6} {r['cross_camera_track_pairs_evaluated']:>8} {r['tp']:>5} {r['fp']:>5} {r['fn']:>5} "
              f"{r['precision']:>10.3f} {r['recall']:>8.3f} {r['f1']:>8.3f} {r['throughput_sightings_per_sec']:>14.0f}")

    precisions = [r["precision"] for r in results if r["precision"] == r["precision"]]
    recalls = [r["recall"] for r in results if r["recall"] == r["recall"]]
    f1s = [r["f1"] for r in results if r["f1"] == r["f1"]]
    throughputs = [r["throughput_sightings_per_sec"] for r in results]

    print("\naverage over", len(results), "runs:")
    print(f"  precision: {statistics.mean(precisions):.3f}")
    print(f"  recall:    {statistics.mean(recalls):.3f}")
    print(f"  f1:        {statistics.mean(f1s):.3f}")
    print(f"  throughput: {statistics.mean(throughputs):.0f} sightings/sec (single CPU core, no GPU)")

    mean_precision = statistics.mean(precisions)
    mean_recall = statistics.mean(recalls)
    failed = False
    if args.min_precision is not None and mean_precision < args.min_precision:
        print(f"\nFAIL: precision {mean_precision:.3f} is below --min-precision {args.min_precision}")
        failed = True
    if args.min_recall is not None and mean_recall < args.min_recall:
        print(f"FAIL: recall {mean_recall:.3f} is below --min-recall {args.min_recall}")
        failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
