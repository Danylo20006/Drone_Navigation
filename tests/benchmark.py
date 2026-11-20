"""Engineering benchmark for slide metrics: Full-map vs ROI, SIFT vs ORB, CLAHE On vs Off."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import time
import cv2
import numpy as np

from src.io_utils import load_frames_from_dir, load_frames_from_video
from src.matcher import match_frame
from src.tracker import DroneTracker, TrackerConfig


def bench_full_map(
    map_img: np.ndarray,
    frames: list[np.ndarray],
    method: str = "sift",
    crop_scale: float = 0.25,
    use_clahe: bool = False,
    min_inliers: int = 12,
) -> dict[str, float | int | str]:
    """Run match_frame on the entire map for every frame without ROI cropping."""
    t0 = time.perf_counter()
    valid = 0
    inliers_sum = 0
    for frame in frames:
        H, inliers = match_frame(
            map_img,
            frame,
            method=method,
            crop_scale=crop_scale,
            use_clahe=use_clahe,
        )
        if H is not None and inliers >= min_inliers:
            valid += 1
            inliers_sum += inliers
    total_ms = (time.perf_counter() - t0) * 1000.0
    n = len(frames)
    return {
        "mode": f"Full-map {method.upper()}" + (" + CLAHE" if use_clahe else ""),
        "avg_ms": total_ms / n,
        "valid": valid,
        "total": n,
        "accuracy_pct": 100.0 * valid / n,
        "avg_inliers": (inliers_sum / valid) if valid else 0.0,
    }


def bench_roi_tracker(
    map_img: np.ndarray,
    frames: list[np.ndarray],
    method: str = "sift",
    crop_scale: float = 0.25,
    use_clahe: bool = False,
) -> dict[str, float | int | str]:
    """Run DroneTracker (local ROI search + sanity checks) across all frames."""
    matcher_fn = partial(
        match_frame,
        method=method,
        crop_scale=crop_scale,
        use_clahe=use_clahe,
    )
    tracker = DroneTracker(map_img=map_img, matcher_fn=matcher_fn, config=TrackerConfig())
    t0 = time.perf_counter()
    results = tracker.track(frames)
    total_ms = (time.perf_counter() - t0) * 1000.0

    n = len(frames)
    valid_items = [r for r in results if r.valid]
    valid = len(valid_items)
    avg_inliers = float(np.mean([r.inliers for r in valid_items])) if valid_items else 0.0
    return {
        "mode": f"ROI {method.upper()}" + (" + CLAHE" if use_clahe else ""),
        "avg_ms": total_ms / n,
        "valid": valid,
        "total": n,
        "accuracy_pct": 100.0 * valid / n,
        "avg_inliers": avg_inliers,
    }


def print_table(title: str, rows: list[dict[str, float | int | str]]) -> None:
    """Print a slide-ready comparison table with speedup relative to the first row."""
    baseline_ms = float(rows[0]["avg_ms"])
    print(f"\n=== {title} ===")
    print(f"{'Mode':<24} | {'Avg ms/frame':>12} | {'Speedup':>8} | {'Success':>14} | {'Avg Inliers':>11}")
    print("-" * 80)
    for row in rows:
        avg_ms = float(row["avg_ms"])
        speedup = baseline_ms / avg_ms if avg_ms > 0 else 0.0
        success_str = f"{row['valid']}/{row['total']} ({row['accuracy_pct']:.1f}%)"
        print(
            f"{str(row['mode']):<24} | {avg_ms:>12.1f} | {speedup:>7.2f}x | "
            f"{success_str:>14} | {float(row['avg_inliers']):>11.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run engineering benchmarks for UAV navigation.")
    parser.add_argument("--map", default="data/global_map.png", help="Path to map image.")
    parser.add_argument("--crops", default="data/crops", help="Path to Basic crops folder.")
    parser.add_argument("--video", default="", help="Optional path to Advanced video file.")
    parser.add_argument("--stride", type=int, default=10, help="Frame stride for video.")
    parser.add_argument("--scale", type=float, default=0.25, help="crop_scale passed to match_frame.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max frames per dataset (0 = all).")
    args = parser.parse_args()

    map_img = cv2.imread(args.map)
    if map_img is None:
        raise SystemExit(f"Failed to load map: {args.map}")

    crops_dir = Path(args.crops)
    if crops_dir.exists():
        basic_frames = load_frames_from_dir(crops_dir)
        if args.limit > 0:
            basic_frames = basic_frames[: args.limit]
        basic_rows = [
            bench_full_map(map_img, basic_frames, method="sift", crop_scale=args.scale, use_clahe=False),
            bench_roi_tracker(map_img, basic_frames, method="sift", crop_scale=args.scale, use_clahe=False),
            bench_roi_tracker(map_img, basic_frames, method="sift", crop_scale=args.scale, use_clahe=True),
            bench_roi_tracker(map_img, basic_frames, method="orb", crop_scale=args.scale, use_clahe=False),
            bench_roi_tracker(map_img, basic_frames, method="orb", crop_scale=args.scale, use_clahe=True),
        ]
        print_table(f"Basic Dataset ({len(basic_frames)} frames, scale={args.scale})", basic_rows)

    if args.video:
        video_path = Path(args.video)
        video_frames = load_frames_from_video(video_path, stride=args.stride)
        if args.limit > 0:
            video_frames = video_frames[: args.limit]
        video_rows = [
            bench_full_map(map_img, video_frames, method="sift", crop_scale=args.scale, use_clahe=False),
            bench_roi_tracker(map_img, video_frames, method="sift", crop_scale=args.scale, use_clahe=False),
            bench_roi_tracker(map_img, video_frames, method="sift", crop_scale=args.scale, use_clahe=True),
            bench_roi_tracker(map_img, video_frames, method="orb", crop_scale=args.scale, use_clahe=False),
            bench_roi_tracker(map_img, video_frames, method="orb", crop_scale=args.scale, use_clahe=True),
        ]
        print_table(
            f"Video Dataset: {video_path.name} ({len(video_frames)} frames, stride={args.stride}, scale={args.scale})",
            video_rows,
        )


if __name__ == "__main__":
    main()
