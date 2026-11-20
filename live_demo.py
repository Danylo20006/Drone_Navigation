"""Real-time UAV Visual Navigation Live Demo (Camera / Video Split-Screen + Tactical HUD)."""

from __future__ import annotations

import argparse

from collections import deque
from functools import partial
import time
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from src.matcher import match_frame
from src.schemas import Point2D
from src.tracker import DroneTracker, TrackerConfig


class LiveDroneTracker(DroneTracker):
    """Stateful single-frame wrapper around DroneTracker for real-time streams."""

    def __init__(
        self,
        map_img: np.ndarray,
        matcher_fn,
        config: Optional[TrackerConfig] = None,
    ) -> None:
        super().__init__(map_img=map_img, matcher_fn=matcher_fn, config=config)
        self.last_valid_center: Optional[Point2D] = None
        self.missed_streak: int = 0
        self.frame_id: int = 0

    def reset(self) -> None:
        """Reset tracking state to force immediate global relocalization."""
        self.last_valid_center = None
        self.missed_streak = 0

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[bool, Optional[Point2D], Optional[List[Point2D]], int, Optional[float], str]:
        """Process a single frame and return localization telemetry."""
        frame_h, frame_w = frame.shape[:2]

        use_roi = (
            self.last_valid_center is not None
            and self.missed_streak < self.cfg.max_lost_before_global
        )
        search_mode = "ROI" if use_roi else "GLOBAL"

        if use_roi and self.last_valid_center is not None:
            radius = int(self.cfg.roi_radius * (self.cfg.expand_factor ** self.missed_streak))
            search_img, offset_x, offset_y = self._extract_roi(self.last_valid_center, radius)
        else:
            search_img, offset_x, offset_y = self.map_img, 0, 0

        H, inliers = self.matcher_fn(search_img, frame)

        if H is not None:
            cand_center, cand_corners = self._project_frame(
                H, (frame_h, frame_w), offset_x, offset_y
            )
            if self._is_valid_homography(
                H=H,
                inliers=inliers,
                center=cand_center,
                corners=cand_corners,
                frame_shape=(frame_h, frame_w),
                last_center=self.last_valid_center,
                missed_streak=self.missed_streak,
            ):
                self.last_valid_center = cand_center
                self.missed_streak = 0
                self.frame_id += 1
                heading_deg = (float(np.degrees(np.arctan2(H[1, 0], H[0, 0]))) + 360.0) % 360.0
                return True, cand_center, cand_corners, inliers, heading_deg, search_mode

        self.missed_streak += 1
        self.frame_id += 1
        return False, None, None, inliers, None, search_mode


def _extract_zoom_crop(
    img: np.ndarray,
    center_xy: Optional[Point2D],
    crop_w: int = 960,
    crop_h: int = 720,
) -> np.ndarray:
    """Extract a zoomed map view around center_xy, or return a resized full map if not localized."""
    img_h, img_w = img.shape[:2]
    if center_xy is None:
        return cv2.resize(img, (crop_w, crop_h), interpolation=cv2.INTER_AREA)

    cw = min(crop_w, img_w)
    ch = min(crop_h, img_h)
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))

    x1 = max(0, min(cx - cw // 2, img_w - cw))
    y1 = max(0, min(cy - ch // 2, img_h - ch))
    crop = img[y1 : y1 + ch, x1 : x1 + cw]

    if crop.shape[1] != crop_w or crop.shape[0] != crop_h:
        return cv2.resize(crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    return crop


def _draw_hud(
    panel: np.ndarray,
    status: str,
    status_color: Tuple[int, int, int],
    fps: float,
    inliers: int,
    heading: Optional[float],
    search_mode: str,
) -> np.ndarray:
    """Overlay a semi-transparent tactical HUD bar at the top of the camera panel."""
    out = panel.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], 78), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, out, 0.3, 0, dst=out)

    heading_str = f"{heading:05.1f} deg" if heading is not None else "N/A"
    line1 = f"STATUS: {status}   |   MODE: {search_mode}   |   FPS: {fps:.1f}"
    line2 = f"INLIERS: {inliers}   |   HEADING: {heading_str}   |   [R]eset [C]lear [Space]Pause [Q]uit"

    cv2.putText(out, line1, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)
    cv2.putText(out, line2, (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live UAV Visual Navigation Demo")
    parser.add_argument("--map", type=str, default="data/global_map.png", help="Path to satellite map.")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index (e.g., '0', '1') or path to video file for fallback demo.",
    )
    parser.add_argument("--method", type=str, default="sift", choices=["sift", "orb"])
    parser.add_argument("--crop-scale", type=float, default=0.5, help="Downscale factor for input frames.")
    parser.add_argument("--use-clahe", action="store_true", help="Enable CLAHE contrast normalization.")
    parser.add_argument("--stride", type=int, default=3, help="Frame skip stride when source is a video file.")
    parser.add_argument("--panel-w", type=int, default=720, help="Width of each split-screen panel.")
    parser.add_argument("--panel-h", type=int, default=540, help="Height of each split-screen panel.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    map_img = cv2.imread(args.map)
    if map_img is None:
        raise FileNotFoundError(f"Failed to load map image: {args.map}")

    is_camera = args.source.isdigit()
    source: int | str = int(args.source) if is_camera else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {args.source}")

    matcher_fn = partial(
        match_frame,
        method=args.method,
        crop_scale=args.crop_scale,
        use_clahe=args.use_clahe,
    )
    # tracker = LiveDroneTracker(map_img=map_img, matcher_fn=matcher_fn, config=TrackerConfig())
    
    tracker = LiveDroneTracker(
    map_img=map_img,
    matcher_fn=matcher_fn,
    config=TrackerConfig(roi_radius=300),
)
    
    trail: Deque[Tuple[int, int]] = deque(maxlen=500)
    last_corners: Optional[List[Point2D]] = None
    last_heading: Optional[float] = None
    paused = False
    raw_frame_idx = 0

    window_name = "UAV Visual Navigation - Live Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print(f"Live demo started | source={args.source} | method={args.method} | clahe={args.use_clahe}")

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    if not is_camera:
                        # Loop video seamlessly for stage demos
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        tracker.reset()
                        trail.clear()
                        last_corners = None
                        continue
                    break

                raw_frame_idx += 1
                if not is_camera and (raw_frame_idx % max(1, args.stride) != 0):
                    continue

                t0 = time.perf_counter()
                valid, center, corners, inliers, heading, search_mode = tracker.process_frame(frame)
                dt = max(time.perf_counter() - t0, 1e-6)
                fps = 1.0 / dt

                if valid and center is not None:
                    trail.append((int(round(center[0])), int(round(center[1]))))
                    last_corners = corners
                    last_heading = heading
                    status = "LOCKED"
                    status_color = (0, 230, 0)
                else:
                    status = "RELOCALIZING" if search_mode == "GLOBAL" else "LOST"
                    status_color = (0, 80, 255)

                # Build map canvas with trail and current footprint
                map_canvas = map_img.copy()
                if len(trail) >= 2:
                    pts_trail = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(map_canvas, [pts_trail], isClosed=False, color=(0, 255, 120), thickness=4, lineType=cv2.LINE_AA)

                if last_corners is not None:
                    pts_box = np.array(last_corners, dtype=np.int32).reshape((-1, 1, 2))
                    box_color = (0, 215, 255) if valid else (0, 80, 255)
                    cv2.polylines(map_canvas, [pts_box], isClosed=True, color=box_color, thickness=3, lineType=cv2.LINE_AA)

                if tracker.last_valid_center is not None:
                    cx, cy = int(round(tracker.last_valid_center[0])), int(round(tracker.last_valid_center[1]))
                    cv2.circle(map_canvas, (cx, cy), 10, (0, 165, 255), -1, lineType=cv2.LINE_AA)
                    cv2.circle(map_canvas, (cx, cy), 10, (255, 255, 255), 2, lineType=cv2.LINE_AA)

                # Compose split-screen frame
                left_panel = cv2.resize(frame, (args.panel_w, args.panel_h), interpolation=cv2.INTER_AREA)
                left_panel = _draw_hud(
                    left_panel,
                    status=status,
                    status_color=status_color,
                    fps=fps,
                    inliers=inliers,
                    heading=last_heading,
                    search_mode=search_mode,
                )

                right_panel = _extract_zoom_crop(
                    map_canvas,
                    tracker.last_valid_center,
                    crop_w=args.panel_w,
                    crop_h=args.panel_h,
                )
                cv2.putText(
                    right_panel,
                    "SATELLITE MAP (LIVE TRACK)",
                    (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 120),
                    2,
                    cv2.LINE_AA,
                )

                combined = np.hstack([left_panel, right_panel])
                cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                break
            elif key == ord("r"):
                tracker.reset()
                trail.clear()
                last_corners = None
                print("Tracker reset -> forcing GLOBAL relocalization.")
            elif key == ord("c"):
                trail.clear()
                print("Trajectory trail cleared.")
            elif key == ord(" "):
                paused = not paused
                print("Paused." if paused else "Resumed.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()