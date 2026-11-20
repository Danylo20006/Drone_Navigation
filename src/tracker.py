from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import savgol_filter

from src.schemas import FrameResult, Point2D

# Matcher callable signature:
# (map_crop, drone_frame) -> (H_matrix_3x3 | None, num_inliers)
MatcherFn = Callable[[np.ndarray, np.ndarray], Tuple[Optional[np.ndarray], int]]


@dataclass
class TrackerConfig:
    """Configuration parameters for ROI tracking and homography sanity checks."""
    roi_radius: int = 450              # Base ROI search radius (map pixels)
    min_inliers: int = 12              # Minimum RANSAC/MAGSAC inliers required
    max_jump_px: float = 350.0         # Maximum allowed displacement per frame (map pixels)
    min_area_ratio: float = 0.005      # Min ratio of projected area to original frame area
    max_area_ratio: float = 20.0       # Max ratio of projected area to original frame area
    expand_factor: float = 1.8         # ROI expansion multiplier per consecutive lost frame
    max_lost_before_global: int = 4    # Consecutive misses before falling back to full-map search


class DroneTracker:
    """
    UAV trajectory tracker over a satellite map using local ROI search,
    homography sanity checks, and automatic relocalization.
    """

    def __init__(
        self,
        map_img: np.ndarray,
        matcher_fn: MatcherFn,
        config: Optional[TrackerConfig] = None,
    ) -> None:
        self.map_img = map_img
        self.map_h, self.map_w = map_img.shape[:2]
        self.matcher_fn = matcher_fn
        self.cfg = config or TrackerConfig()

    def _extract_roi(
        self,
        center: Point2D,
        radius: int,
    ) -> Tuple[np.ndarray, int, int]:
        """Extract a square ROI crop around `center`, clamped to map boundaries."""
        cx, cy = int(round(center[0])), int(round(center[1]))
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(self.map_w, cx + radius)
        y2 = min(self.map_h, cy + radius)

        crop = self.map_img[y1:y2, x1:x2]
        return crop, x1, y1

    @staticmethod
    def _project_frame(
        H: np.ndarray,
        frame_shape: Tuple[int, int],
        offset_x: int,
        offset_y: int,
    ) -> Tuple[Point2D, List[Point2D]]:
        """
        Project the frame center and 4 corners [TL, TR, BR, BL]
        into the global satellite map coordinate system.
        """
        h, w = frame_shape
        pts_local = np.array(
            [
                [[0.0, 0.0]],
                [[float(w), 0.0]],
                [[float(w), float(h)]],
                [[0.0, float(h)]],
                [[w / 2.0, h / 2.0]],
            ],
            dtype=np.float32,
        )

        projected = cv2.perspectiveTransform(pts_local, H).reshape(-1, 2)
        projected[:, 0] += offset_x
        projected[:, 1] += offset_y

        corners: List[Point2D] = [
            (float(pt[0]), float(pt[1])) for pt in projected[:4]
        ]
        center: Point2D = (float(projected[4, 0]), float(projected[4, 1]))
        return center, corners

    def _is_valid_homography(
        self,
        H: Optional[np.ndarray],
        inliers: int,
        center: Point2D,
        corners: List[Point2D],
        frame_shape: Tuple[int, int],
        last_center: Optional[Point2D],
        missed_streak: int,
    ) -> bool:
        """Validate physical plausibility of the estimated homography."""
        if H is None or H.shape != (3, 3):
            return False

        if inliers < self.cfg.min_inliers:
            return False

        # 1. Top-left 2x2 determinant (rejects reflections and degenerate projections)
        det = float(H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0])
        if det <= 1e-6:
            return False

        # 2. Projected center must lie within map bounds
        cx, cy = center
        if not (0.0 <= cx <= self.map_w and 0.0 <= cy <= self.map_h):
            return False

        # 3. Projected quadrilateral must be convex
        quad = np.array(corners, dtype=np.float32)
        if not cv2.isContourConvex(quad):
            return False

        # 4. Projected area must be within plausible scale bounds
        frame_area = float(frame_shape[0] * frame_shape[1])
        proj_area = float(cv2.contourArea(quad))
        area_ratio = proj_area / max(frame_area, 1.0)
        if not (self.cfg.min_area_ratio <= area_ratio <= self.cfg.max_area_ratio):
            return False

        # 5. Limit maximum displacement from the last valid position
        if last_center is not None:
            dist = float(np.hypot(cx - last_center[0], cy - last_center[1]))
            allowed_jump = self.cfg.max_jump_px * (missed_streak + 1)
            if dist > allowed_jump:
                return False

        return True

    def track(self, frames: List[np.ndarray]) -> List[FrameResult]:
        """Process a sequence of UAV frames and return the raw trajectory."""
        results: List[FrameResult] = []
        last_valid_center: Optional[Point2D] = None
        missed_streak = 0

        for idx, frame in enumerate(frames):
            frame_h, frame_w = frame.shape[:2]

            # Select search region: local ROI or full map
            use_roi = (
                last_valid_center is not None
                and missed_streak < self.cfg.max_lost_before_global
            )

            if use_roi:
                radius = int(self.cfg.roi_radius * (self.cfg.expand_factor ** missed_streak))
                search_img, offset_x, offset_y = self._extract_roi(last_valid_center, radius)
            else:
                search_img, offset_x, offset_y = self.map_img, 0, 0

            # Run feature matching and homography estimation
            H, inliers = self.matcher_fn(search_img, frame)

            is_valid = False
            center: Optional[Point2D] = None
            corners: Optional[List[Point2D]] = None

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
                    last_center=last_valid_center,
                    missed_streak=missed_streak,
                ):
                    is_valid = True
                    center = cand_center
                    corners = cand_corners

            if is_valid and center is not None:
                last_valid_center = center
                missed_streak = 0
                results.append(
                    FrameResult(
                        frame_id=idx,
                        center=center,
                        corners=corners,
                        inliers=inliers,
                        valid=True,
                    )
                )
            else:
                missed_streak += 1
                results.append(
                    FrameResult(
                        frame_id=idx,
                        center=None,
                        corners=None,
                        inliers=inliers,
                        valid=False,
                    )
                )

        return results


def postprocess_trajectory(
    results: List[FrameResult],
    interpolate: bool = True,
    smooth: bool = True,
    smooth_window: int = 7,
    polyorder: int = 2,
) -> List[FrameResult]:
    """
    Post-process raw trajectory:
    1. Linearly interpolate missing frames between valid detections.
    2. Smooth center coordinates using a Savitzky-Golay filter.
    """
    if not results:
        return results

    # Deep copy results to avoid mutating the input list
    processed = [
        FrameResult(
            frame_id=r.frame_id,
            center=r.center,
            corners=r.corners,
            inliers=r.inliers,
            valid=r.valid,
            interpolated=r.interpolated,
        )
        for r in results
    ]

    valid_indices = [i for i, r in enumerate(processed) if r.valid and r.center is not None]
    if len(valid_indices) < 2:
        return processed

    # 1. Linear interpolation for gaps within the valid frame range
    if interpolate:
        known_x = np.array([processed[i].center[0] for i in valid_indices], dtype=np.float64)
        known_y = np.array([processed[i].center[1] for i in valid_indices], dtype=np.float64)
        known_idx = np.array(valid_indices, dtype=np.float64)

        first_valid, last_valid = valid_indices[0], valid_indices[-1]
        for i in range(first_valid, last_valid + 1):
            if not processed[i].valid or processed[i].center is None:
                ix = float(np.interp(i, known_idx, known_x))
                iy = float(np.interp(i, known_idx, known_y))
                processed[i].center = (ix, iy)
                processed[i].corners = None
                processed[i].valid = True
                processed[i].interpolated = True

    # 2. Savitzky-Golay trajectory smoothing
    if smooth:
        active_indices = [i for i, r in enumerate(processed) if r.valid and r.center is not None]
        n_pts = len(active_indices)

        # Window length must be odd, <= number of points, and > polyorder
        win = min(smooth_window, n_pts if n_pts % 2 == 1 else n_pts - 1)
        if win > polyorder and win >= 3:
            xs = np.array([processed[i].center[0] for i in active_indices], dtype=np.float64)
            ys = np.array([processed[i].center[1] for i in active_indices], dtype=np.float64)

            xs_smooth = savgol_filter(xs, window_length=win, polyorder=polyorder)
            ys_smooth = savgol_filter(ys, window_length=win, polyorder=polyorder)

            for idx, sx, sy in zip(active_indices, xs_smooth, ys_smooth):
                processed[idx].center = (float(sx), float(sy))

    return processed