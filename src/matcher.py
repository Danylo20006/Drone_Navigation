"""Isolated feature-matching module for UAV visual navigation (frame -> map)."""

from __future__ import annotations

import cv2
import numpy as np

MIN_MATCH_COUNT = 4
LOWE_RATIO = 0.75
RANSAC_REPROJ_THRESHOLD = 5.0
HOMOGRAPHY_METHOD = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert an image (2D, 3-channel, or 4-channel) to single-channel uint8."""
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def _preprocess(img: np.ndarray, use_clahe: bool) -> np.ndarray:
    """Convert image to grayscale and optionally apply CLAHE contrast normalization."""
    gray = _to_grayscale(img)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    return gray


def _create_detector_and_matcher(
    method: str,
) -> tuple[cv2.Feature2D, cv2.DescriptorMatcher, bool]:
    """Return (detector, matcher, needs_float32) for the requested feature method."""
    key = method.strip().lower()
    if key == "sift":
        detector = cv2.SIFT_create()
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        return detector, matcher, True
    if key == "orb":
        detector = cv2.ORB_create(nfeatures=5000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        return detector, matcher, False
    raise ValueError(f"Unsupported matching method: {method!r}. Expected 'sift' or 'orb'.")


def match_frame(
    map_img: np.ndarray,
    frame_img: np.ndarray,
    method: str = "sift",
    crop_scale: float = 0.25,
    use_clahe: bool = False,
) -> tuple[np.ndarray | None, int]:
    """Find the homography mapping original frame_img coordinates to map_img coordinates.

    Args:
        map_img: Satellite map (full map or ROI crop).
        frame_img: Drone camera frame (higher resolution than map_img).
        method: Feature extractor/matcher to use ('sift' or 'orb').
        crop_scale: Downscale factor applied to frame_img before feature extraction.
        use_clahe: Whether to apply CLAHE to both images for contrast robustness.

    Returns:
        (H_orig, inliers): 3x3 float64 homography matrix from original frame_img
        to map_img and the number of RANSAC/MAGSAC inliers, or (None, 0) on failure.
    """
    if crop_scale <= 0:
        raise ValueError(f"crop_scale must be positive, got {crop_scale}")
    if map_img is None or frame_img is None or map_img.size == 0 or frame_img.size == 0:
        return None, 0

    detector, matcher, needs_float32 = _create_detector_and_matcher(method)

    map_gray = _preprocess(map_img, use_clahe=use_clahe)
    frame_gray = _preprocess(frame_img, use_clahe=use_clahe)

    # Downscale drone frame to approximate the satellite map resolution
    if crop_scale != 1.0:
        interpolation = cv2.INTER_AREA if crop_scale < 1.0 else cv2.INTER_LINEAR
        frame_scaled = cv2.resize(
            frame_gray,
            (0, 0),
            fx=crop_scale,
            fy=crop_scale,
            interpolation=interpolation,
        )
    else:
        frame_scaled = frame_gray

    if frame_scaled.shape[0] < 2 or frame_scaled.shape[1] < 2:
        return None, 0

    kp_frame, des_frame = detector.detectAndCompute(frame_scaled, None)
    kp_map, des_map = detector.detectAndCompute(map_gray, None)

    if (
        des_frame is None
        or des_map is None
        or len(kp_frame) < MIN_MATCH_COUNT
        or len(kp_map) < MIN_MATCH_COUNT
        or len(des_frame) < 2
        or len(des_map) < 2
    ):
        return None, 0

    if needs_float32:
        des_frame = np.asarray(des_frame, dtype=np.float32)
        des_map = np.asarray(des_map, dtype=np.float32)

    # Filter matches using Lowe's ratio test
    knn_matches = matcher.knnMatch(des_frame, des_map, k=2)
    good_matches = [
        m
        for pair in knn_matches
        if len(pair) == 2
        for m, n in (pair,)
        if m.distance < LOWE_RATIO * n.distance
    ]

    if len(good_matches) < MIN_MATCH_COUNT:
        return None, 0

    src_pts = np.float32([kp_frame[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_map[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate robust homography via MAGSAC/RANSAC
    H_scaled, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        HOMOGRAPHY_METHOD,
        ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
    )
    if H_scaled is None or mask is None:
        return None, 0

    inliers = int(mask.ravel().sum())
    if inliers < MIN_MATCH_COUNT:
        return None, 0

    # Map homography back to the original unscaled frame coordinate space
    scale_matrix = np.diag([crop_scale, crop_scale, 1.0]).astype(np.float64)
    H_orig = H_scaled.astype(np.float64) @ scale_matrix
    return H_orig, inliers