"""Local test and visual verification script for src/matcher.py."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.matcher import match_frame


def natural_sort_key(path: Path) -> list[int | str]:
    """Sort paths naturally so crop_2 comes before crop_10."""
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def project_frame_corners(H: np.ndarray, frame_shape: tuple[int, ...]) -> np.ndarray:
    """Project the 4 corners of frame_img onto map_img using H."""
    h, w = frame_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, H).reshape(-1, 2)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR/BGRA/Gray image to RGB for matplotlib display."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def render_verification_figure(
    map_img: np.ndarray,
    frame_img: np.ndarray,
    H: np.ndarray,
    corners: np.ndarray,
    inliers: int,
    method: str,
    save_path: Path | None,
    show: bool,
) -> None:
    """Create a 3-panel visual check: map position, zoomed map ROI, and warped frame overlay."""
    map_rgb = bgr_to_rgb(map_img)
    frame_rgb = bgr_to_rgb(frame_img)

    h_map, w_map = map_rgb.shape[:2]
    warped_rgb = cv2.warpPerspective(frame_rgb, H, (w_map, h_map))

    center = corners.mean(axis=0)
    pad = 80
    x1 = max(0, int(np.floor(corners[:, 0].min())) - pad)
    y1 = max(0, int(np.floor(corners[:, 1].min())) - pad)
    x2 = min(w_map, int(np.ceil(corners[:, 0].max())) + pad)
    y2 = min(h_map, int(np.ceil(corners[:, 1].max())) + pad)

    map_roi = map_rgb[y1:y2, x1:x2]
    warped_roi = warped_rgb[y1:y2, x1:x2]
    mask = (warped_roi.sum(axis=2) > 0)[..., None]
    blended_roi = np.where(mask, (0.5 * map_roi + 0.5 * warped_roi).astype(np.uint8), map_roi)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Method: {method.upper()} | Inliers: {inliers} | Center: ({center[0]:.1f}, {center[1]:.1f})")

    axes[0].imshow(map_rgb)
    poly = np.vstack([corners, corners[0]])
    axes[0].plot(poly[:, 0], poly[:, 1], color="lime", linewidth=2)
    axes[0].scatter([center[0]], [center[1]], color="red", s=40)
    axes[0].set_title("1. Position on Global Map")
    axes[0].axis("off")

    axes[1].imshow(map_roi)
    axes[1].set_title("2. Map ROI (Ground Truth)")
    axes[1].axis("off")

    axes[2].imshow(blended_roi)
    axes[2].set_title("3. Map + Warped Crop (50/50)")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def run_batch(
    map_img: np.ndarray,
    crops_dir: Path,
    count: int,
    method: str,
    scale: float,
    use_clahe: bool,
    save_path: Path | None,
    show: bool,
) -> None:
    """Run match_frame on the first N crops, print stats, and optionally plot all matches."""
    crop_paths = sorted(crops_dir.glob("*.png"), key=natural_sort_key)[:count]
    if not crop_paths:
        raise SystemExit(f"No PNG crops found in: {crops_dir}")

    print(f"Batch test ({len(crop_paths)} frames) | method={method} | scale={scale} | clahe={use_clahe}")
    print(f"{'Frame':<16} {'Inliers':>8} {'Time (ms)':>10} {'Center (x, y)':>22}")
    print("-" * 60)

    inlier_list: list[int] = []
    time_list: list[float] = []
    matched_items: list[tuple[str, np.ndarray, np.ndarray]] = []
    for path in crop_paths:
        frame_img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if frame_img is None:
            print(f"{path.name:<16} {'ERR':>8}")
            continue
        t0 = time.perf_counter()
        H, inliers = match_frame(
            map_img=map_img,
            frame_img=frame_img,
            method=method,
            crop_scale=scale,
            use_clahe=use_clahe,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        time_list.append(elapsed_ms)
        inlier_list.append(inliers)

        if H is None:
            center_str = "NO MATCH"
        else:
            corners = project_frame_corners(H, frame_img.shape)
            center = corners.mean(axis=0)
            center_str = f"({center[0]:.1f}, {center[1]:.1f})"
            matched_items.append((path.stem, corners, center))
        print(f"{path.name:<16} {inliers:>8} {elapsed_ms:>10.1f} {center_str:>22}")

    if inlier_list:
        print("-" * 60)
        print(
            f"Summary: min_inliers={min(inlier_list)}, avg_inliers={np.mean(inlier_list):.1f}, "
            f"avg_time={np.mean(time_list):.1f} ms"
        )

    if (save_path is not None or show) and matched_items:
        map_rgb = bgr_to_rgb(map_img)
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.imshow(map_rgb)
        centers = np.array([c for _, _, c in matched_items])
        ax.plot(centers[:, 0], centers[:, 1], color="cyan", linewidth=1.5, alpha=0.8)
        for label, corners, center in matched_items:
            poly = np.vstack([corners, corners[0]])
            ax.plot(poly[:, 0], poly[:, 1], color="lime", linewidth=1.0, alpha=0.7)
            ax.scatter([center[0]], [center[1]], color="red", s=18, zorder=5)
            idx_text = label.replace("crop_", "")
            ax.text(center[0] + 6, center[1] - 6, idx_text, color="yellow", fontsize=7, weight="bold")
        ax.set_title(
            f"Batch Match Overview ({len(matched_items)}/{len(crop_paths)} matched) | "
            f"method={method.upper()} | avg_inliers={np.mean(inlier_list):.1f}"
        )
        ax.axis("off")
        plt.tight_layout()
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved batch visualization to: {save_path}")
        if show:
            plt.show()
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test src.matcher.match_frame locally.")
    parser.add_argument("--map", default="data/global_map.png", help="Path to map image.")
    parser.add_argument("--crop", default="data/crops/crop_0.png", help="Path to a single crop image.")
    parser.add_argument("--crops-dir", default="data/crops", help="Crops directory for --batch mode.")
    parser.add_argument("--batch", type=int, default=0, help="If > 0, test first N crops without plotting.")
    parser.add_argument("--method", default="sift", choices=["sift", "orb"])
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--save-vis", default="", help="Optional path to save verification plot.")
    parser.add_argument("--show", action="store_true", help="Open interactive matplotlib window.")
    args = parser.parse_args()

    map_img = cv2.imread(args.map, cv2.IMREAD_UNCHANGED)
    if map_img is None:
        raise SystemExit(f"Failed to load map image: {args.map}")

    if args.batch > 0:
        save_path = Path(args.save_vis) if args.save_vis else None
        run_batch(
            map_img,
            Path(args.crops_dir),
            args.batch,
            args.method,
            args.scale,
            args.clahe,
            save_path,
            args.show,
        )
        return

    frame_img = cv2.imread(args.crop, cv2.IMREAD_UNCHANGED)
    if frame_img is None:
        raise SystemExit(f"Failed to load frame image: {args.crop}")

    t0 = time.perf_counter()
    H, inliers = match_frame(
        map_img=map_img,
        frame_img=frame_img,
        method=args.method,
        crop_scale=args.scale,
        use_clahe=args.clahe,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Map: {args.map} ({map_img.shape[1]}x{map_img.shape[0]})")
    print(f"Frame: {args.crop} ({frame_img.shape[1]}x{frame_img.shape[0]})")
    print(f"Method: {args.method} | crop_scale={args.scale} | use_clahe={args.clahe}")
    print(f"Time: {elapsed_ms:.1f} ms | Inliers: {inliers}")

    if H is None:
        print("Result: NO MATCH (H is None)")
        return

    corners = project_frame_corners(H, frame_img.shape)
    center = corners.mean(axis=0)
    print(f"H shape: {H.shape}, dtype: {H.dtype}")
    print(f"Projected center on map: ({center[0]:.1f}, {center[1]:.1f})")
    print("Projected corners (x, y):")
    for idx, (x, y) in enumerate(corners):
        print(f"  corner {idx}: ({x:.1f}, {y:.1f})")

    if args.save_vis or args.show:
        save_path = Path(args.save_vis) if args.save_vis else None
        render_verification_figure(
            map_img=map_img,
            frame_img=frame_img,
            H=H,
            corners=corners,
            inliers=inliers,
            method=args.method,
            save_path=save_path,
            show=args.show,
        )


if __name__ == "__main__":
    main()
