from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import cv2

from src.io_utils import load_frames_from_dir, load_frames_from_video
from src.matcher import match_frame
from src.schemas import save_trajectory
from src.tracker import DroneTracker, TrackerConfig, postprocess_trajectory
from src.visualizer import draw_trajectory_on_map, render_zoom_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UAV Visual Navigation Pipeline (Map Matching + Trajectory Visualization)"
    )
    parser.add_argument(
        "--map",
        type=str,
        required=True,
        help="data/global_map.png",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="data/frames",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="output",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="sift",
        choices=["sift", "orb"],
        help="match_frame",
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=0.25,
        help="Коефіцієнт масштабування кадру дрона перед матчингом.",
    )
    parser.add_argument(
        "--use-clahe",
        action="store_true",
        help="Увімкнути CLAHE-препроцесинг (корисно для відео з екрана).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Крок вибірки кадрів, якщо на вхід подано відео.",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Вимкнути згладжування траєкторії.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Не рендерити відео із зумом (тільки JSON і картинка).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Завантаження карти
    map_img = cv2.imread(args.map)
    if map_img is None:
        raise FileNotFoundError(f"Не вдалося завантажити карту: {args.map}")
    print(f"[1/4] Карту завантажено: {map_img.shape[1]}x{map_img.shape[0]}")

    # 2. Завантаження кадрів (папка або відео)
    input_path = Path(args.input)
    if input_path.is_dir():
        frames = load_frames_from_dir(input_path)
    elif input_path.is_file():
        frames = load_frames_from_video(input_path, stride=args.stride)
    else:
        raise FileNotFoundError(f"Вхідний шлях не знайдено: {args.input}")
    print(f"[2/4] Завантажено кадрів: {len(frames)}")

    if not frames:
        raise ValueError("Список кадрів порожній.")

    # 3. Трекінг + постобробка
    matcher_fn = partial(
        match_frame,
        method=args.method,
        crop_scale=args.crop_scale,
        use_clahe=args.use_clahe,
    )
    tracker = DroneTracker(map_img=map_img, matcher_fn=matcher_fn, config=TrackerConfig())
    raw_trajectory = tracker.track(frames)

    valid_raw = sum(1 for r in raw_trajectory if r.valid)
    print(f"[3/4] Трекінг завершено: {valid_raw}/{len(frames)} валідних кадрів")

    final_trajectory = postprocess_trajectory(
        raw_trajectory,
        interpolate=True,
        smooth=not args.no_smooth,
    )

    traj_json_path = out_dir / "trajectory.json"
    save_trajectory(final_trajectory, traj_json_path)
    print(f"      Збережено траєкторію: {traj_json_path}")

    # 4. Візуалізація
    route_img_path = out_dir / "route_visualization.png"
    draw_trajectory_on_map(
        map_path=args.map,
        trajectory_path=traj_json_path,
        output_path=route_img_path,
    )

    if not args.no_video:
        video_path = out_dir / "trajectory_zoom.mp4"
        render_zoom_video(
            map_path=args.map,
            trajectory_path=traj_json_path,
            output_video=video_path,
            fps=15,
            steps_per_segment=10,
        )

    print("[4/4] Готово!")


if __name__ == "__main__":
    main()