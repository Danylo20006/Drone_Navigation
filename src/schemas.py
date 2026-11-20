from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple


Point2D = Tuple[float, float]


@dataclass
class FrameResult:
    """Результат локалізації одного кадру дрона на глобальній карті."""
    frame_id: int
    center: Optional[Point2D]                # (x, y) у пікселях глобальної карти
    corners: Optional[List[Point2D]]         # 4 кути кадру [TL, TR, BR, BL] на карті
    inliers: int                             # Кількість inliers від RANSAC
    valid: bool                              # True, якщо гомографія пройшла sanity checks
    interpolated: bool = False               # True, якщо координати відновлено інтерполяцією

    def to_dict(self) -> dict:
        return asdict(self)


def save_trajectory(results: List[FrameResult], output_path: str | Path) -> None:
    """Зберігає список результатів трекінгу у JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)


def load_trajectory(input_path: str | Path) -> List[FrameResult]:
    """Зчитує траєкторію з JSON (зручно для visualizer.py)."""
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return [
        FrameResult(
            frame_id=item["frame_id"],
            center=tuple(item["center"]) if item["center"] is not None else None,
            corners=[tuple(pt) for pt in item["corners"]] if item["corners"] is not None else None,
            inliers=item["inliers"],
            valid=item["valid"],
            interpolated=item.get("interpolated", False),
        )
        for item in raw
    ]