from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from src.schemas import FrameResult, Point2D
from scipy.signal import savgol_filter

# Тип функції матчингу, яку реалізує Учасник 1:
# (map_crop, drone_frame) -> (H_matrix_3x3 | None, num_inliers)
MatcherFn = Callable[[np.ndarray, np.ndarray], Tuple[Optional[np.ndarray], int]]


@dataclass
class TrackerConfig:
    """Параметри трекінгу та перевірок валідності гомографії."""
    roi_radius: int = 450              # Базовий радіус вікна пошуку (пікселі карти)
    min_inliers: int = 12              # Мінімальна кількість inliers від RANSAC
    max_jump_px: float = 350.0         # Максимальний зсув дрона за 1 кадр (пікселі карти)
    min_area_ratio: float = 0.005      # Мін. відношення площі проєкції до площі кадру
    max_area_ratio: float = 20.0       # Макс. відношення площі проєкції до площі кадру
    expand_factor: float = 1.8         # У скільки разів розширювати ROI при втраті кадру
    max_lost_before_global: int = 4    # Після скількох пропусків шукати по всій карті


class DroneTracker:
    """
    Трекінг траєкторії БПЛА по супутниковій карті з використанням
    локального ROI-пошуку, sanity checks гомографії та ре-локалізації.
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
        """Вирізає квадратне вікно ROI навколо center з обмеженням по краях карти."""
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
        Переносить центр і 4 кути [TL, TR, BR, BL] кадру дрона
        в глобальну систему координат карти.
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
        """Перевіряє фізичну адекватність знайденої гомографії."""
        if H is None or H.shape != (3, 3):
            return False

        if inliers < self.cfg.min_inliers:
            return False

        # 1. Визначник 2x2 (відсікає дзеркальні та вироджені проєкції)
        det = float(H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0])
        if det <= 1e-6:
            return False

        # 2. Центр має лежати в межах карти
        cx, cy = center
        if not (0.0 <= cx <= self.map_w and 0.0 <= cy <= self.map_h):
            return False

        # 3. Опуклість чотирикутника
        quad = np.array(corners, dtype=np.float32)
        if not cv2.isContourConvex(quad):
            return False

        # 4. Адекватність площі проєкції
        frame_area = float(frame_shape[0] * frame_shape[1])
        proj_area = float(cv2.contourArea(quad))
        area_ratio = proj_area / max(frame_area, 1.0)
        if not (self.cfg.min_area_ratio <= area_ratio <= self.cfg.max_area_ratio):
            return False

        # 5. Обмеження максимального стрибка від попередньої валідної точки
        if last_center is not None:
            dist = float(np.hypot(cx - last_center[0], cy - last_center[1]))
            allowed_jump = self.cfg.max_jump_px * (missed_streak + 1)
            if dist > allowed_jump:
                return False

        return True

    def track(self, frames: List[np.ndarray]) -> List[FrameResult]:
        """Проганяє послідовність кадрів дрона і повертає сиру траєкторію."""
        results: List[FrameResult] = []
        last_valid_center: Optional[Point2D] = None
        missed_streak = 0

        for idx, frame in enumerate(frames):
            frame_h, frame_w = frame.shape[:2]

            # Визначаємо область пошуку: ROI або вся карта
            use_roi = (
                last_valid_center is not None
                and missed_streak < self.cfg.max_lost_before_global
            )

            if use_roi:
                radius = int(self.cfg.roi_radius * (self.cfg.expand_factor ** missed_streak))
                search_img, offset_x, offset_y = self._extract_roi(last_valid_center, radius)
            else:
                search_img, offset_x, offset_y = self.map_img, 0, 0

            # Виклик зовнішнього матчера (Учасник 1)
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
        Постобробка сирої траєкторії:
        1. Лінійна інтерполяція пропущених кадрів між валідними точками.
        2. Згладжування координат центру фільтром Савицького-Голея.
        """
        if not results:
            return results

        # Копіюємо список, щоб не мутувати оригінал
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

        # 1. Лінійна інтерполяція пропусків (тільки всередині діапазону валідних кадрів)
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

        # 2. Згладжування фільтром Савицького-Голея
        if smooth:
            active_indices = [i for i, r in enumerate(processed) if r.valid and r.center is not None]
            n_pts = len(active_indices)

            # Вікно має бути непарним, <= кількості точок і > polyorder
            win = min(smooth_window, n_pts if n_pts % 2 == 1 else n_pts - 1)
            if win > polyorder and win >= 3:
                xs = np.array([processed[i].center[0] for i in active_indices], dtype=np.float64)
                ys = np.array([processed[i].center[1] for i in active_indices], dtype=np.float64)

                xs_smooth = savgol_filter(xs, window_length=win, polyorder=polyorder)
                ys_smooth = savgol_filter(ys, window_length=win, polyorder=polyorder)

                for idx, sx, sy in zip(active_indices, xs_smooth, ys_smooth):
                    processed[idx].center = (float(sx), float(sy))

        return processed