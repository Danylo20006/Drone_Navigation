import json
from pathlib import Path
import cv2
import numpy as np


def load_trajectory(json_path):
    """Завантажує список точок траєкторії з JSON-файлу."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл траєкторії не знайдено: {json_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Файл траєкторії порожній або має неправильний формат.")

    return data


def draw_trajectory_on_map(
    map_path,
    trajectory_path,
    output_path="route_visualization.png",
    box_alpha=0.25,
):
    """
    Малює маршрут поверх великої карти:
    - лінія маршруту та пронумеровані точки;
    - мітки START (зелений) і FINISH (червоний);
    - напівпрозорі рамки навколо позицій дрона.
    """
    map_img = cv2.imread(str(map_path))
    if map_img is None:
        raise FileNotFoundError(f"Не вдалося відкрити карту: {map_path}")

    trajectory = load_trajectory(trajectory_path)

    valid_points = [
        item for item in trajectory
        if item.get("valid", True) and item.get("center") is not None
    ]
    if len(valid_points) == 0:
        raise ValueError("У траєкторії немає жодної валідної точки.")

    # 1. Напівпрозорі рамки через overlay + cv2.addWeighted
    overlay = map_img.copy()
    for item in valid_points:
        corners = item.get("corners")
        if corners and len(corners) == 4:
            pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color=(255, 200, 0))
            cv2.polylines(map_img, [pts], isClosed=True, color=(0, 215, 255), thickness=2)

    cv2.addWeighted(overlay, box_alpha, map_img, 1.0 - box_alpha, 0, dst=map_img)

    # 2. Лінія маршруту між сусідніми точками
    centers = [
        (int(round(item["center"][0])), int(round(item["center"][1])))
        for item in valid_points
    ]
    for i in range(1, len(centers)):
        cv2.line(map_img, centers[i - 1], centers[i], color=(0, 255, 120), thickness=4)

    # 3. Точки та номери кадрів
    for idx, (item, (cx, cy)) in enumerate(zip(valid_points, centers)):
        frame_id = item.get("frame_id", idx)

        if idx == 0:
            point_color = (0, 200, 0)      # Зелений для START
        elif idx == len(centers) - 1:
            point_color = (0, 0, 230)      # Червоний для FINISH
        else:
            point_color = (255, 120, 0)    # Синій/блакитний для проміжних точок

        cv2.circle(map_img, (cx, cy), radius=8, color=point_color, thickness=-1)
        cv2.circle(map_img, (cx, cy), radius=8, color=(255, 255, 255), thickness=2)

        cv2.putText(
            map_img,
            str(frame_id),
            (cx + 12, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    # 4. Чіткі мітки START і FINISH
    start_x, start_y = centers[0]
    finish_x, finish_y = centers[-1]

    cv2.putText(
        map_img,
        "START",
        (start_x - 20, max(30, start_y - 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 200, 0),
        thickness=3,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        map_img,
        "FINISH",
        (finish_x - 25, max(30, finish_y - 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 230),
        thickness=3,
        lineType=cv2.LINE_AA,
    )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), map_img)
    print(f"Збережено візуалізацію у {out_file}")
    return map_img


def _extract_zoom_window(img, center_xy, crop_w=1280, crop_h=720):
    """Вирізає вікно навколо center_xy з коректною обробкою країв зображення."""
    img_h, img_w = img.shape[:2]
    crop_w = min(crop_w, img_w)
    crop_h = min(crop_h, img_h)

    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))

    x1 = cx - crop_w // 2
    y1 = cy - crop_h // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > img_w:
        x1 -= (x2 - img_w)
        x2 = img_w
    if y2 > img_h:
        y1 -= (y2 - img_h)
        y2 = img_h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    crop = img[y1:y2, x1:x2]

    if crop.shape[1] != crop_w or crop.shape[0] != crop_h:
        canvas = np.zeros((crop_h, crop_w, 3), dtype=img.dtype)
        canvas[0:crop.shape[0], 0:crop.shape[1]] = crop
        return canvas

    return crop


def render_zoom_video(
    map_path,
    trajectory_path,
    output_video="trajectory_zoom.mp4",
    fps=30,
    steps_per_segment=30,
    crop_w=1280,
    crop_h=720,
    box_alpha=0.25,
):
    """Генерує плавне .mp4 відео з ефектом зуму та поступовим малюванням маршруту."""
    base_map = cv2.imread(str(map_path))
    if base_map is None:
        raise FileNotFoundError(f"Не вдалося відкрити карту: {map_path}")

    trajectory = load_trajectory(trajectory_path)
    valid_points = [
        item for item in trajectory
        if item.get("valid", True) and item.get("center") is not None
    ]
    if len(valid_points) < 2:
        raise ValueError("Для відео потрібно щонайменше 2 валідні точки.")

    centers = [
        (float(item["center"][0]), float(item["center"][1]))
        for item in valid_points
    ]

    out_path = Path(output_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (crop_w, crop_h))
    if not writer.isOpened():
        raise RuntimeError(f"Не вдалося відкрити VideoWriter для файлу: {out_path}")

    start_x, start_y = int(round(centers[0][0])), int(round(centers[0][1]))
    finish_x, finish_y = int(round(centers[-1][0])), int(round(centers[-1][1]))

    for seg_idx in range(len(centers) - 1):
        p1 = centers[seg_idx]
        p2 = centers[seg_idx + 1]

        for s in range(steps_per_segment):
            raw_t = s / steps_per_segment
            t = raw_t * raw_t * (3.0 - 2.0 * raw_t)

            cam_x = p1[0] + (p2[0] - p1[0]) * t
            cam_y = p1[1] + (p2[1] - p1[1]) * t
            head_x = cam_x
            head_y = cam_y

            frame_canvas = base_map.copy()

            overlay = frame_canvas.copy()
            for k in range(seg_idx + 1):
                corners = valid_points[k].get("corners")
                if corners and len(corners) == 4:
                    pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [pts], color=(255, 200, 0))
                    cv2.polylines(
                        frame_canvas,
                        [pts],
                        isClosed=True,
                        color=(0, 215, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )
            cv2.addWeighted(overlay, box_alpha, frame_canvas, 1.0 - box_alpha, 0, dst=frame_canvas)

            for k in range(seg_idx):
                pt_a = (int(round(centers[k][0])), int(round(centers[k][1])))
                pt_b = (int(round(centers[k + 1][0])), int(round(centers[k + 1][1])))
                cv2.line(frame_canvas, pt_a, pt_b, color=(0, 255, 120), thickness=4, lineType=cv2.LINE_AA)

            seg_start = (int(round(p1[0])), int(round(p1[1])))
            seg_curr = (int(round(head_x)), int(round(head_y)))
            cv2.line(frame_canvas, seg_start, seg_curr, color=(0, 255, 120), thickness=4, lineType=cv2.LINE_AA)

            for k in range(seg_idx + 1):
                cx_i = int(round(centers[k][0]))
                cy_i = int(round(centers[k][1]))
                frame_id = valid_points[k].get("frame_id", k)

                pt_color = (0, 200, 0) if k == 0 else (255, 120, 0)
                cv2.circle(frame_canvas, (cx_i, cy_i), radius=8, color=pt_color, thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(frame_canvas, (cx_i, cy_i), radius=8, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(
                    frame_canvas,
                    str(frame_id),
                    (cx_i + 12, cy_i - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

            cv2.circle(frame_canvas, seg_curr, radius=10, color=(0, 165, 255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(frame_canvas, seg_curr, radius=10, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

            cv2.putText(
                frame_canvas,
                "START",
                (start_x - 20, max(30, start_y - 28)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 220, 0),
                thickness=3,
                lineType=cv2.LINE_AA,
            )

            zoom_frame = _extract_zoom_window(
                frame_canvas,
                (cam_x, cam_y),
                crop_w=crop_w,
                crop_h=crop_h,
            )
            writer.write(zoom_frame)

    final_canvas = base_map.copy()
    for k in range(len(centers) - 1):
        pt_a = (int(round(centers[k][0])), int(round(centers[k][1])))
        pt_b = (int(round(centers[k + 1][0])), int(round(centers[k + 1][1])))
        cv2.line(final_canvas, pt_a, pt_b, color=(0, 255, 120), thickness=4, lineType=cv2.LINE_AA)
    cv2.putText(
        final_canvas,
        "START",
        (start_x - 20, max(30, start_y - 28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 220, 0),
        thickness=3,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        final_canvas,
        "FINISH",
        (finish_x - 25, max(30, finish_y - 28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 240),
        thickness=3,
        lineType=cv2.LINE_AA,
    )
    final_zoom = _extract_zoom_window(final_canvas, centers[-1], crop_w=crop_w, crop_h=crop_h)
    for _ in range(fps):
        writer.write(final_zoom)

    writer.release()
    print(f"Збережено плавне відео із зумом у {out_path}")
    return str(out_path)

if __name__ == "__main__":
    draw_trajectory_on_map(
        map_path="global_map.png",
        trajectory_path="mock_trajectory.json",
        output_path="route_visualization.png",
    )
    render_zoom_video(
        map_path="global_map.png",
        trajectory_path="mock_trajectory.json",
        output_video="trajectory_zoom.mp4",
        fps=4,
    )