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

    # Беремо лише валідні точки з координатами центру
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
            # Заливка рамки на overlay (жовто-блакитний тон)
            cv2.fillPoly(overlay, [pts], color=(255, 200, 0))
            # Чіткий контур рамки на основному зображенні
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

        # Номер кадру поруч із точкою
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


if __name__ == "__main__":
    draw_trajectory_on_map(
        map_path="global_map.png",
        trajectory_path="mock_trajectory.json",
        output_path="route_visualization.png",
    )