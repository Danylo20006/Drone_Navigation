import json
import math
from pathlib import Path


def build_mock_trajectory(num_points=14, out_file="mock_trajectory.json"):
    """Generate a synthetic sinusoidal trajectory JSON for visualization testing."""
    trajectory = []
    box_half_w = 130
    box_half_h = 85

    start_x = 350
    end_x = 3300
    step_x = (end_x - start_x) / max(1, num_points - 1)

    for i in range(num_points):
        cx = int(round(start_x + i * step_x))
        cy = int(round(920 + 420 * math.sin(i * 0.65)))

        # Axis-aligned bounding box corners [TL, TR, BR, BL]
        corners = [
            [cx - box_half_w, cy - box_half_h],
            [cx + box_half_w, cy - box_half_h],
            [cx + box_half_w, cy + box_half_h],
            [cx - box_half_w, cy + box_half_h],
        ]

        trajectory.append({
            "frame_id": i,
            "center": [cx, cy],
            "corners": corners,
            "valid": True,
        })

    out_path = Path(out_file).resolve()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, indent=2)

    print("Saved to:", out_path)
    print("First point:", trajectory[0]["center"], "Last point:", trajectory[-1]["center"])


if __name__ == "__main__":
    build_mock_trajectory()