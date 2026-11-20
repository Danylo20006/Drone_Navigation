import numpy as np
from src.schemas import save_trajectory
from src.tracker import DroneTracker, TrackerConfig


def dummy_matcher(map_crop: np.ndarray, drone_frame: np.ndarray):
    """Імітує зміщення дрона вправо-вниз у межах поточного ROI."""
    h_map, w_map = map_crop.shape[:2]
    h_fr, w_fr = drone_frame.shape[:2]
    tx = (w_map - w_fr) / 2.0 + 30.0
    ty = (h_map - h_fr) / 2.0 + 15.0
    H = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
    return H, 40


if __name__ == "__main__":
    fake_map = np.zeros((2000, 2000, 3), dtype=np.uint8)
    fake_frames = [np.zeros((200, 200, 3), dtype=np.uint8) for _ in range(10)]

    tracker = DroneTracker(fake_map, matcher_fn=dummy_matcher, config=TrackerConfig())
    trajectory = tracker.track(fake_frames)

    save_trajectory(trajectory, "output/test_trajectory.json")
    valid_count = sum(1 for r in trajectory if r.valid)
    print(f"Оброблено кадрів: {len(trajectory)}, валідних: {valid_count}")
    print(f"Перша точка: {trajectory[0].center}, остання: {trajectory[-1].center}")