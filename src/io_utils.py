from pathlib import Path
import re
import cv2


def _natural_sort_key(path: Path):
    """
    Helper key for natural numerical sorting of filenames.
    Splits filename into text and integer parts so 'frame_2.jpg' comes before 'frame_10.jpg'.
    """
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def load_frames_from_dir(folder_path):
    """
    Load images from a directory in natural numerical order.
    Returns a list of OpenCV BGR frames.
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    # Supported image extensions
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # Collect matching files and sort naturally by embedded numbers
    image_files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]
    image_files.sort(key=_natural_sort_key)

    frames = []
    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is not None:
            frames.append(frame)

    return frames


def load_frames_from_video(video_path, stride=5):
    """
    Open a video file and extract every `stride`-th frame.
    For example, stride=5 extracts frames at indices 0, 5, 10, 15, ...
    """
    if stride < 1:
        raise ValueError("Parameter 'stride' must be at least 1.")

    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Sample every stride-th frame
            if frame_index % stride == 0:
                frames.append(frame)

            frame_index += 1
    finally:
        # Always release video capture handle
        cap.release()

    return frames


if __name__ == "__main__":
    # Quick local sanity check
    test_dir = "test_frames"
    test_video = "test_video.mp4"

    if Path(test_dir).exists():
        dir_frames = load_frames_from_dir(test_dir)
        print(f"Loaded images from directory: {len(dir_frames)}")

    if Path(test_video).exists():
        vid_frames = load_frames_from_video(test_video, stride=5)
        print(f"Extracted frames from video (stride=5): {len(vid_frames)}")