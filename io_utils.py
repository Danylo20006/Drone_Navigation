from pathlib import Path
import re
import cv2


def _natural_sort_key(path: Path):
    """
    Допоміжна функція для правильного числового сортування.
    Розбиває назву файлу на шматки: текст залишає як є, а числа перетворює на int.
    Наприклад, 'frame_10.jpg' -> ['frame_', 10, '.jpg'],
    а 'frame_2.jpg' -> ['frame_', 2, '.jpg'].
    Тоді число 2 гарантовано стане перед числом 10.
    """
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def load_frames_from_dir(folder_path):
    """
    Зчитує зображення з папки у правильному числовому порядку.
    Повертає список кадрів (кожен кадр — це масив пікселів OpenCV).
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Папку не знайдено: {folder_path}")

    # Підтримувані формати фото
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # Беремо всі файли з потрібним розширенням і сортуємо їх за числами в назві
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
    Відкриває відеофайл (.mp4 тощо) і дістає кожен stride-й кадр.
    Наприклад, при stride=5 візьме кадри з індексами 0, 5, 10, 15...
    """
    if stride < 1:
        raise ValueError("Параметр stride має бути не меншим за 1")

    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Відеофайл не знайдено: {video_path}")

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Не вдалося відкрити відео: {video_path}")

    frames = []
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            # Якщо кадрів більше немає — виходимо з циклу
            if not ok:
                break

            # Беремо лише кожен stride-й кадр
            if frame_index % stride == 0:
                frames.append(frame)

            frame_index += 1
    finally:
        # Обов'язково звільняємо відеофайл після читання
        cap.release()

    return frames


if __name__ == "__main__":
    # Приклад швидкої перевірки (підстав свої шляхи, коли з'являться файли)
    test_dir = "test_frames"
    test_video = "test_video.mp4"

    if Path(test_dir).exists():
        dir_frames = load_frames_from_dir(test_dir)
        print(f"Завантажено фото з папки: {len(dir_frames)}")

    if Path(test_video).exists():
        vid_frames = load_frames_from_video(test_video, stride=5)
        print(f"Дістато кадрів із відео (stride=5): {len(vid_frames)}")