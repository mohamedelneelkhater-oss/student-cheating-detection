"""Helpers for locating dataset videos."""

from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
DEFAULT_DATASET_DIRS = ["/kaggle/input", "samples", "data", "dataset"]


def find_videos(search_dirs):
    """Recursively collect video files from the given directories (sorted, unique)."""
    videos = set()
    for directory in search_dirs:
        root = Path(str(directory).strip())
        if root.exists() and root.is_dir():
            for path in root.rglob("*"):
                if path.suffix.lower() in VIDEO_EXTENSIONS:
                    videos.add(str(path))
    return sorted(videos)
