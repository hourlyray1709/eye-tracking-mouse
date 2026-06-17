from pathlib import Path

def unique_filename(prefix, suffix=""):
    path = Path(f"{prefix}{suffix}")

    if not path.exists():
        return path

    i = 1
    while True:
        path = Path(f"{prefix}_{i}{suffix}")
        if not path.exists():
            return path
        i += 1