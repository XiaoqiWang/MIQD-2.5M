"""Image loading and file-list helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image


IMAGE_EXTENSIONS = (".JPEG", ".jpg", ".jpeg", ".png")


def load_rgb_image(path: str | Path) -> Image.Image:
    """Load an image as RGB and close the file handle."""
    with open(path, "rb") as f:
        image = Image.open(f)
        return image.convert("RGB")


def list_images(input_dir: str | Path, extensions: Iterable[str] = IMAGE_EXTENSIONS) -> list[str]:
    """List image file names in a directory."""
    suffixes = tuple(extensions)
    return sorted(
        f
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(suffixes)
    )


def read_csv_image_list(csv_path: str | Path, column: str = "Image Name") -> list[str]:
    """Read image file names from a CSV column."""
    data_info = pd.read_csv(csv_path)
    return data_info[column].dropna().astype(str).tolist()


def read_text_image_list(txt_path: str | Path) -> list[str]:
    """Read one image name or base name per line."""
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def resolve_image_name(input_dir: str | Path, name: str) -> str | None:
    """Resolve a file name or base name to an existing image file name."""
    input_dir = Path(input_dir)
    candidate = input_dir / name
    if candidate.exists():
        return name

    stem = Path(name).stem
    for ext in IMAGE_EXTENSIONS:
        candidate = input_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate.name
    return None


def resolve_original_path(orig_dir: str | Path, base_name: str, preferred_ext: str | None = None) -> Path | None:
    """Find the original image path for a generated image folder."""
    orig_dir = Path(orig_dir)
    extensions = (preferred_ext,) + IMAGE_EXTENSIONS if preferred_ext else IMAGE_EXTENSIONS
    for ext in extensions:
        candidate = orig_dir / f"{base_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def slice_items(items: list[str], start: int = 0, limit: int | None = None, stride: int = 1) -> list[str]:
    """Apply start, limit, and stride to a file list."""
    selected = items[start::stride]
    if limit is not None:
        selected = selected[:limit]
    return selected

