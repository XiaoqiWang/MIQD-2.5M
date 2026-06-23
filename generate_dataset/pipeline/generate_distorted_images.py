"""Generate 12 PSNR-ordered distortion levels for each image."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import cupy as cp
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

from distortions.distortion_funcs import get_distortion_methods
from distortions.distortion_levels import get_distortion_ranges
from pipeline.constants import FINAL_DISTORTION_TYPES
from pipeline.image_io import (
    list_images,
    load_rgb_image,
    read_csv_image_list,
    read_text_image_list,
    resolve_image_name,
    slice_items,
)


def set_seed(seed: int) -> None:
    """Set all random seeds used by the distortion functions."""
    np.random.seed(seed)
    cp.random.seed(seed)
    random.seed(seed)


def parse_distortion_names(value: str, available_names: list[str]) -> list[str]:
    """Parse a distortion selector."""
    if value == "all":
        return available_names
    if value == "final":
        return [name for name in FINAL_DISTORTION_TYPES if name in available_names]
    return [name.strip() for name in value.split(",") if name.strip()]


def collect_image_files(args: argparse.Namespace) -> list[str]:
    """Collect input image file names from a directory, CSV, or text file."""
    if args.image_list_csv:
        image_files = read_csv_image_list(args.image_list_csv, args.image_list_column)
    elif args.image_list_txt:
        names = read_text_image_list(args.image_list_txt)
        image_files = []
        for name in names:
            resolved = resolve_image_name(args.input_dir, name)
            if resolved is not None:
                image_files.append(resolved)
    else:
        image_files = list_images(args.input_dir)

    return slice_items(image_files, start=args.start_index, limit=args.limit, stride=args.stride)


def save_distorted_image(image: Image.Image, output_path: Path) -> None:
    """Save a distorted image using the original PNG settings."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=100, optimize=True)


def generate_distortions_for_image(
    image_path: Path,
    output_root: Path,
    distortion_names: list[str],
    sort_by_psnr: bool = True,
) -> None:
    """Generate all requested distortions for one image."""
    image_orig = load_rgb_image(image_path)
    image_stem = image_path.stem
    image_output_dir = output_root / image_stem
    image_output_dir.mkdir(parents=True, exist_ok=True)

    distortion_methods = get_distortion_methods()
    for method_name in tqdm(distortion_names, leave=False):
        distortion_fn = distortion_methods[method_name]
        dist_levels = get_distortion_ranges(method_name)
        level_results = []

        for source_level, level_param in enumerate(
            tqdm(dist_levels, desc=f"Processing {method_name}", leave=False),
            1,
        ):
            image_dis = distortion_fn(image_orig, level_param)
            if isinstance(image_dis, cp.ndarray):
                image_dis = image_dis.get()

            image_dis = Image.fromarray(np.uint8(image_dis))
            psnr_value = psnr(np.array(image_orig), np.array(image_dis))
            level_results.append(
                {
                    "image": image_dis,
                    "psnr": psnr_value,
                    "source_level": source_level,
                }
            )

        if sort_by_psnr:
            level_results.sort(key=lambda item: item["psnr"], reverse=True)

        for new_level, result in enumerate(level_results, 1):
            output_path = image_output_dir / f"{image_stem}_{method_name}_{new_level}.png"
            save_distorted_image(result["image"], output_path)


def generate_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    image_files: list[str],
    distortion_names: list[str],
    seed_offset: int = 1,
    sort_by_psnr: bool = True,
) -> None:
    """Generate distortion images for a list of image file names."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, image_name in tqdm(
        list(enumerate(image_files)),
        desc="Processing images",
        total=len(image_files),
    ):
        image_path = input_dir / image_name
        if not image_path.exists():
            print(f"Skip missing image: {image_path}")
            continue

        set_seed(idx + seed_offset)
        generate_distortions_for_image(
            image_path=image_path,
            output_root=output_dir,
            distortion_names=distortion_names,
            sort_by_psnr=sort_by_psnr,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate 12 distortion levels and save them in PSNR order."
    )
    parser.add_argument("--input-dir", required=True, help="Directory with source images.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated distortions.")
    parser.add_argument("--image-list-csv", help="Optional CSV with image file names.")
    parser.add_argument("--image-list-column", default="Image Name", help="CSV column name.")
    parser.add_argument("--image-list-txt", help="Optional text file with image names or base names.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in the image list.")
    parser.add_argument("--limit", type=int, help="Maximum number of images to process.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth image from the list.")
    parser.add_argument(
        "--distortions",
        default="all",
        help="Use 'final', 'all', or a comma-separated distortion list.",
    )
    parser.add_argument("--seed-offset", type=int, default=1, help="Seed added to the image index.")
    parser.add_argument(
        "--keep-native-order",
        action="store_true",
        help="Save levels in parameter order instead of PSNR order.",
    )
    return parser


def main() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    args = build_parser().parse_args()

    distortion_methods = get_distortion_methods()
    distortion_names = parse_distortion_names(args.distortions, list(distortion_methods.keys()))
    unknown = [name for name in distortion_names if name not in distortion_methods]
    if unknown:
        raise ValueError(f"Unsupported distortions: {unknown}")

    image_files = collect_image_files(args)
    generate_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_files=image_files,
        distortion_names=distortion_names,
        seed_offset=args.seed_offset,
        sort_by_psnr=not args.keep_native_order,
    )


if __name__ == "__main__":
    main()
