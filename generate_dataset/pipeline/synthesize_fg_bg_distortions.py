"""Fuse foreground and background images with different distortion levels."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from pipeline.constants import BG_STRONGER_PAIRS, FG_STRONGER_PAIRS, FINAL_DISTORTION_TYPES
from pipeline.image_io import read_text_image_list, slice_items


def create_smooth_mask(mask: np.ndarray, blur_radius: int = 5) -> np.ndarray:
    """Create a smooth transition mask from a binary mask."""
    smooth_mask = cv2.GaussianBlur(mask, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
    smooth_mask = smooth_mask / 255.0

    # Keep the core foreground area unchanged.
    kernel = np.ones((3, 3), np.uint8)
    core_area = cv2.erode(mask, kernel, iterations=2)
    smooth_mask[core_area > 0] = 1.0

    return smooth_mask


def create_fused_image(mask: np.ndarray, fg_img: np.ndarray, bg_img: np.ndarray, blur_radius: int = 5) -> np.ndarray:
    """Fuse foreground and background images with a smooth mask."""
    smooth_mask = create_smooth_mask(mask, blur_radius)
    mask_3d = np.stack([smooth_mask] * 3, axis=2)
    fused = fg_img * mask_3d + bg_img * (1 - mask_3d)

    # Blur only the transition region to reduce hard mask edges.
    transition_mask = np.logical_and(smooth_mask > 0.05, smooth_mask < 0.95)
    transition_mask = transition_mask.astype(np.uint8) * 255
    transition_blur = cv2.GaussianBlur(fused, (3, 3), 0)
    fused = np.where(
        np.stack([transition_mask] * 3, axis=2) > 0,
        cv2.addWeighted(fused, 0.5, transition_blur, 0.5, 0),
        fused,
    )

    return fused.astype(np.uint8)


def load_images(
    base_name: str,
    distortion_type: str,
    num_levels: int,
    input_dir: str | Path,
    mask_dir: str | Path,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Load the binary mask and all selected distortion levels."""
    mask_path = Path(mask_dir) / f"{base_name}_binary_mask.png"
    binary_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if binary_mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    distorted_images = []
    for level in range(1, num_levels + 1):
        img_path = Path(input_dir) / base_name / f"{base_name}_{distortion_type}_{level}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Distorted image not found: {img_path}")
        distorted_images.append(img)

    return binary_mask, distorted_images


def process_image_pairs(
    base_name: str,
    pairs: tuple[tuple[int, int], ...],
    output_dir: str | Path,
    input_dir: str | Path,
    mask_dir: str | Path,
    is_fg_stronger: bool = True,
    distortion_type: str = "gaussian_noise",
    blur_radius: int = 32,
    num_levels: int = 5,
) -> None:
    """Generate fused images for a list of foreground/background level pairs."""
    try:
        mask, distorted_images = load_images(base_name, distortion_type, num_levels, input_dir, mask_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for fg_level, bg_level in pairs:
            fg_img = distorted_images[fg_level - 1]
            bg_img = distorted_images[bg_level - 1]
            fused_img = create_fused_image(mask, fg_img, bg_img, blur_radius)

            condition = "fg_stronger" if is_fg_stronger else "bg_stronger"
            output_filename = f"{base_name}_{distortion_type}_{condition}_fg{fg_level}_bg{bg_level}.png"
            output_path = Path(output_dir) / output_filename
            cv2.imwrite(str(output_path), fused_img)
    except Exception as exc:
        print(f"Error processing {base_name}: {exc}")


def parse_distortion_names(value: str) -> list[str]:
    """Parse distortion names from the CLI."""
    if value == "final":
        return list(FINAL_DISTORTION_TYPES)
    return [name.strip() for name in value.split(",") if name.strip()]


def list_base_names(input_dir: str | Path, base_list_txt: str | Path | None = None) -> list[str]:
    """List selected-level image folders or read them from a text file."""
    if base_list_txt:
        return [Path(name).stem for name in read_text_image_list(base_list_txt)]

    input_dir = Path(input_dir)
    return sorted(path.name for path in input_dir.iterdir() if path.is_dir())


def synthesize_dataset(
    input_dir: str | Path,
    mask_dir: str | Path,
    output_dir: str | Path | None,
    base_names: list[str],
    distortion_types: list[str],
    blur_radius: int = 32,
    num_levels: int = 5,
) -> None:
    """Create foreground/background distortion composites for a dataset."""
    input_dir = Path(input_dir)
    output_root = Path(output_dir) if output_dir else input_dir

    for base_name in tqdm(base_names, desc="Processing images"):
        image_output_dir = output_root / base_name
        for distortion_type in distortion_types:
            process_image_pairs(
                base_name=base_name,
                pairs=FG_STRONGER_PAIRS,
                output_dir=image_output_dir,
                input_dir=input_dir,
                mask_dir=mask_dir,
                is_fg_stronger=True,
                distortion_type=distortion_type,
                blur_radius=blur_radius,
                num_levels=num_levels,
            )
            process_image_pairs(
                base_name=base_name,
                pairs=BG_STRONGER_PAIRS,
                output_dir=image_output_dir,
                input_dir=input_dir,
                mask_dir=mask_dir,
                is_fg_stronger=False,
                distortion_type=distortion_type,
                blur_radius=blur_radius,
                num_levels=num_levels,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuse foreground and background distortion levels."
    )
    parser.add_argument("--input-dir", required=True, help="Directory with selected 5-level image folders.")
    parser.add_argument("--mask-dir", required=True, help="Directory with binary masks.")
    parser.add_argument(
        "--output-dir",
        help="Output directory. Defaults to writing into the selected-level dataset directory.",
    )
    parser.add_argument("--base-list-txt", help="Optional text file with image base names.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in the base-name list.")
    parser.add_argument("--limit", type=int, help="Maximum number of images to process.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth image.")
    parser.add_argument("--blur-radius", type=int, default=32, help="Mask transition blur radius.")
    parser.add_argument("--num-levels", type=int, default=5, help="Number of selected distortion levels.")
    parser.add_argument(
        "--distortions",
        default="final",
        help="Use 'final' or a comma-separated distortion list.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_names = list_base_names(args.input_dir, args.base_list_txt)
    base_names = slice_items(base_names, start=args.start_index, limit=args.limit, stride=args.stride)
    synthesize_dataset(
        input_dir=args.input_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        base_names=base_names,
        distortion_types=parse_distortion_names(args.distortions),
        blur_radius=args.blur_radius,
        num_levels=args.num_levels,
    )


if __name__ == "__main__":
    main()

