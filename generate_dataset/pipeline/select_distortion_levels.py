"""Select five final distortion levels from 12 PSNR-ranked candidates."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import piq
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from pipeline.constants import FINAL_DISTORTION_TYPES, SELECTED_RANKS
from pipeline.image_io import read_text_image_list, resolve_original_path, slice_items


def load_and_preprocess_image(image_path: str | Path, min_side: int | None = None) -> torch.Tensor | None:
    """Load an RGB image as a CUDA tensor."""
    try:
        img = Image.open(image_path).convert("RGB")
        if min_side and min(img.size) < min_side:
            ratio = min_side / min(img.size)
            img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)))
        return to_tensor(img).unsqueeze(0).cuda()
    except Exception as exc:
        print(f"Error processing image {image_path}: {exc}")
        return None


def compute_iqa_metrics(batch_orig: torch.Tensor, batch_dist: torch.Tensor, metrics=None) -> dict:
    """Compute image quality metrics for one original/distorted pair."""
    results = {}

    if metrics is None or "psnr" in metrics:
        results["psnr"] = piq.psnr(batch_dist, batch_orig)

    if metrics is None or "ssim" in metrics:
        results["ssim"] = piq.ssim(batch_dist, batch_orig)

    if metrics is None or "ms_ssim" in metrics:
        results["ms_ssim"] = piq.multi_scale_ssim(batch_dist, batch_orig)

    if metrics is None or "vif_p" in metrics:
        results["vif_p"] = piq.vif_p(batch_dist, batch_orig)

    if metrics is None or "vsi" in metrics:
        results["vsi"] = piq.vsi(batch_dist, batch_orig)

    return results


def parse_distortion_names(value: str) -> list[str]:
    """Parse distortion names from the CLI."""
    if value == "final":
        return list(FINAL_DISTORTION_TYPES)
    return [name.strip() for name in value.split(",") if name.strip()]


def list_folders(candidate_dir: str | Path, folder_list_txt: str | Path | None = None) -> list[str]:
    """List generated image folders or read them from a text file."""
    if folder_list_txt:
        return [Path(name).stem for name in read_text_image_list(folder_list_txt)]

    candidate_dir = Path(candidate_dir)
    return sorted(path.name for path in candidate_dir.iterdir() if path.is_dir())


def collect_distorted_images(folder_path: Path, folder_name: str, distortion: str, levels: int) -> list[tuple[Path, int]]:
    """Collect existing distorted images for one distortion type."""
    distorted_images = []
    for level in range(1, levels + 1):
        img_path = folder_path / f"{folder_name}_{distortion}_{level}.png"
        if img_path.exists():
            distorted_images.append((img_path, level))
    return distorted_images


def score_distortion_images(
    orig_tensor: torch.Tensor,
    distorted_images: list[tuple[Path, int]],
    distortion: str,
    min_side: int | None = None,
) -> list[dict]:
    """Compute IQA scores for all candidate levels of one distortion."""
    scores = []
    for img_path, level in distorted_images:
        dist_tensor = load_and_preprocess_image(img_path, min_side=min_side)
        if dist_tensor is None:
            continue

        metrics = compute_iqa_metrics(orig_tensor, dist_tensor, None)
        scores.append(
            {
                "path": str(img_path),
                "level": level,
                "distortion": distortion,
                "psnr": metrics["psnr"].item(),
                "ssim": metrics["ssim"].item(),
                "ms_ssim": metrics["ms_ssim"].item(),
                "vif_p": metrics["vif_p"].item(),
                "vsi": metrics["vsi"].item(),
            }
        )
    return scores


def process_folder(
    folder_path: str | Path,
    orig_image_dir: str | Path,
    output_base_dir: str | Path,
    distortions: list[str],
    selected_ranks: tuple[int, ...] = SELECTED_RANKS,
    candidate_levels: int = 12,
    preferred_ext: str | None = None,
    min_side: int | None = None,
    selected_csv_suffix: str = "iqa_scores",
) -> None:
    """Select final levels for one generated image folder."""
    folder_path = Path(folder_path)
    folder_name = folder_path.name
    orig_image_path = resolve_original_path(orig_image_dir, folder_name, preferred_ext)
    if orig_image_path is None:
        print(f"Skip missing original image for folder: {folder_name}")
        return

    orig_tensor = load_and_preprocess_image(orig_image_path, min_side=min_side)
    if orig_tensor is None:
        return

    output_folder = Path(output_base_dir) / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    all_scores = []
    selected_scores = []

    for distortion in distortions:
        distorted_images = collect_distorted_images(folder_path, folder_name, distortion, candidate_levels)
        distortion_scores = score_distortion_images(
            orig_tensor=orig_tensor,
            distorted_images=distorted_images,
            distortion=distortion,
            min_side=min_side,
        )
        all_scores.extend(distortion_scores)
        distortion_scores.sort(key=lambda item: item["psnr"], reverse=True)

        for new_level, rank in enumerate(selected_ranks, 1):
            if rank > len(distortion_scores):
                continue

            selected_score = distortion_scores[rank - 1].copy()
            selected_score["new_level"] = new_level
            selected_score["rank"] = rank
            new_name = f"{folder_name}_{distortion}_{new_level}.png"
            selected_score["new_path"] = str(output_folder / new_name)
            selected_scores.append(selected_score)
            shutil.copy2(distortion_scores[rank - 1]["path"], selected_score["new_path"])

    pd.DataFrame(all_scores).to_csv(output_folder / f"{folder_name}_all_iqa_scores.csv", index=False)
    pd.DataFrame(selected_scores).to_csv(output_folder / f"{folder_name}_{selected_csv_suffix}.csv", index=False)


def select_levels_dataset(
    candidates_dir: str | Path,
    originals_dir: str | Path,
    output_dir: str | Path,
    folders: list[str],
    distortions: list[str],
    candidate_levels: int = 12,
    preferred_ext: str | None = None,
    min_side: int | None = None,
    selected_csv_suffix: str = "iqa_scores",
) -> None:
    """Select final distortion levels for a list of generated folders."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for folder in tqdm(folders, desc="Processing folders"):
        folder_path = Path(candidates_dir) / folder
        if not folder_path.is_dir():
            print(f"Skip missing folder: {folder_path}")
            continue
        try:
            process_folder(
                folder_path=folder_path,
                orig_image_dir=originals_dir,
                output_base_dir=output_dir,
                distortions=distortions,
                candidate_levels=candidate_levels,
                preferred_ext=preferred_ext,
                min_side=min_side,
                selected_csv_suffix=selected_csv_suffix,
            )
        except Exception as exc:
            print(f"Error processing folder {folder}: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select five final levels from 12 distorted candidates."
    )
    parser.add_argument("--candidates-dir", required=True, help="Directory with 12-level distorted folders.")
    parser.add_argument("--originals-dir", required=True, help="Directory with original images.")
    parser.add_argument("--output-dir", required=True, help="Directory for the selected 5-level dataset.")
    parser.add_argument("--folder-list-txt", help="Optional text file with folder/base names.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in the folder list.")
    parser.add_argument("--limit", type=int, help="Maximum number of folders to process.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth folder.")
    parser.add_argument("--candidate-levels", type=int, default=12, help="Number of candidate levels.")
    parser.add_argument("--preferred-ext", help="Preferred original image extension, for example .JPEG or .jpg.")
    parser.add_argument("--min-side", type=int, help="Resize images whose shorter side is below this value.")
    parser.add_argument("--cuda-device", type=int, help="CUDA device id.")
    parser.add_argument(
        "--distortions",
        default="final",
        help="Use 'final' or a comma-separated distortion list.",
    )
    parser.add_argument(
        "--selected-csv-suffix",
        default="iqa_scores",
        help="Suffix for the selected score CSV.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cuda_device is not None:
        torch.cuda.set_device(args.cuda_device)

    folders = list_folders(args.candidates_dir, args.folder_list_txt)
    folders = slice_items(folders, start=args.start_index, limit=args.limit, stride=args.stride)
    select_levels_dataset(
        candidates_dir=args.candidates_dir,
        originals_dir=args.originals_dir,
        output_dir=args.output_dir,
        folders=folders,
        distortions=parse_distortion_names(args.distortions),
        candidate_levels=args.candidate_levels,
        preferred_ext=args.preferred_ext,
        min_side=args.min_side,
        selected_csv_suffix=args.selected_csv_suffix,
    )


if __name__ == "__main__":
    main()

