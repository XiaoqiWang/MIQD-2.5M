# MIQD-2.5M Dataset Generation

This directory contains the scripts used to build the MIQD distortion dataset.
The code is organized as a three-step pipeline:

1. Generate 12 distorted candidates for each original image.
2. Re-score the 12 candidates and keep ranks `[1, 5, 8, 10, 12]` as final levels `1-5`.
3. Fuse foreground and background regions with different final distortion levels.

## Directory Layout

```text
generate_dataset/
|-- distortions/
|   |-- distortion_funcs.py       # Distortion functions
|   |-- distortion_levels.py      # 12 candidate parameter levels
|   `-- figs/                     # Frost texture assets
|-- pipeline/
|   |-- constants.py              # Shared distortion lists and level pairs
|   |-- image_io.py               # Image-list and loading helpers
|   |-- generate_distorted_images.py
|   |-- select_distortion_levels.py
|   `-- synthesize_fg_bg_distortions.py
|-- data_info/                    # ImageNet metadata
`-- README.md
```

## Distortion Types

The selected and fused dataset uses these 10 distortion types:

```text
gaussian_noise, glass_blur, motion_blur, defocus_blur,
contrast, pixelate, jpeg_compression, darkness, snow, fog
```

`pipeline/generate_distorted_images.py` defaults to all distortion functions
registered by `distortions/distortion_funcs.py`, matching the original
candidate-generation scripts. Use `--distortions final` when you only need the
10 final types.

## Step 1: Generate 12 PSNR-Ordered Levels

Generate ImageNet candidates:

```bash
python -m pipeline.generate_distorted_images \
  --input-dir /public/datasets/imagenet/val \
  --output-dir /public/datasets/imagenet_c_900k \
  --image-list-csv data_info/top_5_quality_images_per_category.csv \
  --image-list-column "Image Name" \
```

Generate COCO candidates:

```bash
python -m pipeline.generate_distorted_images \
  --input-dir /public/datasets/coco2017/val2017 \
  --output-dir /public/datasets/coco_c_900k \
```

Generate only a COCO gray subset:

```bash
python -m pipeline.generate_distorted_images \
  --input-dir /public/datasets/coco2017/val2017 \
  --output-dir /public/datasets/coco_c_900k \
  --image-list-txt coco_gray.txt
```

Output format:

```text
<output-dir>/<image_id>/<image_id>_<distortion>_<level>.png
```

The saved level is `1-12`, sorted by PSNR from high quality to low quality.

## Step 2: Keep Five Final Levels

The final levels are selected by rank:

```python
selected_ranks = [1, 5, 8, 10, 12]
```

The selected images are copied and renamed to final levels `1-5`.

ImageNet:

```bash
python -m pipeline.select_distortion_levels \
  --candidates-dir /public/datasets/imagenet_c_900k \
  --originals-dir /public/datasets/imagenet/val \
  --output-dir /public/datasets/miqa_cls \
  --preferred-ext .JPEG \
  --limit 2500 \
  --cuda-device 1
```

COCO:

```bash
python -m pipeline.select_distortion_levels \
  --candidates-dir /public/datasets/coco_c_900k \
  --originals-dir /public/datasets/coco2017/val2017 \
  --output-dir /public/datasets/miqa_det \
  --preferred-ext .jpg \
  --limit 2500 \
  --cuda-device 2
```

Some COCO images with very small width/height dimensions require additional processing:

```bash
python -m pipeline.select_distortion_levels \
  --candidates-dir /public/datasets/coco_c_900k \
  --originals-dir /public/datasets/coco2017/val2017 \
  --output-dir /public/datasets/miqa_det \
  --folder-list-txt coco_gray.txt \
  --preferred-ext .jpg \
  --min-side 161 \
  --cuda-device 0
```

Each output folder contains:

```text
<image_id>_<distortion>_1.png
...
<image_id>_<distortion>_5.png
<image_id>_all_iqa_scores.csv
<image_id>_iqa_scores.csv
```

## Step 3: Fuse Foreground and Background Distortions
Download the binary mask archive from 👉[Google Drive](https://drive.google.com/file/d/14cbhjIW95Be9zwhf0fJc_RAwXBrhXTHi/view?usp=sharing) 

Extract it to a local folder and two folders will be obtained:

```text
imagenet_val_binary_mask/
mscoco_val_binary_mask/
```

ImageNet foreground/background synthesis:

```bash
python -m pipeline.synthesize_fg_bg_distortions \
  --input-dir /public/datasets/miqa_cls \
  --mask-dir /home/wxq/projects/miqa/dino-vit-features/imagenet_val_binary_mask \
  --limit 2500
```

COCO foreground/background synthesis:

```bash
python -m pipeline.synthesize_fg_bg_distortions \
  --input-dir /public/datasets/miqa_det \
  --mask-dir /home/wxq/projects/miqa/dino-vit-features/mscoco_val_binary_mask \
  --limit 2500
```

By default, fused images are written back into each image folder. Use
`--output-dir` to write them to a separate directory.

The script creates both cases:

```text
fg_stronger: foreground level > background level
bg_stronger: background level > foreground level
```

Example output:

```text
<image_id>_<distortion>_fg_stronger_fg5_bg3.png
<image_id>_<distortion>_bg_stronger_fg2_bg5.png
```

## Useful Options

Common options:

```text
--start-index N       Skip the first N files.
--limit N             Process at most N files.
--stride N            Process every Nth file.
--distortions all     Generate all registered distortion types.
--distortions final   Use only the final 10 distortion types.
```

Custom distortion list:

```bash
--distortions gaussian_noise,glass_blur,motion_blur
```

## Dependencies

The pipeline expects the original environment used by this project:

```text
cupy, numpy, opencv-python, pillow, scikit-image, scipy, wand,
tqdm, pandas, torch, torchvision, piq
```

`wand` requires ImageMagick to be installed on the system.
