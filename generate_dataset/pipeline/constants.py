"""Shared constants for the dataset generation pipeline."""

FINAL_DISTORTION_TYPES = (
    "gaussian_noise",
    "glass_blur",
    "motion_blur",
    "defocus_blur",
    "contrast",
    "pixelate",
    "jpeg_compression",
    "darkness",
    "snow",
    "fog",
)

SELECTED_RANKS = (1, 5, 8, 10, 12)

FG_STRONGER_PAIRS = (
    (2, 1),
    (3, 1),
    (4, 1),
    (5, 1),
    (3, 2),
    (4, 2),
    (5, 2),
    (4, 3),
    (5, 3),
    (5, 4),
)

BG_STRONGER_PAIRS = (
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 4),
    (3, 5),
    (4, 5),
)

