"""Distortion parameter levels.

Each distortion returns 12 candidate parameter values. The generation pipeline
applies these candidates, computes PSNR against the original image, and saves
the generated images in descending PSNR order.
"""

from __future__ import annotations

import numpy as np
import cupy as cp


def get_distortion_ranges(method_name, levels=12):
    """Return candidate parameters for a distortion method."""
    if method_name == "gaussian_noise":
        dist_levels = distortion_level_method([0.01, 1], levels, mode="log")

    elif method_name == "impulse_noise":
        dist_levels = distortion_level_method([0.0005, 0.5], levels, mode="log")

    elif method_name == "defocus_blur":
        dist_levels = [
            (0.95, 0.5),
            (0.99, 0.54),
            (0.99, 0.6),
            (0.99, 0.7),
            (0.999, 0.9),
            (1, 0.9),
            (2, 0.1),
            (3, 0.5),
            (5, 0.5),
            (7, 0.8),
            (9, 0.9),
            (12, 0.9),
        ]

    elif method_name == "glass_blur":
        dist_levels = [
            (0.4, 1, 0.4),
            (0.42, 1.5, 0.8),
            (0.45, 1.5, 0.9),
            (0.49, 1.5, 0.9),
            (0.55, 1.5, 0.9),
            (0.65, 2, 0.9),
            (0.8, 2.5, 0.9),
            (1.2, 2.5, 0.99),
            (1.5, 2, 1.2),
            (2, 2, 2),
            (3, 3, 3),
            (4, 5, 4),
        ]

    elif method_name == "motion_blur":
        dist_levels = [
            (1.4, 0.6),
            (1.5, 0.7),
            (2, 0.9),
            (4, 1.5),
            (7, 2.5),
            (10.5, 3.5),
            (15, 5.0),
            (20, 7),
            (27, 10),
            (35, 14),
            (47, 20),
            (60, 30),
        ]

    elif method_name == "gaussian_blur":
        dist_levels = [0.45, 0.5, 0.575, 0.65, 0.8, 1, 1.5, 2.3, 3.5, 6, 9, 12]

    elif method_name == "snow":
        start_params = (0.05, 0.1, 1, 0.3, 5, 2, 1)
        end_params = (0.6, 0.4, 5, 0.9, 15, 15, 0.5)
        orig_params_np = np.array([start_params, end_params])
        param_ranges = orig_params_np[-1] - orig_params_np[0]
        factors = np.linspace(0, 1, levels)
        dist_levels = orig_params_np[0] + factors[:, None] * param_ranges[None, :]

    elif method_name == "frost":
        dist_levels = [
            [1.0, 0.05, 1],
            [0.97, 0.06, 2],
            [0.95, 0.18, 3],
            [0.95, 0.14, 4],
            [0.92, 0.25, 5],
            [0.9, 0.27, 1],
            [0.9, 0.2, 2],
            [0.7, 0.65, 3],
            [0.85, 0.37, 4],
            [0.85, 0.55, 5],
            [0.82, 0.58, 1],
            [0.7, 0.8, 5],
        ]

    elif method_name == "fog":
        dist_levels = [
            [0.1, 3.0],
            [0.13, 2.85],
            [0.2, 2.5],
            [0.25, 2.2],
            [0.3, 2.05],
            [0.34, 1.89],
            [0.45, 1.7],
            [0.7, 1.6],
            [0.9, 1.5],
            [1.5, 1.5],
            [2.9, 1.3],
            [5.0, 1.3],
        ]

    elif method_name == "brightness":
        dist_levels = distortion_level_method([0.035, 0.7], levels, mode="increasing")

    elif method_name == "darkness":
        dist_levels = distortion_level_method([0.015, 0.7], levels, mode="log")

    elif method_name == "contrast":
        dist_levels = [0.9, 0.818, 0.736, 0.655, 0.573, 0.491, 0.409, 0.327, 0.246, 0.164, 0.082, 0.05]

    elif method_name == "pixelate":
        dist_levels = [0.95, 0.865, 0.78, 0.695, 0.61, 0.525, 0.44, 0.355, 0.27, 0.185, 0.1, 0.07]

    elif method_name == "jpeg_compression":
        dist_levels = distortion_level_method([100, 0.1], levels, mode="default")

    elif method_name == "saturate":
        dist_levels = [
            (1.1, 0.025),
            (1.2, 0.027),
            (1.3, 0.027),
            (1.5, 0.03),
            (1.6, 0.04),
            (1.7, 0.05),
            (1.8, 0.06),
            (2, 0.08),
            (2.5, 0.09),
            (3, 0.1),
            (5, 0.15),
            (10, 0.35),
        ]

    else:
        print(f"{method_name} is not supported.")
        dist_levels = None

    if isinstance(dist_levels, cp.ndarray):
        dist_levels = dist_levels.get()

    return dist_levels


def distortion_level_method(dist_ranges, num_samples, mode="increasing"):
    """Sample scalar distortion parameters from a numeric range."""
    if mode == "increasing":
        ratio = (dist_ranges[1] / dist_ranges[0]) ** (1 / (num_samples - 1))
        levels = [dist_ranges[0] * ratio**i for i in range(num_samples)]
    elif mode == "decreasing":
        ratio = (dist_ranges[0] / dist_ranges[1]) ** (1 / (num_samples - 1))
        levels = [dist_ranges[1] * ratio**i for i in range(num_samples)]
    elif mode == "log":
        levels = np.logspace(np.log10(dist_ranges[0]), np.log10(dist_ranges[1]), num=num_samples)
    else:
        levels = np.linspace(dist_ranges[0], dist_ranges[1], num_samples)

    return levels
