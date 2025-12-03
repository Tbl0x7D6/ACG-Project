""""Gaussian smoothing for 3D volumes."""

import numpy as np
from skimage.filters import gaussian


def _gaussian(volume: np.ndarray, sigma: float, resolution: int, thickness: int) -> np.ndarray:
    smoothed = np.zeros_like(volume)
    label = np.zeros_like(volume, dtype=bool)
    orig = volume.copy()
    neg = orig < 0.0

    for z in range(1, resolution - 1):
        for y in range(1, resolution - 1):
            for x in range(1, resolution - 1):
                all_inside = True
                for dz in (-1, 0, 1):
                    if not all_inside:
                        break
                    for dy in (-1, 0, 1):
                        if not all_inside:
                            break
                        for dx in (-1, 0, 1):
                            if not neg[z + dz, y + dy, x + dx]:
                                all_inside = False
                                break
                if all_inside:
                    label[z, y, x] = True

    for z in range(0, resolution):
        for y in range(0, resolution):
            for x in range(0, resolution):
                if  z == thickness or z == resolution - thickness - 1 or \
                    y == thickness or y == resolution - thickness - 1 or \
                    x == thickness or x == resolution - thickness - 1:
                    label[z, y, x] = True

    smoothed = gaussian(orig, sigma=sigma).astype(np.float32, copy=False)
    smoothed[label] = orig[label]
    return smoothed


def restricted_gaussian(volume: np.ndarray, sigma: float, resolution: int, thickness: int, iterations: int) -> np.ndarray:
    result = volume.astype(np.float32, copy=True)
    soft = 1.0
    for _ in range(iterations):
        result = _gaussian(result, sigma, resolution, thickness) * soft + result * (1.0 - soft)
        sigma /= 3.0
        soft *= 0.5
    return result