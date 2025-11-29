from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from skimage import measure

RESOLUTION = 64  # grid resolution along each axis
INPUT_PATH = Path("input_0000.txt")
OUTPUT_PATH = Path("output.ply")
UPSAMPLE_FACTOR = 8  # target spacing is 1/8 of original cell length


def load_volume(path: Path, res: int) -> np.ndarray:
    """Return a (res, res, res) float32 volume containing sdf."""

    expected = res ** 3
    if not path.exists():
        raise FileNotFoundError(f"Missing SDF file: {path}")

    # Try fast path first: text file with whitespace-separated floats.
    data = np.fromfile(path, dtype=np.float32, count=expected, sep=" ")

    if data.size != expected:
        # Fallback to a more forgiving parser (e.g., scientific notation, newlines).
        data = np.loadtxt(path, dtype=np.float32)
        if data.size != expected:
            raise ValueError(
                f"Expected {expected} SDF samples but found {data.size} entries in {path}"
            )

    volume = data.reshape((res, res, res))
    return volume.astype(np.float32, copy=False)


def extract_active_region(volume: np.ndarray, pad: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Return tight sub-volume enclosing the water (sdf <= 0) plus padding."""

    water_mask = volume <= 0.0
    if not np.any(water_mask):
        raise ValueError("SDF volume does not contain any water region (values <= 0)")

    indices = np.argwhere(water_mask)
    mins = indices.min(axis=0)
    maxs = indices.max(axis=0)

    mins = np.maximum(mins - pad, 0)
    maxs = np.minimum(maxs + pad, np.array(volume.shape) - 1)

    z0, y0, x0 = mins.astype(int)
    z1, y1, x1 = maxs.astype(int)

    sub_volume = volume[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
    offset = np.array([z0, y0, x0], dtype=np.float32)
    return sub_volume, offset


def _upsample_axis(arr: np.ndarray, factor: int, axis: int) -> np.ndarray:
    """Linearly upsample `arr` along `axis` by `factor`."""

    if factor <= 1:
        return arr

    arr = np.swapaxes(arr, axis, 0)
    depth = arr.shape[0]

    if depth == 1:
        result = np.repeat(arr, 1, axis=0)
    else:
        new_depth = (depth - 1) * factor + 1
        trailing_shape = arr.shape[1:]
        result = np.empty((new_depth, *trailing_shape), dtype=arr.dtype)
        result[::factor] = arr

        for idx in range(depth - 1):
            start = arr[idx]
            end = arr[idx + 1]
            for step in range(1, factor):
                t = step / factor
                result[idx * factor + step] = start * (1.0 - t) + end * t

    result = np.swapaxes(result, 0, axis)
    return result


def upsample_volume(volume: np.ndarray, factor: int) -> np.ndarray:
    """Trilinear upsampling by repeatedly applying 1D interpolation."""

    if factor <= 1:
        return volume.astype(np.float32, copy=False)

    upsampled = volume.astype(np.float32, copy=False)
    for axis in range(3):
        upsampled = _upsample_axis(upsampled, factor, axis)
    return upsampled


def run_marching_cubes(
    volume: np.ndarray,
    level: float,
    spacing: Sequence[float],
    origin: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Run marching cubes via scikit-image and adjust coordinates."""

    if measure is None:
        raise ImportError(
            "scikit-image is required for marching cubes. Install it via `pip install scikit-image`."
        )

    verts, faces, _, _ = measure.marching_cubes(volume, level=level, spacing=spacing)
    verts = verts.astype(np.float32, copy=False)
    faces = faces.astype(np.int32, copy=False)

    verts[:, 0] += origin[0]
    verts[:, 1] += origin[1]
    verts[:, 2] += origin[2]
    verts = verts[:, [2, 1, 0]]
    return verts, faces


def build_mesh(volume: np.ndarray, upsample_factor: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the water surface mesh at higher sampling density."""

    sub_volume, offset = extract_active_region(volume)
    refined = upsample_volume(sub_volume, upsample_factor)
    spacing = tuple(1.0 / upsample_factor for _ in range(3))
    origin = tuple(float(val) for val in offset)
    vertices, faces = run_marching_cubes(refined, level=0.0, spacing=spacing, origin=origin)
    return vertices, faces




def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Serialize mesh data into ASCII PLY format."""

    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for vx, vy, vz in vertices:
            f.write(f"{vx:.6f} {vy:.6f} {vz:.6f}\n")

        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def main(volume, output_path) -> None:
    # volume = load_volume(INPUT_PATH, RESOLUTION)
    vertices, faces = build_mesh(volume, UPSAMPLE_FACTOR)
    write_ply(Path(output_path), vertices, faces)


if __name__ == "__main__":
    output_path = OUTPUT_PATH
    main(load_volume(INPUT_PATH, RESOLUTION), output_path)