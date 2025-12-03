from pathlib import Path
from typing import Sequence, Tuple
import numpy as np
from skimage import measure

from taubin import taubin_smoothing
from gaussian import restricted_gaussian
from subdivision import loop_subdivision

# File paths
INPUT_PATH = "phi_00346.npy"
OUTPUT_PATH = "output_remesh_7.ply"

# Mesh parameters
RESOLUTION = 256
THICKNESS = 6

# Magic numbers
SIGMA = 1.0                 # Gaussian smoothing sigma
LEVEL = 0.0                 # Level set value for surface extraction
LAMBDA = 0.50               # Taubin smoothing lambda
MU = -0.53                  # Taubin smoothing mu
ATTATCH_THRESHOLD = 0.1     # Threshold for attaching boundary vertices

UPSAMPLE = 1                # Volume upsampling factor
MAIN_ITERATIONS = 10        # Number of Taubin smoothing iterations
SUB_ITERATIONS = 0          # Number of Loop subdivision iterations
GAUSSIAN_ITERATIONS = 1     # Number of restricted Gaussian smoothing iterations


def combine_water_and_solid(level_set: np.ndarray) -> np.ndarray:
    solid_phi = np.zeros((RESOLUTION, RESOLUTION, RESOLUTION), dtype=np.float32)
    for i in range(RESOLUTION):
        for j in range(RESOLUTION):
            for k in range(RESOLUTION):
                if  i < THICKNESS or i >= RESOLUTION - THICKNESS or \
                    j < THICKNESS or j >= RESOLUTION - THICKNESS or \
                    k < THICKNESS or k >= RESOLUTION - THICKNESS:
                    solid_phi[i][j][k] = -1.0
                else:
                    solid_phi[i][j][k] = 1.0
    combined = np.maximum(level_set, -solid_phi)
    return combined.astype(np.float32, copy=False)


def extract_active_region(volume: np.ndarray, pad: int = 2) -> Tuple[np.ndarray, np.ndarray]:

    water_mask = volume <= 0.0

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
    if factor <= 1:
        return volume.astype(np.float32, copy=False)

    upsampled = volume.astype(np.float32, copy=False)
    for axis in range(3):
        upsampled = _upsample_axis(upsampled, factor, axis)
    return upsampled


def run_marching_cubes(volume: np.ndarray, level: float, spacing: Sequence[float], origin: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    verts, faces, _, _ = measure.marching_cubes(volume, level=level, spacing=spacing)
    verts = verts.astype(np.float32, copy=False)
    faces = faces.astype(np.int32, copy=False)

    verts[:, 0] += origin[0]
    verts[:, 1] += origin[1]
    verts[:, 2] += origin[2]
    verts = verts[:, [2, 1, 0]]
    return verts, faces


def attach_boundary(vertices: np.ndarray, threshold: float) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float32)
    for i in range(len(verts)):
        for d in range(3):
            if abs(verts[i, d] - THICKNESS) < threshold:
                verts[i, d] = float(THICKNESS)
            if abs(verts[i, d] - (RESOLUTION - THICKNESS)) < threshold:
                verts[i, d] = float(RESOLUTION - THICKNESS)
    return verts.astype(vertices.dtype, copy=False)


def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
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
            # CHANGE THE ORDER !!!
            f.write(f"3 {tri[0]} {tri[2]} {tri[1]}\n")


def level_set_to_mesh(level_set: np.ndarray, output_path: Path) -> None:
    level_set = restricted_gaussian(level_set, SIGMA, RESOLUTION, THICKNESS, GAUSSIAN_ITERATIONS)
    combined_volume = combine_water_and_solid(level_set)
    sub_volume, offset = extract_active_region(combined_volume)
    refined = upsample_volume(sub_volume, UPSAMPLE)
    spacing = tuple(1.0 / UPSAMPLE for _ in range(3))
    origin = tuple(float(val) for val in offset)
    vertices, faces = run_marching_cubes(refined, level=LEVEL, spacing=spacing, origin=origin)
    for _ in range(MAIN_ITERATIONS):
        vertices, faces = taubin_smoothing(vertices, faces, RESOLUTION, THICKNESS, LAMBDA, MU)
    # Final check to deal with bubbles
    vertices = attach_boundary(vertices, ATTATCH_THRESHOLD)
    vertices, faces = loop_subdivision(vertices, faces, SUB_ITERATIONS, RESOLUTION, THICKNESS)
    write_ply(Path(output_path), vertices, faces)


def export(input_path: str, output_path: str) -> None:
    input_path = Path(input_path)
    level_set = np.load(input_path).astype(np.float32, copy=False)
    level_set_to_mesh(level_set, Path(output_path))


if __name__ == "__main__":
    export(INPUT_PATH, OUTPUT_PATH)