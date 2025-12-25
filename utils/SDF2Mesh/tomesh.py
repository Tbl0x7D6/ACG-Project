from pathlib import Path
from typing import Sequence, Tuple
import numpy as np
from skimage import measure

from taubin import taubin_smoothing
from gaussian import restricted_gaussian
from subdivision import loop_subdivision

# File paths
number = "00038"
INPUT_PATH = "//LAPTOP-ASUS/Transfer/test/phi_" + number + ".npy"
OUTPUT_PATH = "E:/ACG-Project/test/output_remesh_" + number + ".ply"

# Mesh parameters
RESOLUTION = 256
THICKNESS = 6

# Magic numbers
SIGMA = 1.0                 # Gaussian smoothing sigma
LEVEL = 0.0                 # Level set value for surface extraction
LAMBDA = 0.50               # Taubin smoothing lambda
MU = -0.53                  # Taubin smoothing mu
ATTATCH_THRESHOLD = 0.40    # Thresholds for attaching boundary vertices, edges, faces
THRESHOLD_LIST = [1.50, 2.80, 3.60]
TAUBIN_THRESHOLD = 0.2      # Threshold for Taubin smoothing boundary vertices

UPSAMPLE = 2                # Volume upsampling factor
MAIN_ITERATIONS = 10        # Number of Taubin smoothing iterations
SUB_ITERATIONS = 0          # Number of Loop subdivision iterations
GAUSSIAN_ITERATIONS = 1     # Number of restricted Gaussian smoothing iterations


def combine_water_and_solid(level_set: np.ndarray, upsample_times: int) -> np.ndarray:
    solid_phi = np.zeros((RESOLUTION * upsample_times, RESOLUTION * upsample_times, RESOLUTION * upsample_times), dtype=np.float32)
    for i in range(RESOLUTION * upsample_times):
        for j in range(RESOLUTION * upsample_times):
            for k in range(RESOLUTION * upsample_times):
                if  i < THICKNESS * upsample_times or i >= RESOLUTION * upsample_times - THICKNESS * upsample_times or \
                    j < THICKNESS * upsample_times or j >= RESOLUTION * upsample_times - THICKNESS * upsample_times or \
                    k < THICKNESS * upsample_times or k >= RESOLUTION * upsample_times - THICKNESS * upsample_times:
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


def _cubic_kernel(x: np.ndarray, a: float = -0.5) -> np.ndarray:
    """Keys cubic convolution kernel (Catmull-Rom with a=-0.5)."""
    absx = np.abs(x)
    absx2 = absx * absx
    absx3 = absx2 * absx

    k1 = ((a + 2.0) * absx3) - ((a + 3.0) * absx2) + 1.0
    k2 = (a * absx3) - (5.0 * a * absx2) + (8.0 * a * absx) - (4.0 * a)
    return np.where(absx <= 1.0, k1, np.where(absx < 2.0, k2, 0.0))


def _cubic_resize_axis(volume: np.ndarray, factor: int, axis: int) -> np.ndarray:
    """Resize along one axis using cubic convolution interpolation."""
    if factor <= 1:
        return volume

    vol = np.moveaxis(volume, axis, 0)
    src_len = vol.shape[0]
    dst_len = src_len * factor

    # Coordinates in source space spaced evenly across the original range.
    coords = np.linspace(0.0, src_len - 1, dst_len, dtype=np.float32)
    base = np.floor(coords).astype(np.int64)
    frac = coords - base

    # Neighbor indices with clamping at boundaries.
    idxs = np.stack((base - 1, base, base + 1, base + 2), axis=0)
    idxs = np.clip(idxs, 0, src_len - 1)

    # Cubic weights for each neighbor offset.
    weights = np.stack(
        (
            _cubic_kernel(frac + 1.0),
            _cubic_kernel(frac),
            _cubic_kernel(frac - 1.0),
            _cubic_kernel(frac - 2.0),
        ),
        axis=0,
    ).astype(np.float32, copy=False)

    # Broadcast weights over trailing dimensions.
    weight_shape = (4, dst_len) + (1,) * (vol.ndim - 1)
    weights = weights.reshape(weight_shape)

    gathered = np.stack([vol[idxs[k]] for k in range(4)], axis=0)
    resized = np.sum(weights * gathered, axis=0)
    return np.moveaxis(resized, 0, axis)


def upsample(volume: np.ndarray, factor: int) -> np.ndarray:
    """Upsample a 3D volume by an integer factor using separable cubic interpolation."""
    if factor <= 1:
        return volume.astype(np.float32, copy=False)

    result = volume.astype(np.float32, copy=False)
    for axis in range(3):
        result = _cubic_resize_axis(result, factor, axis)
    return result


def run_marching_cubes(volume: np.ndarray, level: float, spacing: Sequence[float], origin: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    verts, faces, _, _ = measure.marching_cubes(volume, level=level, spacing=spacing)
    verts = verts.astype(np.float32, copy=False)
    faces = faces.astype(np.int32, copy=False)

    verts[:, 0] += origin[0]
    verts[:, 1] += origin[1]
    verts[:, 2] += origin[2]
    verts = verts[:, [2, 1, 0]]
    return verts, faces


def attach_boundary(vertices: np.ndarray, threshold: Sequence[float]) -> np.ndarray:
    threshold_face = threshold[0]
    threshold_edge = threshold[1]
    threshold_vertex = threshold[2]
    verts = np.asarray(vertices, dtype=np.float32)
    for i in range(len(verts)):
        for d in range(3):
            if abs(verts[i, d] - THICKNESS) < threshold_face:
                verts[i, d] = float(THICKNESS + 0.01)
                other_dims = [od for od in range(3) if od != d]
                for od in other_dims:
                    if abs(verts[i, od] - THICKNESS) < threshold_edge:
                        verts[i, od] = float(THICKNESS + 0.01)
                        left_dim = [ld for ld in other_dims if ld != od][0]
                        if abs(verts[i, left_dim] - THICKNESS) < threshold_vertex:
                            verts[i, left_dim] = float(THICKNESS + 0.01)
                        if abs(verts[i, left_dim] - (RESOLUTION - THICKNESS - 0.5)) < threshold_vertex:
                            verts[i, left_dim] = float(RESOLUTION - THICKNESS - 0.5 - 0.01)
                    if abs(verts[i, od] - (RESOLUTION - THICKNESS - 0.5)) < threshold_edge:
                        verts[i, od] = float(RESOLUTION - THICKNESS - 0.5 - 0.01)
                        left_dim = [ld for ld in other_dims if ld != od][0]
                        if abs(verts[i, left_dim] - THICKNESS) < threshold_vertex:
                            verts[i, left_dim] = float(THICKNESS + 0.01)
                        if abs(verts[i, left_dim] - (RESOLUTION - THICKNESS - 0.5)) < threshold_vertex:
                            verts[i, left_dim] = float(RESOLUTION - THICKNESS - 0.5 - 0.01)
            if abs(verts[i, d] - (RESOLUTION - THICKNESS - 0.5)) < threshold_face:
                verts[i, d] = float(RESOLUTION - THICKNESS - 0.5 - 0.01)
                other_dims = [od for od in range(3) if od != d]
                for od in other_dims:
                    if abs(verts[i, od] - THICKNESS) < threshold_edge:
                        verts[i, od] = float(THICKNESS + 0.01)
                        left_dim = [ld for ld in other_dims if ld != od][0]
                        if abs(verts[i, left_dim] - THICKNESS) < threshold_vertex:
                            verts[i, left_dim] = float(THICKNESS + 0.01)
                        if abs(verts[i, left_dim] - (RESOLUTION - THICKNESS - 0.5)) < threshold_vertex:
                            verts[i, left_dim] = float(RESOLUTION - THICKNESS - 0.5 - 0.01)
                    if abs(verts[i, od] - (RESOLUTION - THICKNESS - 0.5)) < threshold_edge:
                        verts[i, od] = float(RESOLUTION - THICKNESS - 0.5 - 0.01)
                        left_dim = [ld for ld in other_dims if ld != od][0]
                        if abs(verts[i, left_dim] - THICKNESS) < threshold_vertex:
                            verts[i, left_dim] = float(THICKNESS + 0.01)
                        if abs(verts[i, left_dim] - (RESOLUTION - THICKNESS - 0.5)) < threshold_vertex:
                            verts[i, left_dim] = float(RESOLUTION - THICKNESS - 0.5 - 0.01)
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
    refined = upsample(level_set, UPSAMPLE)
    combined_volume = combine_water_and_solid(refined, UPSAMPLE)
    sub_volume, offset = extract_active_region(combined_volume)
    # refined = upsample_volume(sub_volume, UPSAMPLE)
    spacing = tuple(1.0 / UPSAMPLE for _ in range(3))
    origin = (float(offset[0]) / UPSAMPLE, float(offset[1]) / UPSAMPLE, float(offset[2]) / UPSAMPLE)
    vertices, faces = run_marching_cubes(sub_volume, level=LEVEL, spacing=spacing, origin=origin)
    for _ in range(MAIN_ITERATIONS):
        vertices, faces = taubin_smoothing(vertices, faces, RESOLUTION, THICKNESS, LAMBDA, MU, TAUBIN_THRESHOLD)
    # Final check to deal with bubbles
    # vertices = attach_boundary(vertices, ATTATCH_THRESHOLD)
    vertices = attach_boundary(vertices, ATTATCH_THRESHOLD * np.array(THRESHOLD_LIST) / UPSAMPLE)
    vertices, faces = loop_subdivision(vertices, faces, SUB_ITERATIONS, RESOLUTION, THICKNESS)
    write_ply(Path(output_path), vertices, faces)


def export(input_path: str, output_path: str) -> None:
    input_path = Path(input_path)
    level_set = np.load(input_path).astype(np.float32, copy=False)
    level_set_to_mesh(level_set, Path(output_path))


if __name__ == "__main__":
    export(INPUT_PATH, OUTPUT_PATH)