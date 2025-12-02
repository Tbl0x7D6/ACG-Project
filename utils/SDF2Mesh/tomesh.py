from pathlib import Path
import numpy as np
from skimage import measure
from skimage.filters import gaussian
from taubin import taubin_smoothing
from laplacian import curvature_laplacian_smoothing
from subdivision import loop_subdivision
# from marching_cubes import marching_cubes
# from downsampling import downsampling


# File paths
INPUT_PATH = "./tomesh/phi_00060.npy"
OUTPUT_PATH = "./tomesh/output_060.ply"

# Mesh parameters
RESOLUTION = 256
THICKNESS = 6

# Magic numbers
THICKNESS_THRESHOLD = 1.5   # Boundary attachment threshold
ITERATIONS = 1              # Loop subdivision iterations
MAIN_ITERATIONS = 10        # Main smoothing iterations
LAMBDA = 0.5                # Taubin smoothing parameter
MU = -0.53                  # Taubin smoothing parameter
SMOOTHING_STEP = 0.3        # Curvature Laplacian smoothing step size


def load_npy_volume(path: Path) -> np.ndarray:
    """Load volume from a .npy file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing .npy file: {path}")
    data = np.load(path)
    return data.astype(np.float32, copy=False)


def attatch_boundary(vertices: np.ndarray, threshold: float) -> np.ndarray:
    """Attatch boundary vertices to the sides of the volume cube."""
    verts = np.asarray(vertices, dtype=np.float32)
    for i in range(len(verts)):
        for d in range(3):
            if abs(verts[i, d] - THICKNESS) < threshold:
                verts[i, d] = float(THICKNESS)
            if abs(verts[i, d] - (RESOLUTION - THICKNESS)) < threshold:
                verts[i, d] = float(RESOLUTION - THICKNESS)
    return verts.astype(vertices.dtype, copy=False)


def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Form ASCII PLY file from mesh data."""
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


def export_mesh(volume: np.ndarray, output_path: str) -> None:
    """Export mesh from volume data to PLY file."""
    output_path = Path(output_path)
    smoothed_volume = gaussian(volume, sigma=1.0)
    verts, faces, _, _ = measure.marching_cubes(smoothed_volume.astype(np.float32), level=0.0, step_size=2)
    # Remember to adjust axis order !!!
    verts = verts[:, [2, 1, 0]]
    vertices = verts.astype(np.float32, copy=False)
    faces = faces.astype(np.int32, copy=False)
    for i in range(MAIN_ITERATIONS):
        vertices, faces = curvature_laplacian_smoothing(vertices, faces, SMOOTHING_STEP)
        vertices, faces = taubin_smoothing(vertices, faces, LAMBDA, MU)
        vertices, faces = taubin_smoothing(vertices, faces, LAMBDA, MU)
        vertices = attatch_boundary(vertices, THICKNESS_THRESHOLD)
        vertices, faces = taubin_smoothing(vertices, faces, LAMBDA, MU)
        vertices, faces = taubin_smoothing(vertices, faces, LAMBDA, MU)
        print((i + 1) / 10.0 * 100.0, "% done")
    vertices, faces = loop_subdivision(vertices, faces, ITERATIONS)
    vertices, faces = taubin_smoothing(vertices, faces, LAMBDA, MU)
    vertices = attatch_boundary(vertices, THICKNESS_THRESHOLD)
    vertices, faces = taubin_smoothing(vertices, faces, LAMBDA, MU)
    vertices, faces = taubin_smoothing(vertices, faces, LAMBDA, MU)
    print(f"Final mesh: {len(vertices)} vertices, {len(faces)} faces.")
    write_ply(output_path, vertices, faces)


def export(input_path: str, output_path: str) -> None:
    """Entry point for exporting mesh."""
    volume = load_npy_volume(Path(input_path))
    export_mesh(volume, output_path)

if __name__ == "__main__":
    export(INPUT_PATH, OUTPUT_PATH)