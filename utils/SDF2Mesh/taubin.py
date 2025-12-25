"""Taubin smoothing implementation."""

from typing import List, Tuple
import numpy as np


def _build_adjacency(faces: np.ndarray, vertex_count: int, boundary_vertices: np.ndarray) -> List[List[int]]:
	"""Return adjacency lists for each vertex in the mesh."""
	adjacency: List[set[int]] = [set() for _ in range(vertex_count)]
	for tri in faces:
		a, b, c = map(int, tri)
		if not boundary_vertices[a]:
			adjacency[a].update((b, c))
		if not boundary_vertices[b]:
			adjacency[b].update((a, c))
		if not boundary_vertices[c]:
			adjacency[c].update((a, b))
	return [list(neighbors) for neighbors in adjacency]


def _uniform_laplacian(vertices: np.ndarray, adjacency: List[List[int]]) -> np.ndarray:
	"""Compute uniform Laplacian vectors for all vertices."""
	laplacian = np.zeros_like(vertices)
	for idx, neighbors in enumerate(adjacency):
		if not neighbors:
			continue
		neighbor_positions = vertices[neighbors]
		laplacian[idx] = neighbor_positions.mean(axis=0) - vertices[idx]
	return laplacian


def _restricted_taubin_smoothing(vertices: np.ndarray, faces: np.ndarray, boundary_vertices: np.ndarray, Lambda: float, mu: float) -> Tuple[np.ndarray, np.ndarray]:
	"""Apply one iteration of Taubin smoothing and return updated mesh data."""
	if vertices.ndim != 2 or vertices.shape[1] != 3:
		raise ValueError("vertices must be of shape (N, 3)")
	if faces.ndim != 2 or faces.shape[1] != 3:
		raise ValueError("faces must be of shape (M, 3)")

	adjacency = _build_adjacency(faces, len(vertices), boundary_vertices)

	laplacian_lambda = _uniform_laplacian(vertices, adjacency)
	intermediate_vertices = vertices + Lambda * laplacian_lambda

	laplacian_mu = _uniform_laplacian(intermediate_vertices, adjacency)
	smoothed_vertices = intermediate_vertices + mu * laplacian_mu

	return smoothed_vertices, faces.copy()


def taubin_smoothing(vertices: np.ndarray, faces: np.ndarray, resolution: int, thickness: int, Lambda: float, mu: float, taubin_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(vertices, dtype=np.float32)
    boundary_vertices = (
          (np.abs(coords[:, 0] - (thickness)) < taubin_threshold)
        | (np.abs(coords[:, 0] - (resolution - thickness - 0.5)) < taubin_threshold)
        | (np.abs(coords[:, 1] - (thickness)) < taubin_threshold)
        | (np.abs(coords[:, 1] - (resolution - thickness - 0.5)) < taubin_threshold)
        | (np.abs(coords[:, 2] - (thickness)) < taubin_threshold)
        | (np.abs(coords[:, 2] - (resolution - thickness - 0.5)) < taubin_threshold)
    )
    smoothed_vertices, smoothed_faces = _restricted_taubin_smoothing(vertices, faces, boundary_vertices, Lambda, mu)
    return smoothed_vertices, smoothed_faces