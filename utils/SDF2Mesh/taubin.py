"""Taubin smoothing implementation."""


from typing import List, Tuple
import numpy as np


def _build_adjacency(faces: np.ndarray, vertex_count: int) -> List[List[int]]:
	"""Return adjacency lists for each vertex in the mesh."""
	adjacency: List[set[int]] = [set() for _ in range(vertex_count)]
	for tri in faces:
		a, b, c = map(int, tri)
		adjacency[a].update((b, c))
		adjacency[b].update((a, c))
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


def taubin_smoothing(vertices: np.ndarray, faces: np.ndarray, Lambda: float, mu: float) -> Tuple[np.ndarray, np.ndarray]:
	"""Apply one iteration of Taubin smoothing and return updated mesh data."""
	if vertices.ndim != 2 or vertices.shape[1] != 3:
		raise ValueError("vertices must be of shape (N, 3)")
	if faces.ndim != 2 or faces.shape[1] != 3:
		raise ValueError("faces must be of shape (M, 3)")

	adjacency = _build_adjacency(faces, len(vertices))

	laplacian_lambda = _uniform_laplacian(vertices, adjacency)
	intermediate_vertices = vertices + Lambda * laplacian_lambda

	laplacian_mu = _uniform_laplacian(intermediate_vertices, adjacency)
	smoothed_vertices = intermediate_vertices + mu * laplacian_mu

	return smoothed_vertices, faces.copy()