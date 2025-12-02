"""Curvature-weighted Laplacian smoothing utilities."""


from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


def _cotangent(u: np.ndarray, v: np.ndarray) -> float:
	"""Return cot(angle) formed by vectors ``u`` and ``v``."""
	cross_norm = np.linalg.norm(np.cross(u, v))
	if cross_norm < 1e-12:
		return 0.0
	return float(np.dot(u, v) / cross_norm)


def curvature_laplacian_smoothing(vertices: np.ndarray, faces: np.ndarray, smoothing_step: float) -> Tuple[np.ndarray, np.ndarray]:
	"""Apply a single curvature-based Laplacian smoothing pass."""
	if vertices.ndim != 2 or vertices.shape[1] != 3:
		raise ValueError("vertices must be shaped (N, 3)")
	if faces.ndim != 2 or faces.shape[1] != 3:
		raise ValueError("faces must be shaped (M, 3)")

	if len(vertices) == 0:
		return vertices.copy(), faces.copy()

	verts = np.asarray(vertices, dtype=np.float64)
	tris = np.asarray(faces, dtype=np.int32)

	adjacency: List[Dict[int, float]] = [dict() for _ in range(len(verts))]

	def accumulate(i: int, j: int, weight: float) -> None:
		if i == j or weight == 0.0:
			return
		adjacency[i][j] = adjacency[i].get(j, 0.0) + weight

	for face in tris:
		i0, i1, i2 = map(int, face)
		if min(i0, i1, i2) < 0 or max(i0, i1, i2) >= len(verts):
			continue
		if i0 == i1 or i1 == i2 or i0 == i2:
			continue
		p0, p1, p2 = verts[[i0, i1, i2]]
		w0 = 0.5 * _cotangent(p1 - p0, p2 - p0)
		w1 = 0.5 * _cotangent(p2 - p1, p0 - p1)
		w2 = 0.5 * _cotangent(p0 - p2, p1 - p2)
		accumulate(i1, i2, w0)
		accumulate(i2, i1, w0)
		accumulate(i2, i0, w1)
		accumulate(i0, i2, w1)
		accumulate(i0, i1, w2)
		accumulate(i1, i0, w2)

	laplacian = np.zeros_like(verts)
	for idx, neighbors in enumerate(adjacency):
		if not neighbors:
			continue
		displacement = np.zeros(3, dtype=np.float64)
		weight_sum = 0.0
		for nbr, weight in neighbors.items():
			displacement += weight * (verts[nbr] - verts[idx])
			weight_sum += weight
		if abs(weight_sum) > 1e-12:
			laplacian[idx] = displacement / weight_sum
		else:
			laplacian[idx] = displacement

	smoothed = verts + smoothing_step * laplacian
	return smoothed.astype(vertices.dtype, copy=False), tris.copy()