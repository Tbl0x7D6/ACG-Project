"""Mesh downsampling via QEM."""


from __future__ import annotations
from typing import Set, Tuple
import numpy as np


# Parameters
TARGET_RATIO = 0.5
MIN_VERTICES = 4


def downsampling(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""Downsample a triangle mesh using Quadric Error edge collapses."""
	if vertices.ndim != 2 or vertices.shape[1] != 3:
		raise ValueError("vertices must be shaped (N, 3)")
	if faces.ndim != 2 or faces.shape[1] != 3:
		raise ValueError("faces must be shaped (M, 3)")

	if len(vertices) == 0 or len(faces) == 0:
		return vertices.copy(), faces.copy()

	verts = np.asarray(vertices, dtype=np.float64).copy()
	tris = np.asarray(faces, dtype=np.int32).copy()
	valid_vertices = np.ones(len(verts), dtype=bool)
	valid_faces = np.ones(len(tris), dtype=bool)
	target_count = max(int(len(verts) * TARGET_RATIO), MIN_VERTICES)

	def compute_quadrics() -> np.ndarray:
		quadrics = np.zeros((len(verts), 4, 4), dtype=np.float64)
		for idx, is_valid in enumerate(valid_faces):
			if not is_valid:
				continue
			a, b, c = tris[idx]
			if a < 0 or b < 0 or c < 0:
				continue
			va, vb, vc = verts[[a, b, c]]
			normal = np.cross(vb - va, vc - va)
			norm = np.linalg.norm(normal)
			if norm < 1e-12:
				continue
			normal /= norm
			d = -float(np.dot(normal, va))
			k = np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)
			q = np.outer(k, k)
			quadrics[a] += q
			quadrics[b] += q
			quadrics[c] += q
		return quadrics

	def collect_edges() -> Set[Tuple[int, int]]:
		edges: Set[Tuple[int, int]] = set()
		for idx, is_valid in enumerate(valid_faces):
			if not is_valid:
				continue
			a, b, c = tris[idx]
			if a < 0 or b < 0 or c < 0:
				continue
			edges.add(tuple(sorted((a, b))))
			edges.add(tuple(sorted((b, c))))
			edges.add(tuple(sorted((c, a))))
		return edges

	def optimal_position(qsum: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
		mat = qsum[:3, :3]
		vec = -qsum[:3, 3]
		cond = np.linalg.cond(mat)
		if np.isfinite(cond) and cond < 1e12:
			return np.linalg.solve(mat, vec)
		return 0.5 * (v1 + v2)

	def edge_error(qsum: np.ndarray, pos: np.ndarray) -> float:
		v = np.array([pos[0], pos[1], pos[2], 1.0], dtype=np.float64)
		return float(v @ qsum @ v)

	max_iters = len(verts) * 4
	iteration = 0

	while np.count_nonzero(valid_vertices) > target_count and iteration < max_iters:
		iteration += 1
		quadrics = compute_quadrics()
		edges = collect_edges()
		best_edge: Tuple[int, int] | None = None
		best_pos: np.ndarray | None = None
		best_cost = np.inf
		for u, v in edges:
			if u == v:
				continue
			if not (valid_vertices[u] and valid_vertices[v]):
				continue
			dest, src = (u, v) if u < v else (v, u)
			qsum = quadrics[dest] + quadrics[src]
			pos = optimal_position(qsum, verts[dest], verts[src])
			cost = edge_error(qsum, pos)
			if not np.isfinite(cost):
				continue
			if cost < best_cost:
				best_cost = cost
				best_edge = (dest, src)
				best_pos = pos
		if best_edge is None or best_pos is None:
			break

		dest, src = best_edge
		verts[dest] = best_pos
		valid_vertices[src] = False
		for idx, is_valid in enumerate(valid_faces):
			if not is_valid:
				continue
			face = tris[idx]
			replaced = False
			for j in range(3):
				if face[j] == src:
					tris[idx, j] = dest
					replaced = True
			if not replaced:
				continue
			a, b, c = tris[idx]
			if a == b or b == c or a == c:
				valid_faces[idx] = False
				tris[idx] = -1

	valid_index = -np.ones(len(verts), dtype=np.int32)
	new_vertices = []
	for idx, is_valid in enumerate(valid_vertices):
		if not is_valid:
			continue
		valid_index[idx] = len(new_vertices)
		new_vertices.append(verts[idx])

	new_faces = []
	for idx, is_valid in enumerate(valid_faces):
		if not is_valid:
			continue
		a, b, c = tris[idx]
		if a < 0 or b < 0 or c < 0:
			continue
		mapped = [valid_index[a], valid_index[b], valid_index[c]]
		if -1 in mapped:
			continue
		if len({mapped[0], mapped[1], mapped[2]}) < 3:
			continue
		new_faces.append(mapped)

	if not new_vertices:
		return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

	if not new_faces:
		return np.asarray(new_vertices, dtype=np.float32), np.empty((0, 3), dtype=np.int32)

	return np.asarray(new_vertices, dtype=np.float32), np.asarray(new_faces, dtype=np.int32)