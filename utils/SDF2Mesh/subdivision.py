"""Loop subdivision implementation for triangle meshes."""

from typing import Tuple
import numpy as np


def _restricted_subdivision(verts: np.ndarray, tris: np.ndarray, boundary_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    adjacency = [set() for _ in range(len(verts))]
    for a, b, c in tris:
        if not boundary_vertices[a]:
            adjacency[a].update((b, c))
        if not boundary_vertices[b]:
            adjacency[b].update((a, c))
        if not boundary_vertices[c]:
            adjacency[c].update((a, b))

    even = np.empty_like(verts)
    for idx, neighbors in enumerate(adjacency):
        valence = len(neighbors)
        if valence < 3:
            even[idx] = verts[idx]
            continue

        if valence == 3:
            beta = 3.0 / 16.0
        else:
            cos_term = np.cos(2.0 * np.pi / valence)
            beta = (1.0 / valence) * (5.0 / 8.0 - (0.375 + 0.25 * cos_term) ** 2)

        neighbor_sum = np.sum(verts[list(neighbors)], axis=0)
        even[idx] = (1.0 - valence * beta) * verts[idx] + beta * neighbor_sum

    edge_info = {}
    new_vertices = even.tolist()

    for a, b, c in tris:
        edges = ((a, b, c), (b, c, a), (c, a, b))
        for u, v, w in edges:
            edge = (u, v) if u < v else (v, u)
            data = edge_info.get(edge)
            if data is None:
                edge_info[edge] = {"opp": [w], "index": None}
            else:
                data["opp"].append(w)

    for edge, data in edge_info.items():
        u, v = edge
        opp = data["opp"]
        if len(opp) == 2:
            new_pos = 0.375 * (verts[u] + verts[v]) + 0.125 * (verts[opp[0]] + verts[opp[1]])
        else:
            new_pos = 0.5 * (verts[u] + verts[v])

        data["index"] = len(new_vertices)
        new_vertices.append(new_pos.tolist())

    new_faces = []
    for a, b, c in tris:
        ab = edge_info[(a, b) if a < b else (b, a)]["index"]
        bc = edge_info[(b, c) if b < c else (c, b)]["index"]
        ca = edge_info[(c, a) if c < a else (a, c)]["index"]
        new_faces.extend(
            (
                [a, ab, ca],
                [b, bc, ab],
                [c, ca, bc],
                [ab, bc, ca],
            )
        )

    verts = np.asarray(new_vertices, dtype=np.float32)
    tris = np.asarray(new_faces, dtype=np.int32)

    return verts, tris

def loop_subdivision(vertices: np.ndarray, faces: np.ndarray, iterations: int, resolution: int, thickness: int) -> Tuple[np.ndarray, np.ndarray]:

    if iterations <= 0:
        return vertices, faces

    verts = vertices.astype(np.float32, copy=True)
    tris = faces.astype(np.int32, copy=True)

    coords = np.asarray(vertices, dtype=np.float32)
    boundary_vertices = (
          (np.abs(coords[:, 0] - (thickness - 1)) < 0.01)
        | (np.abs(coords[:, 0] - (resolution - thickness)) < 0.01)
        | (np.abs(coords[:, 1] - (thickness - 1)) < 0.01)
        | (np.abs(coords[:, 1] - (resolution - thickness)) < 0.01)
        | (np.abs(coords[:, 2] - (thickness - 1)) < 0.01)
        | (np.abs(coords[:, 2] - (resolution - thickness)) < 0.01)
    )

    for _ in range(iterations):
        verts, tris = _restricted_subdivision(verts, tris, boundary_vertices)

    return verts, tris