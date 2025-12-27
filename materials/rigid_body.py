import taichi as ti
import numpy as np
from . import load_mesh


@ti.func
def quat_mul(v1, v2):
    return ti.Vector([
        v1.x * v2.x - v1.y * v2.y - v1.z * v2.z - v1.w * v2.w,
        v1.x * v2.y + v2.x * v1.y + v1.z * v2.w - v1.w * v2.z,
        v1.x * v2.z + v2.x * v1.z + v1.w * v2.y - v1.y * v2.w,
        v1.x * v2.w + v2.x * v1.w + v1.y * v2.z - v1.z * v2.y
    ])

@ti.func
def quat_conj(q):
    return ti.Vector([q.x, -q.y, -q.z, -q.w])

@ti.func
def rotate(q, v):
    q_v = ti.Vector([0.0, v.x, v.y, v.z])
    return quat_mul(quat_mul(q, q_v), quat_conj(q)).yzw

@ti.func
def rotate_inv(q, v):
    return rotate(quat_conj(q), v)

@ti.func
def dist_aabb(p, bmin, bmax):
    dx = ti.max(ti.max(bmin.x - p.x, 0.0), p.x - bmax.x)
    dy = ti.max(ti.max(bmin.y - p.y, 0.0), p.y - bmax.y)
    dz = ti.max(ti.max(bmin.z - p.z, 0.0), p.z - bmax.z)
    return ti.sqrt(dx * dx + dy * dy + dz * dz)

@ti.func
def dist_triangle(p, a, b, c):
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = ab.dot(ap)
    d2 = ac.dot(ap)
    bp = p - b
    d3 = ab.dot(bp)
    d4 = ac.dot(bp)
    vc = d1 * d4 - d3 * d2
    cp = p - c
    d5 = ab.dot(cp)
    d6 = ac.dot(cp)
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    d = 0.0
    if d1 <= 0.0 and d2 <= 0.0:
        d = (p - a).norm()
    elif d3 >= 0.0 and d4 <= d3:
        d = (p - b).norm()
    elif vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        proj = a + v * ab
        d = (p - proj).norm()
    elif d6 >= 0.0 and d5 <= d6:
        d = (p - c).norm()
    elif vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        proj = a + w * ac
        d = (p - proj).norm()
    elif va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        proj = b + w * (c - b)
        d = (p - proj).norm()
    else:
        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        proj = a + ab * v + ac * w
        d = (p - proj).norm()
    return d

@ti.data_oriented
class RigidBody:

    def __init__(self, mesh_file: str, scale: float, mass: float, x=ti.Vector([0.0, 0.0, 0.0]), v=ti.Vector([0.0, 0.0, 0.0]),
                 omega=ti.Vector([0.0, 0.0, 0.0]), q=ti.Vector([1.0, 0.0, 0.0, 0.0]), fixed: bool=False, use_sdf: bool=True, sdf_resolution: int=128):

        _vertices, _faces = load_mesh.loader(mesh_file, scale)

        self.faces = ti.Vector.field(3, dtype=int, shape=len(_faces))
        self.vertices = ti.Vector.field(3, dtype=float, shape=len(_vertices))

        self.faces.from_numpy(np.array(_faces, dtype=np.int32))
        self.vertices.from_numpy(np.array(_vertices, dtype=np.float32))

        self.m = mass
        self.m_inv = 1.0 / mass
        self._centralize()
        self.inertia = self._inertia()
        self.inertia_inv = self.inertia.inverse()
        self.fixed = fixed

        self._build_bvh()

        self.use_sdf = use_sdf
        if use_sdf:
            self.sdf_resolution = sdf_resolution
            self.sdf = ti.field(dtype=float, shape=(sdf_resolution, sdf_resolution, sdf_resolution))
            self.sdf_1 = ti.field(dtype=float, shape=(sdf_resolution, sdf_resolution, sdf_resolution))
            self.sdf_2 = ti.field(dtype=float, shape=(sdf_resolution, sdf_resolution, sdf_resolution))
            self._build_sdf()

        self.collision = self.collision_sdf if use_sdf else self.collision_bvh

        self.x = ti.Vector.field(3, dtype=float, shape=())
        self.v = ti.Vector.field(3, dtype=float, shape=())
        self.omega = ti.Vector.field(3, dtype=float, shape=())
        self.q = ti.Vector.field(4, dtype=float, shape=())
        self.x[None] = x
        self.v[None] = v
        self.omega[None] = omega
        self.q[None] = q

        self.force = ti.Vector.field(3, dtype=float, shape=())
        self.torque = ti.Vector.field(3, dtype=float, shape=())
        self.force[None] = ti.Vector([0.0, 0.0, 0.0])
        self.torque[None] = ti.Vector([0.0, 0.0, 0.0])

        self._rendered_indices = ti.field(dtype=int, shape=len(_faces) * 3)
        self._rendered_vertices = ti.Vector.field(3, dtype=float, shape=len(_vertices))
        self._rendered_indices.from_numpy(self.faces.to_numpy().flatten())

        self._update_position()

    def _build_bvh(self):
        n_faces = self.faces.shape[0]
        n_nodes = 2 * n_faces - 1

        self.bvh_bmin = ti.Vector.field(3, dtype=float, shape=n_nodes)
        self.bvh_bmax = ti.Vector.field(3, dtype=float, shape=n_nodes)
        self.bvh_left = ti.field(dtype=int, shape=n_nodes)
        self.bvh_right = ti.field(dtype=int, shape=n_nodes)
        self.bvh_face_id = ti.field(dtype=int, shape=n_nodes)

        face_centers = []
        face_bmin = []
        face_bmax = []
        vertices_np = self.vertices.to_numpy()
        faces_np = self.faces.to_numpy()

        for i in range(n_faces):
            f = faces_np[i]
            v0, v1, v2 = vertices_np[f[0]], vertices_np[f[1]], vertices_np[f[2]]
            bmin = np.minimum(np.minimum(v0, v1), v2)
            bmax = np.maximum(np.maximum(v0, v1), v2)
            center = (v0 + v1 + v2) / 3.0
            face_centers.append(center)
            face_bmin.append(bmin)
            face_bmax.append(bmax)

        face_centers = np.array(face_centers)
        face_bmin = np.array(face_bmin)
        face_bmax = np.array(face_bmax)

        bvh_bmin_np = np.zeros((n_nodes, 3), dtype=np.float32)
        bvh_bmax_np = np.zeros((n_nodes, 3), dtype=np.float32)
        bvh_left_np = np.full(n_nodes, -1, dtype=np.int32)
        bvh_right_np = np.full(n_nodes, -1, dtype=np.int32)
        bvh_face_id_np = np.full(n_nodes, -1, dtype=np.int32)

        node_counter = [0]

        def _recursive_build(face_indices):
            node_id = node_counter[0]
            node_counter[0] += 1

            if len(face_indices) == 1:
                fid = face_indices[0]
                bvh_bmin_np[node_id] = face_bmin[fid]
                bvh_bmax_np[node_id] = face_bmax[fid]
                bvh_face_id_np[node_id] = fid
                return node_id

            current_bmin = face_bmin[face_indices].min(axis=0)
            current_bmax = face_bmax[face_indices].max(axis=0)
            bvh_bmin_np[node_id] = current_bmin
            bvh_bmax_np[node_id] = current_bmax

            extent = current_bmax - current_bmin
            axis = np.argmax(extent)

            centers = face_centers[face_indices]
            sorted_indices = face_indices[np.argsort(centers[:, axis])]

            mid = len(sorted_indices) // 2
            left_indices = sorted_indices[:mid]
            right_indices = sorted_indices[mid:]

            left_child = _recursive_build(left_indices)
            right_child = _recursive_build(right_indices)

            bvh_left_np[node_id] = left_child
            bvh_right_np[node_id] = right_child

            return node_id

        _recursive_build(np.arange(n_faces))

        self.bvh_bmin.from_numpy(bvh_bmin_np)
        self.bvh_bmax.from_numpy(bvh_bmax_np)
        self.bvh_left.from_numpy(bvh_left_np)
        self.bvh_right.from_numpy(bvh_right_np)
        self.bvh_face_id.from_numpy(bvh_face_id_np)

    @ti.kernel
    def _reinit_iter(self):
        # Godunov Hamiltonian
        dx = self.extent.x / self.sdf_resolution
        dtau = 0.5 * dx
        res = self.sdf_resolution

        for i, j, k in ti.ndrange(res, res, res):
            if i == 0 or i == res - 1 or j == 0 or j == res - 1 or k == 0 or k == res - 1:
                if i == 0:
                    self.sdf_2[i, j, k] = self.sdf_1[i + 1, j, k]
                elif i == res - 1:
                    self.sdf_2[i, j, k] = self.sdf_1[i - 1, j, k]
                elif j == 0:
                    self.sdf_2[i, j, k] = self.sdf_1[i, j + 1, k]
                elif j == res - 1:
                    self.sdf_2[i, j, k] = self.sdf_1[i, j - 1, k]
                elif k == 0:
                    self.sdf_2[i, j, k] = self.sdf_1[i, j, k + 1]
                else:
                    self.sdf_2[i, j, k] = self.sdf_1[i, j, k - 1]
            else:
                phi_0 = self.sdf[i, j, k]
                s = phi_0 / ti.sqrt(phi_0 * phi_0 + dx * dx)

                dx_minus = (self.sdf_1[i, j, k] - self.sdf_1[i - 1, j, k]) / dx
                dx_plus = (self.sdf_1[i + 1, j, k] - self.sdf_1[i, j, k]) / dx
                dy_minus = (self.sdf_1[i, j, k] - self.sdf_1[i, j - 1, k]) / dx
                dy_plus = (self.sdf_1[i, j + 1, k] - self.sdf_1[i, j, k]) / dx
                dz_minus = (self.sdf_1[i, j, k] - self.sdf_1[i, j, k - 1]) / dx
                dz_plus = (self.sdf_1[i, j, k + 1] - self.sdf_1[i, j, k]) / dx

                grad_sq_x = 0.0
                grad_sq_y = 0.0
                grad_sq_z = 0.0

                if phi_0 > 0:
                    grad_sq_x = ti.max(ti.max(dx_minus, 0.0) ** 2, ti.min(dx_plus, 0.0) ** 2)
                    grad_sq_y = ti.max(ti.max(dy_minus, 0.0) ** 2, ti.min(dy_plus, 0.0) ** 2)
                    grad_sq_z = ti.max(ti.max(dz_minus, 0.0) ** 2, ti.min(dz_plus, 0.0) ** 2)
                else:
                    grad_sq_x = ti.max(ti.min(dx_minus, 0.0) ** 2, ti.max(dx_plus, 0.0) ** 2)
                    grad_sq_y = ti.max(ti.min(dy_minus, 0.0) ** 2, ti.max(dy_plus, 0.0) ** 2)
                    grad_sq_z = ti.max(ti.min(dz_minus, 0.0) ** 2, ti.max(dz_plus, 0.0) ** 2)

                grad_norm = ti.sqrt(grad_sq_x + grad_sq_y + grad_sq_z)
                self.sdf_2[i, j, k] = self.sdf_1[i, j, k] - dtau * s * (grad_norm - 1.0)

        for i, j, k in ti.ndrange(res, res, res):
            self.sdf_1[i, j, k] = self.sdf_2[i, j, k]

    def _reinit_sdf(self, n_iters: int = 50):
        self.sdf_1.copy_from(self.sdf)
        for _ in range(n_iters):
            self._reinit_iter()
        self.sdf.copy_from(self.sdf_1)

    @ti.kernel
    def _init_sdf(self):
        res = self.sdf_resolution
        for i, j, k in ti.ndrange(res, res, res):
            t = ti.Vector([
                (i + 0.5) / res,
                (j + 0.5) / res,
                (k + 0.5) / res
            ])
            p_local = self.bmin + t * self.extent

            # Query distance using BVH
            min_dist, closest = self._query_dist_bvh(p_local)
            self.sdf[i, j, k] = min_dist

    @ti.kernel
    def _mark_sdf_sign(self):
        # Mark sign using ray casting
        res = self.sdf_resolution
        for i, j, k in ti.ndrange(res, res, res):
            t = ti.Vector([
                (i + 0.5) / res,
                (j + 0.5) / res,
                (k + 0.5) / res
            ])
            p_local = self.bmin + t * self.extent

            dir = ti.Vector([1.0, 0.0, 0.0])
            count = 0
            for f in ti.ndrange(self.faces.shape[0]):
                A = self.vertices[self.faces[f][0]]
                B = self.vertices[self.faces[f][1]]
                C = self.vertices[self.faces[f][2]]

                # Moller-Trumbore intersection
                edge1 = B - A
                edge2 = C - A
                h = dir.cross(edge2)
                a = edge1.dot(h)
                if -1e-6 < a < 1e-6:
                    continue
                f_inv = 1.0 / a
                s = p_local - A
                u = f_inv * s.dot(h)
                if u < 0.0 or u > 1.0:
                    continue
                q = s.cross(edge1)
                v = f_inv * dir.dot(q)
                if v < 0.0 or u + v > 1.0:
                    continue
                t_hit = f_inv * edge2.dot(q)
                if t_hit > 1e-6:
                    count += 1
            if count % 2 == 1:
                self.sdf[i, j, k] = -self.sdf[i, j, k]

    def _build_sdf(self):
        # aabb of the mesh
        vertices_np = self.vertices.to_numpy()
        bmin = np.min(vertices_np, axis=0)
        bmax = np.max(vertices_np, axis=0)
        bmin -= 0.1 * (bmax - bmin)
        bmax += 0.1 * (bmax - bmin)
        self.bmin = ti.Vector(bmin)
        self.bmax = ti.Vector(bmax)
        self.extent = self.bmax - self.bmin
        self._init_sdf()
        self._mark_sdf_sign()
        self._reinit_sdf()

    @ti.func
    def interpolate_sdf(self, p: ti.types.vector(3, float)) -> float:
        p_local = rotate_inv(self.q[None], p - self.x[None])
        p_grid = (p_local - self.bmin) / self.extent * self.sdf_resolution - ti.Vector([0.5, 0.5, 0.5])

        res = self.sdf_resolution
        i = ti.cast(ti.floor(p_grid.x), int)
        j = ti.cast(ti.floor(p_grid.y), int)
        k = ti.cast(ti.floor(p_grid.z), int)

        dist = 0.0

        if i < 0 or i >= res - 1 or j < 0 or j >= res - 1 or k < 0 or k >= res - 1:
            dist = 1e6
        else:
            fx = p_grid.x - i
            fy = p_grid.y - j
            fz = p_grid.z - k

            c000 = self.sdf[i, j, k]
            c100 = self.sdf[i + 1, j, k]
            c010 = self.sdf[i, j + 1, k]
            c110 = self.sdf[i + 1, j + 1, k]
            c001 = self.sdf[i, j, k + 1]
            c101 = self.sdf[i + 1, j, k + 1]
            c011 = self.sdf[i, j + 1, k + 1]
            c111 = self.sdf[i + 1, j + 1, k + 1]

            c00 = c000 * (1 - fx) + c100 * fx
            c01 = c001 * (1 - fx) + c101 * fx
            c10 = c010 * (1 - fx) + c110 * fx
            c11 = c011 * (1 - fx) + c111 * fx

            c0 = c00 * (1 - fy) + c10 * fy
            c1 = c01 * (1 - fy) + c11 * fy

            dist = c0 * (1 - fz) + c1 * fz

        return dist

    @ti.func
    def interpolate_sdf_gradient(self, p: ti.types.vector(3, float)) -> ti.types.vector(3, float):
        dx = self.extent.x / self.sdf_resolution * 0.5
        grad = ti.Vector([
            self.interpolate_sdf(p + ti.Vector([dx, 0.0, 0.0])) - self.interpolate_sdf(p - ti.Vector([dx, 0.0, 0.0])),
            self.interpolate_sdf(p + ti.Vector([0.0, dx, 0.0])) - self.interpolate_sdf(p - ti.Vector([0.0, dx, 0.0])),
            self.interpolate_sdf(p + ti.Vector([0.0, 0.0, dx])) - self.interpolate_sdf(p - ti.Vector([0.0, 0.0, dx]))
        ]) / (2.0 * dx)
        return grad.normalized()

    @ti.kernel
    def _centralize(self):
        cm = ti.Vector.zero(float, 3)
        area = 0.0
        for i in ti.grouped(self.faces):
            A = self.vertices[self.faces[i][0]]
            B = self.vertices[self.faces[i][1]]
            C = self.vertices[self.faces[i][2]]
            da = (B - A).cross(C - A).norm()
            cm += (A + B + C) * da / 3.0
            area += da
        cm /= area
        for i in ti.grouped(self.vertices):
            self.vertices[i] -= cm
        # print("Center of Mass:", cm)

    @ti.kernel
    def _inertia(self) -> ti.types.matrix(3, 3, float):
        I = ti.Matrix.zero(float, 3, 3)
        area = 0.0
        for k in ti.grouped(self.faces):
            A = self.vertices[self.faces[k][0]]
            B = self.vertices[self.faces[k][1]]
            C = self.vertices[self.faces[k][2]]
            da = (B - A).cross(C - A).norm()
            area += da
            center = (A + B + C) / 3.0
            center_square = center.dot(center)
            for i in range(3):
                for j in range(3):
                    if i == j:
                        I[i, j] += (center_square - center[i] ** 2) * da
                    else:
                        I[i, j] -= center[i] * center[j] * da
        return I * self.m / area

    @ti.kernel
    def _update_position(self):
        for i in ti.grouped(self.vertices):
            rotated_pos = rotate(self.q[None], self.vertices[i])
            self._rendered_vertices[i] = rotated_pos + self.x[None]

    @ti.func
    def substep(self, dt: float):
        if not self.fixed:
            self.q[None] += dt * 0.5 * quat_mul(
                self.q[None],
                ti.Vector([
                    0.0, self.omega[None].x, self.omega[None].y, self.omega[None].z
                ])
            )
            self.q[None] = self.q[None].normalized()
            self.omega[None] += dt * self.inertia_inv @ (
                rotate_inv(self.q[None], self.torque[None]) -
                ti.math.cross(self.omega[None], self.inertia @ self.omega[None])
            )

            self.v[None] += dt * self.m_inv * self.force[None]
            self.x[None] += dt * self.v[None]

    @ti.func
    def _query_dist_bvh(self, p):
        min_dist = float('inf')
        closest = -1

        stack = ti.Vector([0] * 64, dt=int)
        stack_size = 1
        stack[0] = 0

        while stack_size > 0:
            stack_size -= 1
            node_id = stack[stack_size]

            dist_to_box = dist_aabb(p, self.bvh_bmin[node_id], self.bvh_bmax[node_id])

            if dist_to_box < min_dist:
                face_id = self.bvh_face_id[node_id]

                if face_id >= 0:
                    A = self.vertices[self.faces[face_id][0]]
                    B = self.vertices[self.faces[face_id][1]]
                    C = self.vertices[self.faces[face_id][2]]
                    dist = dist_triangle(p, A, B, C)
                    if dist < min_dist:
                        min_dist = dist
                        closest = face_id
                else:
                    left = self.bvh_left[node_id]
                    right = self.bvh_right[node_id]

                    dist_left = dist_aabb(p, self.bvh_bmin[left], self.bvh_bmax[left])
                    dist_right = dist_aabb(p, self.bvh_bmin[right], self.bvh_bmax[right])

                    if dist_left < dist_right:
                        if dist_right < min_dist:
                            stack[stack_size] = right
                            stack_size += 1
                        if dist_left < min_dist:
                            stack[stack_size] = left
                            stack_size += 1
                    else:
                        if dist_left < min_dist:
                            stack[stack_size] = left
                            stack_size += 1
                        if dist_right < min_dist:
                            stack[stack_size] = right
                            stack_size += 1
        return min_dist, closest

    @ti.func
    def collision_bvh(self, p: ti.types.vector(3, float), eps: float=1e-2):
        p_local = rotate_inv(self.q[None], p - self.x[None])

        min_dist, closest = self._query_dist_bvh(p_local)

        collision, normal = False, ti.Vector([0.0, 0.0, 0.0])
        if min_dist < eps:
            collision = True
            normal = (self.vertices[self.faces[closest][1]] - self.vertices[self.faces[closest][0]]).cross(
                self.vertices[self.faces[closest][2]] - self.vertices[self.faces[closest][0]]
            ).normalized()
            normal = rotate(self.q[None], normal)
        return collision, normal

    @ti.func
    def collision_sdf(self, p: ti.types.vector(3, float), eps: float = 3e-2):
        sdf_val = self.interpolate_sdf(p)
        collision = sdf_val < eps
        normal = ti.Vector([0.0, 0.0, 0.0])
        if collision:
            normal = self.interpolate_sdf_gradient(p).normalized()
        return collision, normal

    @ti.func
    def velocity_at_point(self, p):
        omega_world = rotate(self.q[None], self.omega[None])
        return self.v[None] + omega_world.cross(p - self.x[None])

    def render(self, scene):
        self._update_position()
        scene.mesh(self._rendered_vertices, self._rendered_indices)

