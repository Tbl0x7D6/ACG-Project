import taichi as ti
from typing import Callable, Tuple

gravity = ti.Vector([0.0, -9.8, 0.0])

@ti.data_oriented
class Cloth:

    def __init__(self, k: float, mass: float, quad_size: float, num_particles_width: int, num_particles_height: int,
                 dashpot_damping: float, drag_damping: float):
        self.num_particles_width = num_particles_width
        self.num_particles_height = num_particles_height
        self.quad_size = quad_size

        self.k = k
        self.mass = mass
        self.particle_mass = mass / (num_particles_width * num_particles_height)

        self.x = ti.Vector.field(3, dtype=float, shape=(num_particles_width, num_particles_height))
        self.v = ti.Vector.field(3, dtype=float, shape=(num_particles_width, num_particles_height))
        self._init_mass_points()

        self.dashpot_damping = dashpot_damping
        self.drag_damping = drag_damping

        self._spring_offsets = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                    self._spring_offsets.append(ti.Vector([i, j]))

        num_triangles = (num_particles_width - 1) * (num_particles_height - 1) * 2
        self._rendered_indices = ti.field(dtype=int, shape=num_triangles * 3)
        self._rendered_vertices = ti.Vector.field(3, dtype=float, shape=(num_particles_width * num_particles_height))
        self._rendered_colors = ti.Vector.field(3, dtype=float, shape=(num_particles_width * num_particles_height))
        self._init_mesh()

        self._update_position()

    @ti.kernel
    def _init_mesh(self):
        n = self.num_particles_width
        m = self.num_particles_height
        for i, j in ti.ndrange(n - 1, m - 1):
            quad_id = (i * (n - 1)) + j
            self._rendered_indices[quad_id * 6 + 0] = i * n + j
            self._rendered_indices[quad_id * 6 + 1] = (i + 1) * n + j
            self._rendered_indices[quad_id * 6 + 2] = i * n + (j + 1)
            self._rendered_indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
            self._rendered_indices[quad_id * 6 + 4] = i * n + (j + 1)
            self._rendered_indices[quad_id * 6 + 5] = (i + 1) * n + j

        for i, j in ti.ndrange(n, m):
            if (i // 4 + j // 4) % 2 == 0:
                self._rendered_colors[i * n + j] = (0.22, 0.72, 0.52)
            else:
                self._rendered_colors[i * n + j] = (1, 0.334, 0.52)

    @ti.kernel
    def _update_position(self):
        n = self.num_particles_width
        m = self.num_particles_height
        for i, j in ti.ndrange(n, m):
            self._rendered_vertices[i * n + j] = self.x[i, j]

    @ti.kernel
    def _init_mass_points(self):
        for i, j in self.x:
            self.x[i, j] = [
                i * self.quad_size - 0.5, 1.0,
                j * self.quad_size - 0.5
            ]
            self.v[i, j] = [0.0, 0.0, 0.0]

    @ti.func
    def substep(self, dt: float):
        for i in ti.grouped(self.x):
            force = ti.Vector([0.0, 0.0, 0.0])
            for spring_offset in ti.static(self._spring_offsets):
                j = i + spring_offset
                if 0 <= j[0] < self.num_particles_width and 0 <= j[1] < self.num_particles_height:
                    x_ij = self.x[i] - self.x[j]
                    v_ij = self.v[i] - self.v[j]
                    d = x_ij.normalized()
                    rest_length = spring_offset.norm() * self.quad_size
                    force += - self.k * d * (x_ij.norm() / rest_length - 1.0)
                    force += - v_ij.dot(d) * d * self.dashpot_damping * self.quad_size
            self.v[i] += force * dt
        for i in ti.grouped(self.x):
            self.v[i] *= ti.exp(-self.drag_damping * dt)
            self.x[i] += self.v[i] * dt

    def render(self, scene):
        self._update_position()
        scene.mesh(self._rendered_vertices, self._rendered_indices, per_vertex_color=self._rendered_colors, two_sided=True)