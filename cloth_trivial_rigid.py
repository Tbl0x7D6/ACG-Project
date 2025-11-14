import taichi as ti

ti.init(arch=ti.cuda)

# parameters
n = 128
dt = 4e-2 / n
substeps = int(1 / 60 // dt)
quad_size = 1.0 / n

ball_radius = 0.3
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_v = ti.Vector.field(3, dtype=float, shape=(1, ))

gravity = ti.Vector([0.0, -9.8, 0.0])
Y = 3e4
damping = 1e4
drag_damping = 3

# position and velocity
x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))
m = 10000.0
m_inv = 1.0 / m

@ti.kernel
def init():
    for i, j in x:
        x[i, j] = [i * quad_size - 0.5, 0, j * quad_size - 0.5]
        v[i, j] = [0.0, 0.0, 0.0]
    ball_center[0] = [0.0, 0.6, 0.0]
    ball_v[0] = [0.0, -0.5, 0.0] + [0.3, 0.0, 0.0] * ti.random()

spring_offsets = []
for i in range(-2, 3):
    for j in range(-2, 3):
        if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
            spring_offsets.append(ti.Vector([i, j]))

@ti.kernel
def substep(current_t: float):
    for i in ti.grouped(x):
        v[i] += gravity * dt

    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                d = x_ij.normalized()
                rest_length = spring_offset.norm() * quad_size
                force += - Y * d * (x_ij.norm() / rest_length - 1.0)
                force += - v_ij.dot(d) * d * damping * quad_size
        v[i] += force * dt

    for i in ti.grouped(x):
        v[i] *= ti.exp(-drag_damping * dt)
        offset_to_center = x[i] - ball_center[0]
        if offset_to_center.norm() <= ball_radius:
            n = offset_to_center.normalized()
            delta_v = -min((v[i] - ball_v[0]).dot(n), 0) * n
            v[i] += delta_v
            x[i] = ball_center[0] + n * ball_radius
            ball_v[0] += -delta_v * m_inv
        x[i] += v[i] * dt

    ball_v[0] += gravity * dt
    ball_center[0] += ball_v[0] * dt

    # fix 4 vertices
    v[0, n - 1] = ti.Vector([0.0, 0.0, 0.0])
    x[0, n - 1] = ti.Vector([-0.5, 0.0, 0.5 - quad_size])

    x[0, 0] = ti.Vector([-0.5, 0.0, -0.5])
    v[0, 0] = ti.Vector([0.0, 0.0, 0.0])

    if current_t < 3.0:
        x[n - 1, 0] = ti.Vector([0.5 - quad_size, 0.0, -0.5])
        v[n - 1, 0] = ti.Vector([0.0, 0.0, 0.0])

        x[n - 1, n - 1] = ti.Vector([0.5 - quad_size, 0.0, 0.5 - quad_size])
        v[n - 1, n - 1] = ti.Vector([0.0, 0.0, 0.0])


# rendering
num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]

window = ti.ui.Window("Taichi Simulation on GGUI", (512, 512), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0
init()

@ti.kernel
def init_mesh():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # 1st triangle of the square
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # 2nd triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = (0.22, 0.72, 0.52)
        else:
            colors[i * n + j] = (1, 0.334, 0.52)

init_mesh()

while window.running:
    if current_t > 6:
        # Reset
        init()
        current_t = 0

    for i in range(substeps):
        substep(current_t)
        current_t += dt
    update_vertices()

    camera.position(1, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)

    # Draw a smaller ball to avoid visual penetration
    scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()
