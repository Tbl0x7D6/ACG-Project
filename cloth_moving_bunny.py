import taichi as ti
from utils.Cloth2Mesh.clothtomesh import export
from materials import RigidBody, Cloth
from pathlib import Path

ti.init(arch=ti.cuda)

# parameters
n = 128
dt = 4e-2 / n
substeps = int(1 / 60 // dt)
quad_size = 1.0 / n
theta = 0.3

# cloth + bunny setup
gravity = ti.Vector([0.0, -9.8, 0.0])
Y = 1e4
damping = 5e4
drag_damping = 3
cloth_particle_mass = 0.003
cloth_total_mass = cloth_particle_mass * n * n
mu = 0.2

cloth = Cloth(
    k=Y,
    mass=cloth_total_mass,
    quad_size=quad_size,
    num_particles_width=n,
    num_particles_height=n,
    dashpot_damping=damping,
    drag_damping=drag_damping,
)

bunny_mass = 15.0
bunny_start_height = 0.6
bunny_initial_velocity = ti.Vector([0.0, 0.0, 0.0])
bunny = RigidBody(
    "objects/bunny.obj",
    scale=0.3,
    mass=bunny_mass,
    x=ti.Vector([0.0, bunny_start_height, 0.0]),
    v=bunny_initial_velocity,
    fixed=False,
    use_sdf=True,
    sdf_resolution=128,
)
# position and velocity (cloth fields)
x = cloth.x
v = cloth.v

upsample_rate = 2
new_n = (n - 1) * upsample_rate + 1
y = ti.Vector.field(3, dtype=x.dtype, shape=(new_n, new_n))

@ti.kernel
def init():
    for i, j in x:
        x[i, j] = [(i * quad_size - 0.5) * ti.cos(theta), - (i * quad_size - 0.5) * ti.sin(theta), j * quad_size - 0.5]
        v[i, j] = [0.0, 0.0, 0.0]
    bunny.x[None] = ti.Vector([0.0, bunny_start_height, 0.0])
    bunny.v[None] = bunny_initial_velocity
    bunny.omega[None] = ti.Vector([0.0, 0.0, 0.0])
    bunny.q[None] = ti.Vector([1.0, 0.0, 0.0, 0.0])
    bunny.force[None] = gravity * bunny_mass
    bunny.torque[None] = ti.Vector([0.0, 0.0, 0.0])
    
@ti.kernel
def bunny_init():
    bunny.x[None] = ti.Vector([0.0, bunny_start_height, 0.0])
    bunny.v[None] = bunny_initial_velocity
    bunny.omega[None] = ti.Vector([0.0, 0.0, 0.0])
    bunny.q[None] = ti.Vector([1.0, 0.0, 0.0, 0.0])
    bunny.force[None] = gravity * bunny_mass
    bunny.torque[None] = ti.Vector([0.0, 0.0, 0.0])

spring_offsets = []
for i in range(-2, 3):
    for j in range(-2, 3):
        if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
            spring_offsets.append(ti.Vector([i, j]))

@ti.kernel
def substep(current_t: float):
    bunny.force[None] = gravity * bunny_mass
    bunny.torque[None] = ti.Vector([0.0, 0.0, 0.0])

    for i in ti.grouped(x):
        v[i] += gravity * dt

    cloth.substep(dt)

    for i in ti.grouped(x):
        collision, normal = bunny.collision(x[i])
        if collision:
            v_surface = bunny.velocity_at_point(x[i])
            v_rel = v[i] - v_surface
            vn = v_rel.dot(normal)
            if vn < 0:
                delta_v = -vn * normal
                v[i] += delta_v
                impulse = delta_v * cloth.particle_mass
                bunny.v[None] -= impulse * bunny.m_inv
                bunny.torque[None] += (x[i] - bunny.x[None]).cross(-impulse / dt)
                x[i] += normal * 1e-3
            v_tangent = v_rel - vn * normal
            t_norm = v_tangent.norm()
            if t_norm > 1e-6:
                friction_dir = v_tangent / t_norm
                friction_delta_v = -mu * v_tangent
                v[i] += friction_delta_v
                friction_impulse = friction_delta_v * cloth.particle_mass
                bunny.v[None] -= friction_impulse * bunny.m_inv
                bunny.torque[None] += (x[i] - bunny.x[None]).cross(-friction_impulse / dt)

    bunny.substep(dt)

    # fix 4 vertices
    v[0, n - 1] = ti.Vector([0.0, 0.0, 0.0])
    x[0, n - 1] = ti.Vector([-0.5 * ti.cos(theta), 0.5 * ti.sin(theta), 0.5 - quad_size])

    x[0, 0] = ti.Vector([-0.5 * ti.cos(theta), 0.5 * ti.sin(theta), -0.5])
    v[0, 0] = ti.Vector([0.0, 0.0, 0.0])

    x[n - 1, 0] = ti.Vector([(0.5 - quad_size) * ti.cos(theta), -(0.5 - quad_size) * ti.sin(theta), -0.5])
    v[n - 1, 0] = ti.Vector([0.0, 0.0, 0.0])

    x[n - 1, n - 1] = ti.Vector([(0.5 - quad_size) * ti.cos(theta), -(0.5 - quad_size) * ti.sin(theta), 0.5 - quad_size])
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

@ti.func
def sample(src, i, j):
    ii = max(0, min(n - 1, i))
    jj = max(0, min(n - 1, j))
    return src[ii, jj]

@ti.func
def catmull_rom(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * ((2.0 * p1) + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)

@ti.func
def bicubic(src, u, v):
    # i0 = ti.floor(u).astype(int)
    # j0 = ti.floor(v).astype(int)
    i0 = ti.cast(ti.floor(u), ti.i32)
    j0 = ti.cast(ti.floor(v), ti.i32)
    du = u - float(i0)
    dv = v - float(j0)

    r0 = catmull_rom(sample(src, i0 - 1, j0 - 1), sample(src, i0, j0 - 1), sample(src, i0 + 1, j0 - 1), sample(src, i0 + 2, j0 - 1), du)
    r1 = catmull_rom(sample(src, i0 - 1, j0), sample(src, i0, j0), sample(src, i0 + 1, j0), sample(src, i0 + 2, j0), du)
    r2 = catmull_rom(sample(src, i0 - 1, j0 + 1), sample(src, i0, j0 + 1), sample(src, i0 + 1, j0 + 1), sample(src, i0 + 2, j0 + 1), du)
    r3 = catmull_rom(sample(src, i0 - 1, j0 + 2), sample(src, i0, j0 + 2), sample(src, i0 + 1, j0 + 2), sample(src, i0 + 2, j0 + 2), du)

    return catmull_rom(r0, r1, r2, r3, dv)

@ti.kernel
def upsample(x: ti.template(), n: int, upsamle_rate: int):
    for i in range(new_n):
        for j in range(new_n):
            u = ti.float32(i) / upsamle_rate
            v = ti.float32(j) / upsamle_rate
            y[i, j] = bicubic(x, u, v)

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
frame_count = 0

if Path("cloth").exists():
    import shutil
    shutil.rmtree("cloth")
Path("cloth").mkdir(parents=True, exist_ok=True)

window = ti.ui.Window("Cloth with Moving Bunny", (800, 800))
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
init()
# while frame_count < 1100:

#     for i in range(substeps):
#         substep(current_t)
#         current_t += dt
#     update_vertices()
#     if frame_count < 500:
#         bunny_init()
#     elif frame_count >= 600:
#         bunny_init()
#         upsample(x, n, upsample_rate)
#         export(frame_count - 500, y, new_n, bunny.x[None], bunny.q[None])
#     else:
#         upsample(x, n, upsample_rate)
#         export(frame_count - 500, y, new_n, bunny.x[None], bunny.q[None])
#         # export(frame_count - 500, x, n, bunny.x[None], bunny.q[None])
        
#     if frame_count == 599:
#         path = f"cloth/final.txt"
#         with open(path, "w") as f:
#             f.write(f"{bunny.x[None][0]}, {bunny.x[None][1]}, {bunny.x[None][2]}, {bunny.q[None][0]}, {bunny.q[None][1]}, {bunny.q[None][2]}, {bunny.q[None][3]}\n")
#             f.write(f"{bunny.v[None][0]}, {bunny.v[None][1]}, {bunny.v[None][2]}, {bunny.omega[None][0]}, {bunny.omega[None][1]}, {bunny.omega[None][2]}\n")
#     frame_count += 1
while frame_count < 700:

    for i in range(substeps):
        substep(current_t)
        current_t += dt
    update_vertices()
    if frame_count < 500:
        bunny_init()
    frame_count += 1


# Pos:
# (0.4787141084671021, 0.9940526485443115, 0.5012763405684386)

# Quat:
# (0.7191821932792664, -0.14212842285633087, 0.30610981583595276, -0.6073492765426636)

    camera.position(1, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    bunny.render(scene)
    cloth.render(scene)
    canvas.scene(scene)
    window.show()