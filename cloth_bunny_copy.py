import taichi as ti
import os
from materials import RigidBody, Cloth
from utils.Cloth2Mesh.clothtomesh import export

os.makedirs("output", exist_ok=True)

ti.init(arch=ti.cuda)

bunny = RigidBody("objects/bunny.obj", scale=0.5, mass=4.0, fixed=True)

quad_size = 1.0 / 128
cloth = Cloth(k=3e4, mass=1, quad_size=quad_size, num_particles_width=128, num_particles_height=128, dashpot_damping=1e4, drag_damping=1)

x = cloth.x
n = 128

upsample_rate = 2
new_n = (n - 1) * upsample_rate + 1
y = ti.Vector.field(3, dtype=x.dtype, shape=(new_n, new_n))

# window = ti.ui.Window("Cloth with Fixed Bunny", (800, 800))
# canvas = window.get_canvas()
# canvas.set_background_color((1, 1, 1))
# scene = ti.ui.Scene()
# camera = ti.ui.Camera()

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

# parameters
dt = 1e-4
substeps = int(1 / 60 // dt)
gravity = ti.Vector([0.0, -9.8, 0.0])



@ti.kernel
def substep(dt: float):
    bunny.substep(dt)
    cloth.substep(dt)
    bunny.torque[None] = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.grouped(cloth.x):
        cloth.v[i] += gravity * dt
        collision, normal = bunny.collision(cloth.x[i])
        if collision:
            p = cloth.x[i]
            v_surface = bunny.velocity_at_point(p)
            v_rel = cloth.v[i] - v_surface
            delta_v = ti.max(-v_rel.dot(normal), 0) * normal
            cloth.v[i] += delta_v
            cloth.x[i] += normal * 1e-3
            bunny.torque[None] += (p - bunny.x[None]).cross(-delta_v * cloth.particle_mass / dt)

def quat_mul(v1, v2):
    return ti.Vector([
        v1.x * v2.x - v1.y * v2.y - v1.z * v2.z - v1.w * v2.w,
        v1.x * v2.y + v2.x * v1.y + v1.z * v2.w - v1.w * v2.z,
        v1.x * v2.z + v2.x * v1.z + v1.w * v2.y - v1.y * v2.w,
        v1.x * v2.w + v2.x * v1.w + v1.y * v2.z - v1.z * v2.y
    ])

current_t = 0.0
frame_count = 0
while True: 
    if current_t > 2.0:
        cloth._init_mass_points()
        current_t = 0
    for i in range(substeps):
        substep(dt)
        current_t += dt
    mouse = [0.5, 0.5]

    # set bunny orientation
    angle_x = (mouse[1] - 0.5) * 3.14159 * -1.5
    angle_y = (mouse[0] - 0.5) * 3.14159 * 1.5
    qx = ti.Vector([ti.cos(angle_x / 2), ti.sin(angle_x / 2), 0.0, 0.0])
    qy = ti.Vector([ti.cos(angle_y / 2), 0.0, ti.sin(angle_y / 2), 0.0])
    last_q = bunny.q[None]
    bunny.q[None] = quat_mul(qy, qx).normalized()
    bunny.omega[None] = 2 * (bunny.q[None] - last_q).yzw / (dt * substeps)
    
    
    upsample(x, n, upsample_rate)
    export(frame_count, y, new_n, bunny.x[None], bunny.q[None])
    # export(frame_count, x, n, bunny.x[None], bunny.q[None])
    frame_count += 1

    # camera.position(2, 2, 2)
    # camera.lookat(0, 0, 0)
    # scene.set_camera(camera)
    # scene.point_light(pos=(4, 4, 4), color=(1, 1, 1))
    # bunny.render(scene)
    # cloth.render(scene)
    # canvas.scene(scene)
    # # window.save_image("output/{:05d}.png".format(frame_count))
    # window.show()
    # frame_count += 1
