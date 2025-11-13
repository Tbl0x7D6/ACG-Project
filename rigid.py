import taichi as ti
import taichi.math as tm
import numpy as np

ti.init(arch=ti.cuda)

# parameters
dt = 1e-3
substeps = int(1 / 60 // dt)

# cuboid
a = 20.0
b = 1.0

I = 1.0 / 12 * ti.Matrix([
    [a**2 + b**2, 0.0, 0.0],
    [0.0, 2 * b**2, 0.0],
    [0.0, 0.0, a**2 + b**2]
])
I_inv = I.inverse()

# angular velocity and quaternion
omega = ti.Vector.field(3, dtype=float, shape=())
q = ti.Vector.field(4, dtype=float, shape=())

omega[None] = tm.vec3(20.0, 0.1, 0.0)
q[None] = tm.vec4(1.0, 0.0, 0.0, 0.0)

@ti.func
def quat_mul(v1, v2):
    return ti.Vector([
        v1.x * v2.x - v1.y * v2.y - v1.z * v2.z - v1.w * v2.w,
        v1.x * v2.y + v2.x * v1.y + v1.z * v2.w - v1.w * v2.z,
        v1.x * v2.z + v2.x * v1.z + v1.w * v2.y - v1.y * v2.w,
        v1.x * v2.w + v2.x * v1.w + v1.y * v2.z - v1.z * v2.y
    ])

@ti.func
def quat_conj(v):
    return ti.Vector([v.x, -v.y, -v.z, -v.w])

@ti.func
def quat_rotate(q, v):
    q_v = ti.Vector([0.0, v.x, v.y, v.z])
    return quat_mul(quat_mul(q, q_v), quat_conj(q)).yzw

@ti.kernel
def substep():
    q[None] += dt * 0.5 * quat_mul(q[None], ti.Vector([0.0, omega[None].x, omega[None].y, omega[None].z]))
    q[None] = q[None].normalized()
    omega[None] -= dt * I_inv @ (tm.cross(omega[None], I @ omega[None]))

indices = ti.field(dtype=int, shape=36)
vertices = ti.Vector.field(3, dtype=float, shape=8)

indices.from_numpy(np.array([
    0, 1, 2, 0, 2, 3,
    4, 6, 5, 4, 7, 6,
    0, 4, 5, 0, 5, 1,
    3, 2, 6, 3, 6, 7,
    1, 5, 6, 1, 6, 2,
    0, 3, 7, 0, 7, 4
], dtype=np.int32))

@ti.kernel
def update_vertices():
    for i in range(8):
        x = -a / 2 if ((i+1) & 2) == 0 else a / 2
        y = -b / 2 if (i & 2) == 0 else b / 2
        z = -b / 2 if (i & 4) == 0 else b / 2
        vertices[i] = quat_rotate(q[None], tm.vec3(x, y, z))

window = ti.ui.Window("Rigid Body Rotation", (800, 800))
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0
while window.running:
    for i in range(substeps):
        substep()
        current_t += dt
    update_vertices()

    camera.position(30, 30, 30)
    camera.lookat(0, 0, 0)
    scene.set_camera(camera)
    scene.mesh(vertices, indices, color=(0.2, 0.6, 0.8))
    canvas.scene(scene)
    window.show()