import taichi as ti
import os
from materials import RigidBody, Cloth

os.makedirs("output", exist_ok=True)

ti.init(arch=ti.cuda)

window = ti.ui.Window("Cloth with Fixed Bunny", (800, 800))
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

# parameters
dt = 1e-4
substeps = int(1 / 60 // dt)
gravity = ti.Vector([0.0, -9.8, 0.0])

bunny = RigidBody("objects/bunny.obj", scale=0.5, mass=1.0)
bunny.fixed = True

quad_size = 1.0 / 128
cloth = Cloth(k=3e4, mass=1, quad_size=quad_size, num_particles_width=128, num_particles_height=128, dashpot_damping=1e4, drag_damping=1)

@ti.kernel
def substep(dt: float):
    bunny.substep(dt)
    cloth.substep(dt)
    for i in ti.grouped(cloth.x):
        cloth.v[i] += gravity * dt
        collision, normal = bunny.collision(cloth.x[i])
        if collision:
            cloth.v[i] -= cloth.v[i].dot(normal) * normal
            cloth.x[i] += normal * 1e-3

@ti.kernel
def generate_random_vec4()-> ti.math.vec4:
    random_vec = ti.Vector([ti.random() for _ in range(4)])
    return random_vec.normalized()

current_t = 0.0
frame_count = 0
while window.running:
    if current_t > 1.5:
        cloth._init_mass_points()
        bunny.q[None] = generate_random_vec4()
        current_t = 0
    for i in range(substeps):
        substep(dt)
        current_t += dt
    camera.position(2, 2, 2)
    camera.lookat(0, 0, 0)
    scene.set_camera(camera)
    scene.point_light(pos=(4, 4, 4), color=(1, 1, 1))
    bunny.render(scene)
    cloth.render(scene)
    canvas.scene(scene)
    # window.save_image("output/{:05d}.png".format(frame_count))
    window.show()
    frame_count += 1
