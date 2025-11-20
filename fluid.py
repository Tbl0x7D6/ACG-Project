import taichi as ti
import numpy as np

ti.init(arch=ti.cuda)

# MAC grid
res = 512
dt = 1e-4
substeps = int(1 / 60 // dt)
jacobi_iters = 50

dx = 1.0 / res
rho = 1000.0

u = ti.field(dtype=ti.f32, shape=(res + 1, res))
v = ti.field(dtype=ti.f32, shape=(res, res + 1))
u_new = ti.field(dtype=ti.f32, shape=(res + 1, res))
v_new = ti.field(dtype=ti.f32, shape=(res, res + 1))

p = ti.field(dtype=ti.f32, shape=(res, res))
p_new = ti.field(dtype=ti.f32, shape=(res, res))
div = ti.field(dtype=ti.f32, shape=(res, res))

num_particles_per_cell = 256
particle_radius = dx * 1.001 * ti.sqrt(2) / 2
particles = ti.Vector.field(2, dtype=ti.f32, shape=(res * res * num_particles_per_cell))
available_particles = ti.field(dtype=ti.i32, shape=(res * res * num_particles_per_cell))

# level set
solid_phi = ti.field(dtype=ti.f32, shape=(res, res))
phi = ti.field(dtype=ti.f32, shape=(res, res))
phi_new = ti.field(dtype=ti.f32, shape=(res, res))

@ti.func
def interpolate_phi(x, y):
    i = ti.cast(x / dx, ti.i32)
    j = ti.cast(y / dx, ti.i32)

    i = ti.max(0, ti.min(res - 1, i))
    j = ti.max(0, ti.min(res - 1, j))

    fx = (x / dx) - i
    fy = (y / dx) - j
    fx = ti.max(0.0, ti.min(1.0, fx))
    fy = ti.max(0.0, ti.min(1.0, fy))

    i_next = ti.min(i + 1, res - 1)
    j_next = ti.min(j + 1, res - 1)

    return (phi[i, j] * (1 - fx) * (1 - fy) +
            phi[i_next, j] * fx * (1 - fy) +
            phi[i, j_next] * (1 - fx) * fy +
            phi[i_next, j_next] * fx * fy)

@ti.func
def interpolate_solid_phi(x, y):
    i = ti.cast(x / dx, ti.i32)
    j = ti.cast(y / dx, ti.i32)

    i = ti.max(0, ti.min(res - 1, i))
    j = ti.max(0, ti.min(res - 1, j))

    fx = (x / dx) - i
    fy = (y / dx) - j
    fx = ti.max(0.0, ti.min(1.0, fx))
    fy = ti.max(0.0, ti.min(1.0, fy))

    i_next = ti.min(i + 1, res - 1)
    j_next = ti.min(j + 1, res - 1)

    return (solid_phi[i, j] * (1 - fx) * (1 - fy) +
            solid_phi[i_next, j] * fx * (1 - fy) +
            solid_phi[i, j_next] * (1 - fx) * fy +
            solid_phi[i_next, j_next] * fx * fy)

@ti.kernel
def init():
    for i in ti.grouped(u):
        u[i] = 0.0
        u_new[i] = 0.0

    for i in ti.grouped(v):
        v[i] = 0.0
        v_new[i] = 0.0

    for i in ti.grouped(p):
        p[i] = 0.0
        p_new[i] = 0.0
        div[i] = 0.0

    for i, j in ti.ndrange(res, res):
        if i < res // 16 or i > res * 15 / 16 or j < res // 16 or j > res * 15 / 16:
            solid_phi[i, j] = -1.0
        else:
            solid_phi[i, j] = 1.0

    for i, j in ti.ndrange(res, res):
        r = ((i - res / 2) ** 2 + (j - res / 2) ** 2) ** 0.5 * dx
        phi[i, j] = r - 0.2

    for i, j, k in ti.ndrange(res, res, num_particles_per_cell):
        idx = (i * res + j) * num_particles_per_cell + k
        offset = ti.Vector([ti.random(), ti.random()])
        pos = offset * dx + ti.Vector([i * dx, j * dx])
        particles[idx] = pos
        if interpolate_phi(pos.x, pos.y) < 0 and interpolate_solid_phi(pos.x, pos.y) > 0:
            available_particles[idx] = 1
        else:
            available_particles[idx] = 0

@ti.kernel
def apply_gravity():
    g = -4.0
    for i, j in ti.ndrange(res, res + 1):
        v[i, j] += g * dt

u_valid_old = ti.field(dtype=ti.i32, shape=(res + 1, res))
v_valid_old = ti.field(dtype=ti.i32, shape=(res, res + 1))
u_valid = ti.field(dtype=ti.i32, shape=(res + 1, res))
v_valid = ti.field(dtype=ti.i32, shape=(res, res + 1))

# solve laplace equation by Jacobi iteration for velocity
@ti.kernel
def extrapolate():
    for i, j in ti.ndrange(res + 1, res):
        if (i < res and solid_phi[i, j] > 0 and phi[i, j] < 0) or (i > 0 and solid_phi[i - 1, j] > 0 and phi[i - 1, j] < 0):
            u_valid[i, j] = 1
        else:
            u_valid[i, j] = 0
    for i, j in ti.ndrange(res, res + 1):
        if (j < res and solid_phi[i, j] > 0 and phi[i, j] < 0) or (j > 0 and solid_phi[i, j - 1] > 0 and phi[i, j - 1] < 0):
            v_valid[i, j] = 1
        else:
            v_valid[i, j] = 0

    for i in ti.grouped(u_new):
        u_new[i] = u[i]
    for i in ti.grouped(v_new):
        v_new[i] = v[i]

    # for _ in range(10):
    for i, j in ti.ndrange(res + 1, res):
        u_valid_old[i, j] = u_valid[i, j]
    for i, j in ti.ndrange(res, res + 1):
        v_valid_old[i, j] = v_valid[i, j]

    for i, j in ti.ndrange(res + 1, res):
        if u_valid_old[i, j] == 0:
            sum_u = 0.0
            count = 0
            if i > 0 and u_valid_old[i - 1, j] == 1:
                sum_u += u[i - 1, j]
                count += 1
            if i < res and u_valid_old[i + 1, j] == 1:
                sum_u += u[i + 1, j]
                count += 1
            if j > 0 and u_valid_old[i, j - 1] == 1:
                sum_u += u[i, j - 1]
                count += 1
            if j < res - 1 and u_valid_old[i, j + 1] == 1:
                sum_u += u[i, j + 1]
                count += 1
            if count > 0:
                u_new[i, j] = sum_u / count
                u_valid[i, j] = 1

    for i, j in ti.ndrange(res, res + 1):
        if v_valid_old[i, j] == 0:
            sum_v = 0.0
            count = 0
            if i > 0 and v_valid_old[i - 1, j] == 1:
                sum_v += v[i - 1, j]
                count += 1
            if i < res - 1 and v_valid_old[i + 1, j] == 1:
                sum_v += v[i + 1, j]
                count += 1
            if j > 0 and v_valid_old[i, j - 1] == 1:
                sum_v += v[i, j - 1]
                count += 1
            if j < res and v_valid_old[i, j + 1] == 1:
                sum_v += v[i, j + 1]
                count += 1
            if count > 0:
                v_new[i, j] = sum_v / count
                v_valid[i, j] = 1

    for i, j in ti.ndrange(res + 1, res):
        u[i, j] = u_new[i, j]
    for i, j in ti.ndrange(res, res + 1):
        v[i, j] = v_new[i, j]

@ti.func
def sample_u(x, y):
    i = ti.cast(x / dx, ti.i32)
    j = ti.cast((y - 0.5 * dx) / dx, ti.i32)

    i = ti.max(0, ti.min(res, i))
    j = ti.max(0, ti.min(res - 1, j))

    fx = (x / dx) - i
    fy = (y - 0.5 * dx) / dx - j
    fx = ti.max(0.0, ti.min(1.0, fx))
    fy = ti.max(0.0, ti.min(1.0, fy))

    i_next = ti.min(i + 1, res)
    j_next = ti.min(j + 1, res - 1)

    return (u[i, j] * (1 - fx) * (1 - fy) +
            u[i_next, j] * fx * (1 - fy) +
            u[i, j_next] * (1 - fx) * fy +
            u[i_next, j_next] * fx * fy)

@ti.func
def sample_v(x, y):
    i = ti.cast((x - 0.5 * dx) / dx, ti.i32)
    j = ti.cast(y / dx, ti.i32)

    i = ti.max(0, ti.min(res - 1, i))
    j = ti.max(0, ti.min(res, j))

    fx = (x - 0.5 * dx) / dx - i
    fy = (y / dx) - j
    fx = ti.max(0.0, ti.min(1.0, fx))
    fy = ti.max(0.0, ti.min(1.0, fy))

    i_next = ti.min(i + 1, res - 1)
    j_next = ti.min(j + 1, res)

    return (v[i, j] * (1 - fx) * (1 - fy) +
            v[i_next, j] * fx * (1 - fy) +
            v[i, j_next] * (1 - fx) * fy +
            v[i_next, j_next] * fx * fy)

@ti.func
def rk2_trace(x, y, dt):
    u1 = sample_u(x, y)
    v1 = sample_v(x, y)

    x_mid = x + 0.5 * dt * u1
    y_mid = y + 0.5 * dt * v1

    u2 = sample_u(x_mid, y_mid)
    v2 = sample_v(x_mid, y_mid)

    x_back = x + dt * u2
    y_back = y + dt * v2

    return ti.Vector([x_back, y_back])

@ti.kernel
def advect_particles():
    for i in ti.ndrange(res * res * num_particles_per_cell):
        if available_particles[i] == 1:
            particles[i] = rk2_trace(particles[i].x, particles[i].y, dt)
            if interpolate_solid_phi(particles[i].x, particles[i].y) < 0:
                # move particle to solid surface
                if particles[i].x < res * dx / 16:
                    particles[i].x = res * dx / 16
                if particles[i].x > res * dx * 15 / 16:
                    particles[i].x = res * dx * 15 / 16
                if particles[i].y < res * dx / 16:
                    particles[i].y = res * dx / 16
                if particles[i].y > res * dx * 15 / 16:
                    particles[i].y = res * dx * 15 / 16
            # TODO: handle solid boundary collision

@ti.func
def interpolate_phi_grad(x, y):
    i = ti.cast(x / dx - 0.5, ti.i32)
    j = ti.cast(y / dx - 0.5, ti.i32)

    fx = (x / dx) - i
    fy = (y / dx) - j

    fx = ti.max(0.0, ti.min(1.0, fx))
    fy = ti.max(0.0, ti.min(1.0, fy))
    
    i = ti.max(0, ti.min(res - 2, i))
    j = ti.max(0, ti.min(res - 2, j))
    
    i_next = ti.min(i + 1, res - 1)
    j_next = ti.min(j + 1, res - 1)

    gx_1 = (phi[i_next, j] - phi[i, j]) / dx
    gx_2 = (phi[i_next, j_next] - phi[i, j_next]) / dx

    gy_1 = (phi[i, j_next] - phi[i, j]) / dx
    gy_2 = (phi[i_next, j_next] - phi[i_next, j]) / dx

    return ti.Vector([
        gx_1 * (1 - fy) + gx_2 * fy,
        gy_1 * (1 - fx) + gy_2 * fx
    ])

@ti.kernel
def update_level_set():
    for i, j in ti.ndrange(res, res):
        phi[i, j] = 3 * dx

    for i in ti.ndrange(res * res * num_particles_per_cell):
        if available_particles[i] == 1:
            pos = particles[i]
            cell_x = ti.cast(pos.x / dx, ti.i32)
            cell_y = ti.cast(pos.y / dx, ti.i32)
            for offset_x in ti.static([-1, 0, 1]):
                for offset_y in ti.static([-1, 0, 1]):
                    nx = cell_x + offset_x
                    ny = cell_y + offset_y
                    if 0 <= nx < res and 0 <= ny < res:
                        d = (pos - ti.Vector([(nx + 0.5) * dx, (ny + 0.5) * dx])).norm() - particle_radius
                        phi[nx, ny] = ti.min(phi[nx, ny], d)
    # TODO: smooth phi field
    # for i, j in ti.ndrange(res, res):
    #     grad = interpolate_phi_grad((i + 0.5) * dx, (j + 0.5) * dx)
    #     grad_norm = grad.norm() + 1e-6
    #     phi_new[i, j] = -phi[i, j] / ti.sqrt(ti.pow(phi[i, j], 2) + dx * dx) * (grad_norm - 1)
    # for i, j in ti.ndrange(res, res):
    #     phi[i, j] = phi_new[i, j]

    # for _ in range(10):
    #     for i, j in ti.ndrange(res, res):
    #         phi_new[i, j] = phi[i, j]
    #     for i, j in ti.ndrange(res, res):
    #         sum_phi = 0.0
    #         count = 0
    #         if i > 0:
    #             sum_phi += phi_new[i - 1, j]
    #             count += 1
    #         if i < res - 1:
    #             sum_phi += phi_new[i + 1, j]
    #             count += 1
    #         if j > 0:
    #             sum_phi += phi_new[i, j - 1]
    #             count += 1
    #         if j < res - 1:
    #             sum_phi += phi_new[i, j + 1]
    #             count += 1
    #         phi[i, j] = (phi_new[i, j] + sum_phi) / (count + 1)

@ti.kernel
def advect():
    for i, j in ti.ndrange(res + 1, res):
        x = i * dx
        y = (j + 0.5) * dx

        back = rk2_trace(x, y, -dt)
        u_new[i, j] = sample_u(back.x, back.y)

    for i, j in ti.ndrange(res, res + 1):
        x = (i + 0.5) * dx
        y = j * dx

        back = rk2_trace(x, y, -dt)
        v_new[i, j] = sample_v(back.x, back.y)

    for i, j in ti.ndrange(res + 1, res):
        u[i, j] = u_new[i, j]
    for i, j in ti.ndrange(res, res + 1):
        v[i, j] = v_new[i, j]

@ti.kernel
def solid_constraint():
    for i, j in ti.ndrange(res, res):
        if i < res - 1 and solid_phi[i + 1, j] < 0 and u[i + 1, j] > 0:
            u[i + 1, j] = 0.0
        if i > 0 and solid_phi[i - 1, j] < 0 and u[i, j] < 0:
            u[i, j] = 0.0
        if j < res - 1 and solid_phi[i, j + 1] < 0 and v[i, j + 1] > 0:
            v[i, j + 1] = 0.0
        if j > 0 and solid_phi[i, j - 1] < 0 and v[i, j] < 0:
            v[i, j] = 0.0
        # if solid_phi[i, j] < 0:
        #     u[i, j] = 0.0
        #     u[i + 1, j] = 0.0
        #     v[i, j] = 0.0
        #     v[i, j + 1] = 0.0

@ti.kernel
def compute_div():
    for i, j in ti.ndrange(res, res):
        div[i, j] = ((u[i + 1, j] - u[i, j]) + (v[i, j + 1] - v[i, j])) / dx

@ti.kernel
def pressure_jacobi():
    for i, j in ti.ndrange(res, res):
        if phi[i, j] > 0 and solid_phi[i, j] > 0:
            p_new[i, j] = 0.0
        elif solid_phi[i, j] < 0:
            if i < res // 16:
                p_new[i, j] = p[i + 1, j]
            elif i > res * 15 / 16:
                p_new[i, j] = p[i - 1, j]
            elif j < res // 16:
                p_new[i, j] = p[i, j + 1]
            elif j > res * 15 / 16:
                p_new[i, j] = p[i, j - 1]
        else:
            p_new[i, j] = (p[i - 1, j] + p[i + 1, j] +
                           p[i, j - 1] + p[i, j + 1] -
                           rho * dx * dx * div[i, j] / dt) * 0.25

    for i, j in ti.ndrange(res, res):
        p[i, j] = p_new[i, j]

@ti.kernel
def apply_pressure_gradient():
    # TODO: surface tension
    for i, j in ti.ndrange(res + 1, res):
        if i > 0 and i < res:
            if phi[i - 1, j] < 0 or phi[i, j] < 0:
                if solid_phi[i - 1, j] >= 0 and solid_phi[i, j] >= 0:
                    u[i, j] -= dt / (rho * dx) * (p[i, j] - p[i - 1, j])
    for i, j in ti.ndrange(res, res + 1):
        if j > 0 and j < res:
            if phi[i, j - 1] < 0 or phi[i, j] < 0:
                if solid_phi[i, j - 1] >= 0 and solid_phi[i, j] >= 0:
                    v[i, j] -= dt / (rho * dx) * (p[i, j] - p[i, j - 1])

def substep():
    apply_gravity()
    extrapolate()
    advect_particles()
    update_level_set()
    advect()
    solid_constraint()
    compute_div()
    for _ in range(jacobi_iters):
        pressure_jacobi()
    apply_pressure_gradient()

init()

window = ti.ui.Window("2D Fluid Simulation", (512, 512))
canvas = window.get_canvas()
pixels = ti.field(dtype=ti.f32, shape=(res, res, 3))

@ti.kernel
def render():
    for i, j in ti.ndrange(res, res):
        if phi[i, j] < 0:
            # speed = ti.sqrt(0.5 * (u[i, j] ** 2 + u[i + 1, j] ** 2 +
            #                        v[i, j] ** 2 + v[i, j + 1] ** 2))
            speed = 1
            pixels[i, j, 0] = speed * 5.0
            pixels[i, j, 1] = speed * 5.0
            pixels[i, j, 2] = speed * 5.0
        else:
            pixels[i, j, 0] = 0.0
            pixels[i, j, 1] = 0.0
            pixels[i, j, 2] = 0.0
        
        if solid_phi[i, j] < 0:
            pixels[i, j, 0] = 0.2
            pixels[i, j, 1] = 0.2
            pixels[i, j, 2] = 0.8

frame_count = 0
while window.running:
    for _ in range(substeps):
        substep()

    render()
    canvas.set_image(pixels)
    # window.save_image("output/{:05d}.png".format(frame_count))
    window.show()
    frame_count += 1