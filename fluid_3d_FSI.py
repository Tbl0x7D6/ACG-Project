import taichi as ti
from materials import RigidBody, rotate, rotate_inv
import numpy as np
import json
import os

ti.init(arch=ti.cuda, device_memory_fraction=0.3)

USE_REFLECTION = False

# Rigid body initial configuration
RIGID_OBJ_PATH = "objects/bunny.obj"
RIGID_SCALE = 0.2
RIGID_INIT_POS = [0.6, 0.25, 0.5]
RIGID_MASS = 10.0

N1 = 256
N2 = 256
N3 = 256
dt = 1e-4
dtau = 0.001

fps = 120
substeps = int(1.0 / fps // dt)

boundary_thickness = 6

dx = 1.0 / 256
rho = 1000.0
kappa = 30

g = 9.81
restitution = 0.8

# Create directories for output
os.makedirs("levelset", exist_ok=True)
os.makedirs("rigid_states", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("plys", exist_ok=True)

vx = ti.field(dtype=ti.f32, shape=(N1 + 1, N2, N3))
vy = ti.field(dtype=ti.f32, shape=(N1, N2 + 1, N3))
vz = ti.field(dtype=ti.f32, shape=(N1, N2, N3 + 1))
vx_new = ti.field(dtype=ti.f32, shape=(N1 + 1, N2, N3))
vy_new = ti.field(dtype=ti.f32, shape=(N1, N2 + 1, N3))
vz_new = ti.field(dtype=ti.f32, shape=(N1, N2, N3 + 1))

p = ti.field(dtype=ti.f32, shape=(N1, N2, N3))
r = ti.field(dtype=ti.f32, shape=(N1, N2, N3))
p_cg = ti.field(dtype=ti.f32, shape=(N1, N2, N3))
Ap = ti.field(dtype=ti.f32, shape=(N1, N2, N3))

A_diag = ti.field(dtype=ti.f32, shape=(N1, N2, N3))
A_plus_i = ti.field(dtype=ti.f32, shape=(N1, N2, N3))
A_plus_j = ti.field(dtype=ti.f32, shape=(N1, N2, N3))
A_plus_k = ti.field(dtype=ti.f32, shape=(N1, N2, N3))
b = ti.field(dtype=ti.f32, shape=(N1, N2, N3))

J_trans = ti.Vector.field(3, dtype=ti.f32, shape=(N1, N2, N3))
J_rot = ti.Vector.field(3, dtype=ti.f32, shape=(N1, N2, N3))

p_force = ti.Vector.field(3, dtype=ti.f32, shape=())
p_torque = ti.Vector.field(3, dtype=ti.f32, shape=())

rigid = RigidBody(RIGID_OBJ_PATH, scale=RIGID_SCALE, x=ti.Vector(RIGID_INIT_POS), mass=RIGID_MASS, sdf_resolution=max(N1, N2, N3))

solid_phi = ti.field(dtype=ti.f32, shape=(N1, N2, N3))
phi = ti.field(dtype=ti.f32, shape=(N1, N2, N3))
phi_1 = ti.field(dtype=ti.f32, shape=(N1, N2, N3))
phi_2 = ti.field(dtype=ti.f32, shape=(N1, N2, N3))

# volume control (PI controller)
correct_volume = ti.field(dtype=ti.f32, shape=())
current_volume = ti.field(dtype=ti.f32, shape=())
err = ti.field(dtype=ti.f32, shape=())
err_int = ti.field(dtype=ti.f32, shape=())
zeta = 2.0
kp = 2.3 / (1 / 10)
ki = (kp / (2 * zeta)) ** 2

@ti.kernel
def calc_volume():
    count = 0.0
    for i, j, k in ti.ndrange(N1, N2, N3):
        x, y, z = (i + 0.5) * dx, (j + 0.5) * dx, (k + 0.5) * dx
        if solid_phi[i, j, k] > 0 and rigid.interpolate_sdf(ti.Vector([x, y, z])) > 0:
            count += heaviside(phi[i, j, k])
    current_volume[None] = count

@ti.func
def heaviside(phi_val):
    epsilon = 1.5 * dx
    h = 0.0
    if phi_val < -epsilon:
        h = 1.0
    elif phi_val > epsilon:
        h = 0.0
    else:
        h = 0.5 - 0.75 * (phi_val / epsilon) + 0.25 * (phi_val / epsilon) ** 3
    return h

@ti.func
def cubic_interp(v0, v1, v2, v3, f):
    return v1 + 0.5 * f * (v2 - v0 + f * (2.0 * v0 - 5.0 * v1 + 4.0 * v2 - v3 + f * (3.0 * (v1 - v2) + v3 - v0)))

@ti.func
def y_interpolate_phi(phi, x, j, z):
    # MUST use sharp cubic interpolation to avoid mass loss
    i = ti.cast(x / dx - 0.5, ti.i32)
    k = ti.cast(z / dx - 0.5, ti.i32)

    i = ti.max(1, ti.min(N1 - 2, i))
    k = ti.max(1, ti.min(N3 - 2, k))

    fx = (x / dx - 0.5) - i
    fz = (z / dx - 0.5) - k
    fx = ti.max(0.0, ti.min(1.0, fx))
    fz = ti.max(0.0, ti.min(1.0, fz))

    col0 = cubic_interp(phi[i - 1, j, k - 1], phi[i, j, k - 1], phi[i + 1, j, k - 1], phi[i + 2, j, k - 1], fx)
    col1 = cubic_interp(phi[i - 1, j, k    ], phi[i, j, k    ], phi[i + 1, j, k    ], phi[i + 2, j, k    ], fx)
    col2 = cubic_interp(phi[i - 1, j, k + 1], phi[i, j, k + 1], phi[i + 1, j, k + 1], phi[i + 2, j, k + 1], fx)
    col3 = cubic_interp(phi[i - 1, j, k + 2], phi[i, j, k + 2], phi[i + 1, j, k + 2], phi[i + 2, j, k + 2], fx)

    return cubic_interp(col0, col1, col2, col3, fz)

@ti.func
def interpolate_phi(phi, p):
    x, y, z = p.x, p.y, p.z
    j = ti.cast(y / dx - 0.5, ti.i32)
    j = ti.max(1, ti.min(N2 - 2, j))

    fy = (y / dx - 0.5) - j
    fy = ti.max(0.0, ti.min(1.0, fy))

    col0 = y_interpolate_phi(phi, x, j - 1, z)
    col1 = y_interpolate_phi(phi, x, j    , z)
    col2 = y_interpolate_phi(phi, x, j + 1, z)
    col3 = y_interpolate_phi(phi, x, j + 2, z)

    return cubic_interp(col0, col1, col2, col3, fy)

@ti.func
def interpolate_phi_grad(phi, p):
    grad = ti.Vector([
        interpolate_phi(phi, p + ti.Vector([dx, 0.0, 0.0])) - interpolate_phi(phi, p - ti.Vector([dx, 0.0, 0.0])),
        interpolate_phi(phi, p + ti.Vector([0.0, dx, 0.0])) - interpolate_phi(phi, p - ti.Vector([0.0, dx, 0.0])),
        interpolate_phi(phi, p + ti.Vector([0.0, 0.0, dx])) - interpolate_phi(phi, p - ti.Vector([0.0, 0.0, dx]))
    ]) / (2.0 * dx)
    return grad.normalized()

@ti.kernel
def init():
    for i in ti.grouped(vx):
        vx[i] = 0.0
        vx_new[i] = 0.0

    for i in ti.grouped(vy):
        vy[i] = 0.0
        vy_new[i] = 0.0

    for i in ti.grouped(vz):
        vz[i] = 0.0
        vz_new[i] = 0.0

    for i in ti.grouped(p):
        p[i] = 0.0

    for i, j, k in ti.ndrange(N1, N2, N3):
        x, y, z = (i + 0.5) * dx, (j + 0.5) * dx, (k + 0.5) * dx
        solid_phi[i, j, k] = min(
            x - boundary_thickness * dx,
            (N1 - boundary_thickness) * dx - x,
            z - boundary_thickness * dx,
            (N3 - boundary_thickness) * dx - z,
            y - boundary_thickness * dx
        )

    for i, j, k in ti.ndrange(N1, N2, N3):
        # Example 1
        # r = ((i - N1 / 3) ** 2 + (j - N2 / 3) ** 2 + (k - N3 / 2) ** 2) ** 0.5 * dx
        # phi[i, j, k] = r - 0.2

        # phi_bottom = max(j * dx - 0.1, (boundary_thickness - j) * dx, 
        #                 (i - N1 + boundary_thickness) * dx, (boundary_thickness - i) * dx,
        #                 (k - N3 + boundary_thickness) * dx, (boundary_thickness - k) * dx)

        # phi[i, j, k] = min(phi[i, j, k], phi_bottom)

        # Example 2
        # pos = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx, (k + 0.5) * dx])
        # phi[i, j, k] = min(bunny.interpolate_sdf(pos), 1.0)

        # Example 3, FSI
        phi[i, j, k] = max(j * dx - 0.8, 0.1 - j * dx,
                        i * dx - 0.4, 0.1 - i * dx,
                        k * dx - 0.8, 0.2 - k * dx)

def init_volume():
    calc_volume()
    correct_volume[None] = current_volume[None]
    err[None] = 0.0
    err_int[None] = 0.0

@ti.kernel
def apply_gravity():
    for i, j, k in ti.ndrange(N1, N2 + 1, N3):
        if phi[i, j - 1, k] < 0 or phi[i, j, k] < 0 and solid_phi[i, j - 1, k] > 0 and solid_phi[i, j, k] > 0:
            vy[i, j, k] -= g * dt

vx_valid_old = ti.field(dtype=ti.i32, shape=(N1 + 1, N2, N3))
vy_valid_old = ti.field(dtype=ti.i32, shape=(N1, N2 + 1, N3))
vz_valid_old = ti.field(dtype=ti.i32, shape=(N1, N2, N3 + 1))
vx_valid = ti.field(dtype=ti.i32, shape=(N1 + 1, N2, N3))
vy_valid = ti.field(dtype=ti.i32, shape=(N1, N2 + 1, N3))
vz_valid = ti.field(dtype=ti.i32, shape=(N1, N2, N3 + 1))

@ti.kernel
def extrapolate_mark_valid():
    for i, j, k in ti.ndrange(N1 + 1, N2, N3):
        if (i < N1 and phi[i, j, k] < 0 and solid_phi[i, j, k] > 0) or (i > 0 and phi[i - 1, j, k] < 0 and solid_phi[i - 1, j, k] > 0):
            vx_valid[i, j, k] = 1
        else:
            vx_valid[i, j, k] = 0
    for i, j, k in ti.ndrange(N1, N2 + 1, N3):
        if (j < N2 and phi[i, j, k] < 0 and solid_phi[i, j, k] > 0) or (j > 0 and phi[i, j - 1, k] < 0 and solid_phi[i, j - 1, k] > 0):
            vy_valid[i, j, k] = 1
        else:
            vy_valid[i, j, k] = 0
    for i, j, k in ti.ndrange(N1, N2, N3 + 1):
        if (k < N3 and phi[i, j, k] < 0 and solid_phi[i, j, k] > 0) or (k > 0 and phi[i, j, k - 1] < 0 and solid_phi[i, j, k - 1] > 0):
            vz_valid[i, j, k] = 1
        else:
            vz_valid[i, j, k] = 0

    for i in ti.grouped(vx):
        vx_new[i] = vx[i]
    for i in ti.grouped(vy):
        vy_new[i] = vy[i]
    for i in ti.grouped(vz):
        vz_new[i] = vz[i]

@ti.kernel
def extrapolate_iter():
    for i, j, k in ti.ndrange(N1 + 1, N2, N3):
        vx_valid_old[i, j, k] = vx_valid[i, j, k]
    for i, j, k in ti.ndrange(N1, N2 + 1, N3):
        vy_valid_old[i, j, k] = vy_valid[i, j, k]
    for i, j, k in ti.ndrange(N1, N2, N3 + 1):
        vz_valid_old[i, j, k] = vz_valid[i, j, k]

    for i, j, k in ti.ndrange(N1 + 1, N2, N3):
        if vx_valid_old[i, j, k] == 0:
            sum_vx = 0.0
            count = 0
            if i > 0 and vx_valid_old[i - 1, j, k] == 1:
                sum_vx += vx[i - 1, j, k]
                count += 1
            if i < N1 and vx_valid_old[i + 1, j, k] == 1:
                sum_vx += vx[i + 1, j, k]
                count += 1
            if j > 0 and vx_valid_old[i, j - 1, k] == 1:
                sum_vx += vx[i, j - 1, k]
                count += 1
            if j < N2 - 1 and vx_valid_old[i, j + 1, k] == 1:
                sum_vx += vx[i, j + 1, k]
                count += 1
            if k > 0 and vx_valid_old[i, j, k - 1] == 1:
                sum_vx += vx[i, j, k - 1]
                count += 1
            if k < N3 - 1 and vx_valid_old[i, j, k + 1] == 1:
                sum_vx += vx[i, j, k + 1]
                count += 1
            if count > 0:
                vx_new[i, j, k] = sum_vx / count
                vx_valid[i, j, k] = 1

    for i, j, k in ti.ndrange(N1, N2 + 1, N3):
        if vy_valid_old[i, j, k] == 0:
            sum_vy = 0.0
            count = 0
            if i > 0 and vy_valid_old[i - 1, j, k] == 1:
                sum_vy += vy[i - 1, j, k]
                count += 1
            if i < N1 - 1 and vy_valid_old[i + 1, j, k] == 1:
                sum_vy += vy[i + 1, j, k]
                count += 1
            if j > 0 and vy_valid_old[i, j - 1, k] == 1:
                sum_vy += vy[i, j - 1, k]
                count += 1
            if j < N2 and vy_valid_old[i, j + 1, k] == 1:
                sum_vy += vy[i, j + 1, k]
                count += 1
            if k > 0 and vy_valid_old[i, j, k - 1] == 1:
                sum_vy += vy[i, j, k - 1]
                count += 1
            if k < N3 - 1 and vy_valid_old[i, j, k + 1] == 1:
                sum_vy += vy[i, j, k + 1]
                count += 1
            if count > 0:
                vy_new[i, j, k] = sum_vy / count
                vy_valid[i, j, k] = 1

    for i, j, k in ti.ndrange(N1, N2, N3 + 1):
        if vz_valid_old[i, j, k] == 0:
            sum_vz = 0.0
            count = 0
            if i > 0 and vz_valid_old[i - 1, j, k] == 1:
                sum_vz += vz[i - 1, j, k]
                count += 1
            if i < N1 - 1 and vz_valid_old[i + 1, j, k] == 1:
                sum_vz += vz[i + 1, j, k]
                count += 1
            if j > 0 and vz_valid_old[i, j - 1, k] == 1:
                sum_vz += vz[i, j - 1, k]
                count += 1
            if j < N2 - 1 and vz_valid_old[i, j + 1, k] == 1:
                sum_vz += vz[i, j + 1, k]
                count += 1
            if k > 0 and vz_valid_old[i, j, k - 1] == 1:
                sum_vz += vz[i, j, k - 1]
                count += 1
            if k < N3 and vz_valid_old[i, j, k + 1] == 1:
                sum_vz += vz[i, j, k + 1]
                count += 1
            if count > 0:
                vz_new[i, j, k] = sum_vz / count
                vz_valid[i, j, k] = 1

    for i, j, k in ti.ndrange(N1 + 1, N2, N3):
        vx[i, j, k] = vx_new[i, j, k]
    for i, j, k in ti.ndrange(N1, N2 + 1, N3):
        vy[i, j, k] = vy_new[i, j, k]
    for i, j, k in ti.ndrange(N1, N2, N3 + 1):
        vz[i, j, k] = vz_new[i, j, k]

def extrapolate():
    extrapolate_mark_valid()
    for _ in range(10):
        extrapolate_iter()

@ti.func
def sample_vx(x, y, z):
    i = ti.cast(x / dx, ti.i32)
    j = ti.cast((y - 0.5 * dx) / dx, ti.i32)
    k = ti.cast((z - 0.5 * dx) / dx, ti.i32)

    i = ti.max(0, ti.min(N1, i))
    j = ti.max(0, ti.min(N2 - 1, j))
    k = ti.max(0, ti.min(N3 - 1, k))

    fx = (x / dx) - i
    fy = (y - 0.5 * dx) / dx - j
    fz = (z - 0.5 * dx) / dx - k
    fx = ti.max(0.0, ti.min(1.0, fx))
    fy = ti.max(0.0, ti.min(1.0, fy))
    fz = ti.max(0.0, ti.min(1.0, fz))

    i_next = ti.min(i + 1, N1)
    j_next = ti.min(j + 1, N2 - 1)
    k_next = ti.min(k + 1, N3 - 1)

    return (vx[i, j, k] * (1 - fx) * (1 - fy) * (1 - fz) + \
            vx[i_next, j, k] * fx * (1 - fy) * (1 - fz) + \
            vx[i, j_next, k] * (1 - fx) * fy * (1 - fz) + \
            vx[i, j, k_next] * (1 - fx) * (1 - fy) * fz + \
            vx[i_next, j_next, k] * fx * fy * (1 - fz) + \
            vx[i_next, j, k_next] * fx * (1 - fy) * fz + \
            vx[i, j_next, k_next] * (1 - fx) * fy * fz + \
            vx[i_next, j_next, k_next] * fx * fy * fz)

@ti.func
def sample_vy(x, y, z):
    i = ti.cast((x - 0.5 * dx) / dx, ti.i32)
    j = ti.cast(y / dx, ti.i32)
    k = ti.cast((z - 0.5 * dx) / dx, ti.i32)

    i = ti.max(0, ti.min(N1 - 1, i))
    j = ti.max(0, ti.min(N2, j))
    k = ti.max(0, ti.min(N3 - 1, k))

    fx = (x - 0.5 * dx) / dx - i
    fy = (y / dx) - j
    fz = (z - 0.5 * dx) / dx - k
    fx = ti.max(0.0, ti.min(1.0, fx))
    fy = ti.max(0.0, ti.min(1.0, fy))
    fz = ti.max(0.0, ti.min(1.0, fz))

    i_next = ti.min(i + 1, N1 - 1)
    j_next = ti.min(j + 1, N2)
    k_next = ti.min(k + 1, N3 - 1)

    return (vy[i, j, k] * (1 - fx) * (1 - fy) * (1 - fz) + \
            vy[i_next, j, k] * fx * (1 - fy) * (1 - fz) + \
            vy[i, j_next, k] * (1 - fx) * fy * (1 - fz) + \
            vy[i, j, k_next] * (1 - fx) * (1 - fy) * fz + \
            vy[i_next, j_next, k] * fx * fy * (1 - fz) + \
            vy[i_next, j, k_next] * fx * (1 - fy) * fz + \
            vy[i, j_next, k_next] * (1 - fx) * fy * fz + \
            vy[i_next, j_next, k_next] * fx * fy * fz)

@ti.func
def sample_vz(x, y, z):
    i = ti.cast((x - 0.5 * dx) / dx, ti.i32)
    j = ti.cast((y - 0.5 * dx) / dx, ti.i32)
    k = ti.cast(z / dx, ti.i32)

    i = ti.max(0, ti.min(N1 - 1, i))
    j = ti.max(0, ti.min(N2 - 1, j))
    k = ti.max(0, ti.min(N3, k))

    fx = (x - 0.5 * dx) / dx - i
    fy = (y - 0.5 * dx) / dx - j
    fz = (z / dx) - k
    fx = ti.max(0.0, ti.min(1.0, fx))
    fy = ti.max(0.0, ti.min(1.0, fy))
    fz = ti.max(0.0, ti.min(1.0, fz))

    i_next = ti.min(i + 1, N1 - 1)
    j_next = ti.min(j + 1, N2 - 1)
    k_next = ti.min(k + 1, N3)

    return (vz[i, j, k] * (1 - fx) * (1 - fy) * (1 - fz) + \
            vz[i_next, j, k] * fx * (1 - fy) * (1 - fz) + \
            vz[i, j_next, k] * (1 - fx) * fy * (1 - fz) + \
            vz[i, j, k_next] * (1 - fx) * (1 - fy) * fz + \
            vz[i_next, j_next, k] * fx * fy * (1 - fz) + \
            vz[i_next, j, k_next] * fx * (1 - fy) * fz + \
            vz[i, j_next, k_next] * (1 - fx) * fy * fz + \
            vz[i_next, j_next, k_next] * fx * fy * fz)

@ti.func
def rk2_trace(x, y, z, dt):
    u1 = sample_vx(x, y, z)
    v1 = sample_vy(x, y, z)
    w1 = sample_vz(x, y, z)

    x_mid = x + 0.5 * dt * u1
    y_mid = y + 0.5 * dt * v1
    z_mid = z + 0.5 * dt * w1

    u2 = sample_vx(x_mid, y_mid, z_mid)
    v2 = sample_vy(x_mid, y_mid, z_mid)
    w2 = sample_vz(x_mid, y_mid, z_mid)

    x_back = x + dt * u2
    y_back = y + dt * v2
    z_back = z + dt * w2

    return ti.Vector([x_back, y_back, z_back])

@ti.kernel
def advect_levelset():
    for i, j, k in ti.ndrange(N1, N2, N3):
        pos = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx, (k + 0.5) * dx])
        back_pos = rk2_trace(pos.x, pos.y, pos.z, -dt)
        val_back = interpolate_phi(phi, back_pos)
        fwd_pos = rk2_trace(back_pos.x, back_pos.y, back_pos.z, dt)
        val_fwd = interpolate_phi(phi, fwd_pos)
        phi_1[i, j, k] = val_back + 0.5 * (phi[i, j, k] - val_fwd)
    for i, j, k in ti.ndrange(N1, N2, N3):
        phi[i, j, k] = phi_1[i, j, k]

@ti.kernel
def reinit_levelset_iter():
    for i, j, k in ti.ndrange(N1, N2, N3):
        if i == 0 or i == N1 - 1 or j == 0 or j == N2 - 1 or k == 0 or k == N3 - 1:
            if i == 0:
                phi_2[i, j, k] = phi_1[i + 1, j, k]
            elif i == N1 - 1:
                phi_2[i, j, k] = phi_1[i - 1, j, k]
            elif j == 0:
                phi_2[i, j, k] = phi_1[i, j + 1, k]
            elif j == N2 - 1:
                phi_2[i, j, k] = phi_1[i, j - 1, k]
            elif k == 0:
                phi_2[i, j, k] = phi_1[i, j, k + 1]
            elif k == N3 - 1:
                phi_2[i, j, k] = phi_1[i, j, k - 1]
        else:
            phi_0 = phi[i, j, k]
            s = phi_0 / ti.sqrt(phi_0*phi_0 + dx*dx)
            # D_x^-
            dx_minus = (phi_1[i, j, k] - phi_1[i - 1, j, k]) / dx
            # D_x^+
            dx_plus  = (phi_1[i + 1, j, k] - phi_1[i, j, k]) / dx
            # D_y^-
            dy_minus = (phi_1[i, j, k] - phi_1[i, j - 1, k]) / dx
            # D_y^+
            dy_plus  = (phi_1[i, j + 1, k] - phi_1[i, j, k]) / dx
            # D_z^-
            dz_minus = (phi_1[i, j, k] - phi_1[i, j, k - 1]) / dx
            # D_z^+
            dz_plus  = (phi_1[i, j, k + 1] - phi_1[i, j, k]) / dx

            grad_sq_x = 0.0
            grad_sq_y = 0.0
            grad_sq_z = 0.0

            if phi_0 > 0:
                term_xm = ti.max(dx_minus, 0.0)**2
                term_xp = ti.min(dx_plus,  0.0)**2
                grad_sq_x = ti.max(term_xm, term_xp)

                term_ym = ti.max(dy_minus, 0.0)**2
                term_yp = ti.min(dy_plus,  0.0)**2
                grad_sq_y = ti.max(term_ym, term_yp)

                term_zm = ti.max(dz_minus, 0.0)**2
                term_zp = ti.min(dz_plus,  0.0)**2
                grad_sq_z = ti.max(term_zm, term_zp)
            else:
                term_xm = ti.min(dx_minus, 0.0)**2
                term_xp = ti.max(dx_plus,  0.0)**2
                grad_sq_x = ti.max(term_xm, term_xp)

                term_ym = ti.min(dy_minus, 0.0)**2
                term_yp = ti.max(dy_plus,  0.0)**2
                grad_sq_y = ti.max(term_ym, term_yp)

                term_zm = ti.min(dz_minus, 0.0)**2
                term_zp = ti.max(dz_plus,  0.0)**2
                grad_sq_z = ti.max(term_zm, term_zp)

            grad_norm = ti.sqrt(grad_sq_x + grad_sq_y + grad_sq_z)

            phi_2[i, j, k] = phi_1[i, j, k] - dtau * s * (grad_norm - 1.0)

def reinit_levelset(num_iters=30):
    phi_1.copy_from(phi)
    for _ in range(num_iters):
        reinit_levelset_iter()
        phi_1.copy_from(phi_2)
    phi.copy_from(phi_1)

@ti.func
def curvature(i, j, k):
    return (phi[i + 1, j, k] + phi[i - 1, j, k] +
            phi[i, j + 1, k] + phi[i, j - 1, k] +
            phi[i, j, k + 1] + phi[i, j, k - 1] -
            6.0 * phi[i, j, k]) / (dx * dx)

@ti.kernel
def advect(dt: float):
    for i, j, k in ti.ndrange(N1 + 1, N2, N3):
        x = i * dx
        y = (j + 0.5) * dx
        z = (k + 0.5) * dx

        back = rk2_trace(x, y, z, -dt)
        val_back = sample_vx(back.x, back.y, back.z)
        fwd_pos = rk2_trace(back.x, back.y, back.z, dt)
        val_fwd = sample_vx(fwd_pos.x, fwd_pos.y, fwd_pos.z)
        vx_new[i, j, k] = val_back + 0.5 * (vx[i, j, k] - val_fwd)

    for i, j, k in ti.ndrange(N1, N2 + 1, N3):
        x = (i + 0.5) * dx
        y = j * dx
        z = (k + 0.5) * dx

        back = rk2_trace(x, y, z, -dt)
        val_back = sample_vy(back.x, back.y, back.z)
        fwd_pos = rk2_trace(back.x, back.y, back.z, dt)
        val_fwd = sample_vy(fwd_pos.x, fwd_pos.y, fwd_pos.z)
        vy_new[i, j, k] = val_back + 0.5 * (vy[i, j, k] - val_fwd)

    for i, j, k in ti.ndrange(N1, N2, N3 + 1):
        x = (i + 0.5) * dx
        y = (j + 0.5) * dx
        z = k * dx

        back = rk2_trace(x, y, z, -dt)
        val_back = sample_vz(back.x, back.y, back.z)
        fwd_pos = rk2_trace(back.x, back.y, back.z, dt)
        val_fwd = sample_vz(fwd_pos.x, fwd_pos.y, fwd_pos.z)
        vz_new[i, j, k] = val_back + 0.5 * (vz[i, j, k] - val_fwd)

    for i in ti.grouped(vx):
        vx[i] = vx_new[i]
    for i in ti.grouped(vy):
        vy[i] = vy_new[i]
    for i in ti.grouped(vz):
        vz[i] = vz_new[i]

@ti.kernel
def constrain():
    for i, j, k in ti.ndrange(N1, N2, N3):
        if i < N1 - 1 and solid_phi[i + 1, j, k] < 0 and vx[i + 1, j, k] > 0:
            vx[i + 1, j, k] = 0.0
        elif i > 0 and solid_phi[i - 1, j, k] < 0 and vx[i, j, k] < 0:
            vx[i, j, k] = 0.0
        if j < N2 - 1 and solid_phi[i, j + 1, k] < 0 and vy[i, j + 1, k] > 0:
            vy[i, j + 1, k] = 0.0
        elif j > 0 and solid_phi[i, j - 1, k] < 0 and vy[i, j, k] < 0:
            vy[i, j, k] = 0.0
        if k < N3 - 1 and solid_phi[i, j, k + 1] < 0 and vz[i, j, k + 1] > 0:
            vz[i, j, k + 1] = 0.0
        elif k > 0 and solid_phi[i, j, k - 1] < 0 and vz[i, j, k] < 0:
            vz[i, j, k] = 0.0

        vel_left = rigid.velocity_at_point(ti.Vector([i, j + 0.5, k + 0.5]) * dx).x
        vel_right = rigid.velocity_at_point(ti.Vector([i + 1, j + 0.5, k + 0.5]) * dx).x
        vel_down = rigid.velocity_at_point(ti.Vector([i + 0.5, j, k + 0.5]) * dx).y
        vel_up = rigid.velocity_at_point(ti.Vector([i + 0.5, j + 1, k + 0.5]) * dx).y
        vel_back = rigid.velocity_at_point(ti.Vector([i + 0.5, j + 0.5, k]) * dx).z
        vel_front = rigid.velocity_at_point(ti.Vector([i + 0.5, j + 0.5, k + 1]) * dx).z

        if rigid.interpolate_sdf(ti.Vector([i - 0.5, j + 0.5, k + 0.5]) * dx) < 0 and vx[i, j, k] < vel_left:
            vx[i, j, k] = vel_left
        elif rigid.interpolate_sdf(ti.Vector([i + 1.5, j + 0.5, k + 0.5]) * dx) < 0 and vx[i + 1, j, k] > vel_right:
            vx[i + 1, j, k] = vel_right
        if rigid.interpolate_sdf(ti.Vector([i + 0.5, j - 0.5, k + 0.5]) * dx) < 0 and vy[i, j, k] < vel_down:
            vy[i, j, k] = vel_down
        elif rigid.interpolate_sdf(ti.Vector([i + 0.5, j + 1.5, k + 0.5]) * dx) < 0 and vy[i, j + 1, k] > vel_up:
            vy[i, j + 1, k] = vel_up
        if rigid.interpolate_sdf(ti.Vector([i + 0.5, j + 0.5, k - 0.5]) * dx) < 0 and vz[i, j, k] < vel_back:
            vz[i, j, k] = vel_back
        elif rigid.interpolate_sdf(ti.Vector([i + 0.5, j + 0.5, k + 1.5]) * dx) < 0 and vz[i, j, k + 1] > vel_front:
            vz[i, j, k + 1] = vel_front

@ti.kernel
def build_matrix(volume_correction: ti.types.f32):
    for i, j, k in ti.ndrange(N1, N2, N3):
        b[i, j, k] = 0.0
    for i, j, k in ti.ndrange(N1, N2, N3):
        A_diag[i, j, k] = 0.0
        A_plus_i[i, j, k] = 0.0
        A_plus_j[i, j, k] = 0.0
        A_plus_k[i, j, k] = 0.0
        J_trans[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        J_rot[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        if phi[i, j, k] <= 0 and solid_phi[i, j, k] > 0:
            if i > 0 and solid_phi[i - 1, j, k] > 0:
                A_diag[i, j, k] += 1.0
                b[i, j, k] += vx[i, j, k]
            if i < N1 - 1 and solid_phi[i + 1, j, k] > 0:
                A_diag[i, j, k] += 1.0
                b[i, j, k] -= vx[i + 1, j, k]
            if j > 0 and solid_phi[i, j - 1, k] > 0:
                A_diag[i, j, k] += 1.0
                b[i, j, k] += vy[i, j, k]
            if j < N2 - 1 and solid_phi[i, j + 1, k] > 0:
                A_diag[i, j, k] += 1.0
                b[i, j, k] -= vy[i, j + 1, k]
            if k > 0 and solid_phi[i, j, k - 1] > 0:
                A_diag[i, j, k] += 1.0
                b[i, j, k] += vz[i, j, k]
            if k < N3 - 1 and solid_phi[i, j, k + 1] > 0:
                A_diag[i, j, k] += 1.0
                b[i, j, k] -= vz[i, j, k + 1]

            b[i, j, k] += volume_correction

            if (phi[i + 1, j, k] > 0 and solid_phi[i + 1, j, k] > 0) or \
               (phi[i - 1, j, k] > 0 and solid_phi[i - 1, j, k] > 0) or \
               (phi[i, j + 1, k] > 0 and solid_phi[i, j + 1, k] > 0) or \
               (phi[i, j - 1, k] > 0 and solid_phi[i, j - 1, k] > 0) or \
               (phi[i, j, k + 1] > 0 and solid_phi[i, j, k + 1] > 0) or \
               (phi[i, j, k - 1] > 0 and solid_phi[i, j, k - 1] > 0):
                b[i, j, k] -= kappa * curvature(i, j, k) * dt / (rho * dx)

            if i < N1 - 1 and phi[i + 1, j, k] <= 0 and solid_phi[i + 1, j, k] > 0:
                A_plus_i[i, j, k] = -1.0
            if j < N2 - 1 and phi[i, j + 1, k] <= 0 and solid_phi[i, j + 1, k] > 0:
                A_plus_j[i, j, k] = -1.0
            if k < N3 - 1 and phi[i, j, k + 1] <= 0 and solid_phi[i, j, k + 1] > 0:
                A_plus_k[i, j, k] = -1.0

            r = ti.Vector([i + 0.5, j + 0.5, k + 0.5]) * dx - rigid.x[None]

            if rigid.interpolate_sdf(ti.Vector([i - 0.5, j + 0.5, k + 0.5]) * dx) <= 0:
                J_trans[i, j, k].x -= dx * dx
                J_rot[i, j, k].z += dx * dx * r.y
                J_rot[i, j, k].y -= dx * dx * r.z
                b[i, j, k] += rigid.velocity_at_point(ti.Vector([i, j + 0.5, k + 0.5]) * dx).x
            if rigid.interpolate_sdf(ti.Vector([i + 1.5, j - 0.5, k + 0.5]) * dx) <= 0:
                J_trans[i, j, k].x += dx * dx
                J_rot[i, j, k].z -= dx * dx * r.y
                J_rot[i, j, k].y += dx * dx * r.z
                b[i, j, k] -= rigid.velocity_at_point(ti.Vector([i + 1, j + 0.5, k + 0.5]) * dx).x
            if rigid.interpolate_sdf(ti.Vector([i + 0.5, j - 0.5, k + 0.5]) * dx) <= 0:
                J_trans[i, j, k].y -= dx * dx
                J_rot[i, j, k].x += dx * dx * r.z
                J_rot[i, j, k].z -= dx * dx * r.x
                b[i, j, k] += rigid.velocity_at_point(ti.Vector([i + 0.5, j, k + 0.5]) * dx).y
            if rigid.interpolate_sdf(ti.Vector([i + 0.5, j + 1.5, k + 0.5]) * dx) <= 0:
                J_trans[i, j, k].y += dx * dx
                J_rot[i, j, k].x -= dx * dx * r.z
                J_rot[i, j, k].z += dx * dx * r.x
                b[i, j, k] -= rigid.velocity_at_point(ti.Vector([i + 0.5, j + 1, k + 0.5]) * dx).y
            if rigid.interpolate_sdf(ti.Vector([i + 0.5, j + 0.5, k - 0.5]) * dx) <= 0:
                J_trans[i, j, k].z -= dx * dx
                J_rot[i, j, k].y += dx * dx * r.x
                J_rot[i, j, k].x -= dx * dx * r.y
                b[i, j, k] += rigid.velocity_at_point(ti.Vector([i + 0.5, j + 0.5, k]) * dx).z
            if rigid.interpolate_sdf(ti.Vector([i + 0.5, j + 0.5, k + 1.5]) * dx) <= 0:
                J_trans[i, j, k].z += dx * dx
                J_rot[i, j, k].y -= dx * dx * r.x
                J_rot[i, j, k].x += dx * dx * r.y
                b[i, j, k] -= rigid.velocity_at_point(ti.Vector([i + 0.5, j + 0.5, k + 1]) * dx).z

@ti.kernel
def init_pressure_zero():
    for i, j, k in ti.ndrange(N1, N2, N3):
        p[i, j, k] = 0.0

@ti.func
def pressure_force_on_rigid(p):
    p_force[None] = ti.Vector([0.0, 0.0, 0.0])
    p_torque[None] = ti.Vector([0.0, 0.0, 0.0])
    for i, j, k in ti.ndrange(N1, N2, N3):
        p_force[None] += J_trans[i, j, k] * p[i, j, k]
        p_torque[None] += J_rot[i, j, k] * p[i, j, k]

@ti.func
def I_inverse_world(r):
    r_cm = rotate(rigid.q[None], r)
    res = rotate_inv(rigid.q[None], rigid.inertia_inv @ r_cm)
    return res

@ti.func
def apply_A(p):
    for i, j, k in ti.ndrange(N1, N2, N3):
        val = A_diag[i, j, k] * p[i, j, k]
        if i < N1 - 1:
            val += A_plus_i[i, j, k] * p[i + 1, j, k]
        if i > 0:
            val += A_plus_i[i - 1, j, k] * p[i - 1, j, k]
        if j < N2 - 1:
            val += A_plus_j[i, j, k] * p[i, j + 1, k]
        if j > 0:
            val += A_plus_j[i, j - 1, k] * p[i, j - 1, k]
        if k < N3 - 1:
            val += A_plus_k[i, j, k] * p[i, j, k + 1]
        if k > 0:
            val += A_plus_k[i, j, k - 1] * p[i, j, k - 1]
        Ap[i, j, k] = val

    pressure_force_on_rigid(p)
    alpha_world = I_inverse_world(p_torque[None])
    a = rigid.m_inv * p_force[None]
    for i, j, k in ti.ndrange(N1, N2, N3):
        Ap[i, j, k] += rho * (a @ J_trans[i, j, k])
        Ap[i, j, k] += rho * (alpha_world @ J_rot[i, j, k])

@ti.kernel
def compute_residual():
    apply_A(p)
    for i, j, k in ti.ndrange(N1, N2, N3):
        r[i, j, k] = b[i, j, k] - Ap[i, j, k]

@ti.kernel
def compute_Ap():
    apply_A(p_cg)

@ti.kernel
def init_search_dirction():
    for i, j, k in ti.ndrange(N1, N2, N3):
        p_cg[i, j, k] = r[i, j, k]

@ti.kernel
def dot_product_kernel(x: ti.template(), y: ti.template()) -> ti.f32:
    sum_val = 0.0
    for i, j, k in ti.ndrange(N1, N2, N3):
        sum_val += x[i, j, k] * y[i, j, k]
    return sum_val

@ti.kernel
def update_pressure(alpha: ti.f32):
    for i, j, k in ti.ndrange(N1, N2, N3):
        p[i, j, k] += alpha * p_cg[i, j, k]

@ti.kernel
def update_residual(alpha: ti.f32):
    for i, j, k in ti.ndrange(N1, N2, N3):
        r[i, j, k] -= alpha * Ap[i, j, k]

@ti.kernel
def update_search_direction(beta: ti.f32):
    for i, j, k in ti.ndrange(N1, N2, N3):
        p_cg[i, j, k] = r[i, j, k] + beta * p_cg[i, j, k]

def CG_solve(max_iters=15):
    calc_volume()
    err[None] = current_volume[None] / correct_volume[None] - 1.0
    err_int[None] += err[None] * dt
    c = -(kp * err[None] + ki * err_int[None]) / (1 + err[None])

    init_pressure_zero()
    build_matrix(c)
    compute_residual()
    init_search_dirction()
    rtr_old = dot_product_kernel(r, r)
    for _ in range(max_iters):
        compute_Ap()
        pAp = dot_product_kernel(p_cg, Ap)
        alpha = rtr_old / (pAp + 1e-9)
        update_pressure(alpha)
        update_residual(alpha)
        rtr_new = dot_product_kernel(r, r)
        # if rtr_new < res * res * 1e-9:
        #     break
        beta = rtr_new / (rtr_old + 1e-9)
        update_search_direction(beta)
        rtr_old = rtr_new

@ti.kernel
def apply_pressure_gradient():
    # extrapolate pressure into neighboring air cells
    # surface tension should not lead to surface expansion
    for i, j, k in ti.ndrange(N1, N2, N3):
        if solid_phi[i, j, k] > 0 and phi[i, j, k] > 0:
            p_sum = 0.0
            count = 0
            if i < N1 - 1 and (phi[i + 1, j, k] <= 0 or solid_phi[i + 1, j, k] > 0):
                p_sum += p[i + 1, j, k]
                count += 1
            if i > 0 and (phi[i - 1, j, k] <= 0 or solid_phi[i - 1, j, k] > 0):
                p_sum += p[i - 1, j, k]
                count += 1
            if j < N2 - 1 and (phi[i, j + 1, k] <= 0 or solid_phi[i, j + 1, k] > 0):
                p_sum += p[i, j + 1, k]
                count += 1
            if j > 0 and (phi[i, j - 1, k] <= 0 or solid_phi[i, j - 1, k] > 0):
                p_sum += p[i, j - 1, k]
                count += 1
            if k < N3 - 1 and (phi[i, j, k + 1] <= 0 or solid_phi[i, j, k + 1] > 0):
                p_sum += p[i, j, k + 1]
                count += 1
            if k > 0 and (phi[i, j, k - 1] <= 0 or solid_phi[i, j, k - 1] > 0):
                p_sum += p[i, j, k - 1]
                count += 1
            if count > 0:
                p[i, j, k] = p_sum / count

    for i, j, k in ti.ndrange(N1 + 1, N2, N3):
        if i > 0 and i < N1:
            if phi[i - 1, j, k] < 0 or phi[i, j, k] < 0:
                if solid_phi[i - 1, j, k] >= 0 and solid_phi[i, j, k] >= 0:
                    vx[i, j, k] -= p[i, j, k] - p[i - 1, j, k]
    for i, j, k in ti.ndrange(N1, N2 + 1, N3):
        if j > 0 and j < N2:
            if phi[i, j - 1, k] < 0 or phi[i, j, k] < 0:
                if solid_phi[i, j - 1, k] >= 0 and solid_phi[i, j, k] >= 0:
                    vy[i, j, k] -= p[i, j, k] - p[i, j - 1, k]
    for i, j, k in ti.ndrange(N1, N2, N3 + 1):
        if k > 0 and k < N3:
            if phi[i, j, k - 1] < 0 or phi[i, j, k] < 0:
                if solid_phi[i, j, k - 1] >= 0 and solid_phi[i, j, k] >= 0:
                    vz[i, j, k] -= p[i, j, k] - p[i, j, k - 1]

def project():
    CG_solve()
    apply_pressure_gradient()

@ti.func
def handle_boundary_collision():
    pos = ti.Vector([0.0, 0.0, 0.0])
    collision_count = 0
    for i in ti.grouped(rigid.vertices):
        vertex_pos = rigid.x[None] + rotate(rigid.q[None], rigid.vertices[i])
        if interpolate_phi(solid_phi, vertex_pos) < 0:
            n = interpolate_phi_grad(solid_phi, vertex_pos)
            if rigid.velocity_at_point(vertex_pos).dot(n) < 0:
                collision_count += 1
                pos += vertex_pos
    if collision_count > 0:
        pos = pos / collision_count
        r = pos - rigid.x[None]
        n = interpolate_phi_grad(solid_phi, pos)
        vp = rigid.velocity_at_point(pos)
        j = -(1 + restitution) * n.dot(vp) / (
            I_inverse_world(r.cross(n)).dot(r.cross(n)) +
            rigid.m_inv
        )
        rigid.v[None] += rigid.m_inv * j * n
        rigid.omega[None] += I_inverse_world(r.cross(n)) * j
        rigid.x[None] -= interpolate_phi(solid_phi, pos) * n

@ti.kernel
def substep_rigid():
    handle_boundary_collision()

    pressure_force_on_rigid(p)
    p_force[None] *= dx * rho / dt
    p_torque[None] *= dx * rho / dt

    rigid.force[None] = ti.Vector([0.0, -g, 0.0]) * rigid.m + p_force[None]
    rigid.torque[None] = p_torque[None]

    rigid.substep(dt)

vx_half = ti.field(dtype=ti.f32, shape=vx.shape)
vy_half = ti.field(dtype=ti.f32, shape=vy.shape)
vz_half = ti.field(dtype=ti.f32, shape=vz.shape)

counter = 0
def substep():
    global counter
    apply_gravity()
    extrapolate()
    advect_levelset()
    if counter % int(0.006 / dt) == 0:
        reinit_levelset()
    if USE_REFLECTION:
        advect(dt / 2)
        vx_half.copy_from(vx_new)
        vy_half.copy_from(vy_new)
        vz_half.copy_from(vz_new)
        project()

        @ti.kernel
        def reflect():
            for i in ti.grouped(vx):
                vx[i] = 2 * vx[i] - vx_half[i]
            for i in ti.grouped(vy):
                vy[i] = 2 * vy[i] - vy_half[i]
            for i in ti.grouped(vz):
                vz[i] = 2 * vz[i] - vz_half[i]
        reflect()
        advect(dt / 2)
    else:
        advect(dt)
        project()
    constrain()

    substep_rigid()   

    counter += 1

@ti.kernel
def add_drop():
    x = 0.5 + (ti.random() * 2 - 1) * 0.3
    y = 0.75
    z = 0.5 + (ti.random() * 2 - 1) * 0.3
    radius = 0.1
    for i, j, k in ti.ndrange(N1, N2, N3):
        r = ti.sqrt(((i + 0.5) * dx - x) ** 2 + ((j + 0.5) * dx - y) ** 2 + ((k + 0.5) * dx - z) ** 2)
        drop_phi = r - radius
        if drop_phi < phi[i, j, k]:
            phi[i, j, k] = drop_phi
        if r < radius:
            vx[i + 1, j, k] = 0.0
            vx[i, j, k] = 0.0
            vy[i, j + 1, k] = 0.0
            vy[i, j, k] = 0.0
            vz[i, j, k + 1] = 0.0
            vz[i, j, k] = 0.0

    delta_V = 4.0 / 3.0 * 3.1415 * radius * radius * radius / (dx * dx * dx)
    correct_volume[None] += delta_V
    current_volume[None] += delta_V

init()
init_volume()

# Save rigid body initial configuration
rigid_config = {
    'obj_path': RIGID_OBJ_PATH,
    'scale': RIGID_SCALE,
    'initial_position': RIGID_INIT_POS,
    'mass': RIGID_MASS,
    'sdf_resolution': max(N1, N2, N3),
    'grid_size': [N1, N2, N3],
    'dx': dx
}
with open('rigid_states/config.json', 'w') as f:
    json.dump(rigid_config, f, indent=4)

# window = ti.ui.Window("3D Fluid Simulation", (800, 800))
# canvas = window.get_canvas()
# canvas.set_background_color((0.2, 0.2, 0.2))
# scene = ti.ui.Scene()
# camera = ti.ui.Camera()

# particles = ti.Vector.field(3, dtype=ti.f32, shape=(N1 * N2 * N3 // 3 + 1))

current_t = 0.0

@ti.kernel
def gaussian_blur(phi_1: ti.template(), phi: ti.template()):
    for i, j, k in ti.ndrange(i, j, k):
        if i > 0 and i < N1 - 1 and j > 0 and j < N2 - 1 and k > 0 and k < N3 - 1:
            phi_1[i, j, k] = (8 * phi[i, j, k] + 4 * (phi[i - 1, j, k] + phi[i + 1, j, k] + phi[i, j - 1, k] + phi[i, j + 1, k] + phi[i, j, k - 1] + phi[i, j, k + 1]) + phi[i - 1, j - 1, k] + phi[i - 1, j + 1, k] + phi[i + 1, j - 1, k] + phi[i + 1, j + 1, k] + phi[i - 1, j, k - 1] + phi[i - 1, j, k + 1] + phi[i + 1, j, k - 1] + phi[i + 1, j, k + 1] + phi[i, j - 1, k - 1] + phi[i, j - 1, k + 1] + phi[i, j + 1, k - 1] + phi[i, j + 1, k + 1]) / 64.0
        else:
            phi_1[i, j, k] = phi[i, j, k]

@ti.kernel
def render():
    for i, j, k in ti.ndrange(N1, N2, N3):
        pos = ti.Vector([i, j, k]) * dx
        index = i * N2 * N3 + j * N3 + k
        if index % 3 == 0:
            particles[index // 3] = pos if (phi[i, j, k] < 0 and solid_phi[i, j, k] > 0) else ti.Vector([3, 3, 3])

def export(count):
    # Export fluid levelset
    np.save(f"levelset/phi_{count:05d}", phi.to_numpy())
    
    # Export rigid body state
    rigid_state = {
        'position': rigid.x[None].to_numpy().tolist(),
        'quaternion': rigid.q[None].to_numpy().tolist()
    }
    with open(f"rigid_states/state_{count:05d}.json", 'w') as f:
        json.dump(rigid_state, f)

frame_count = 0
# while window.running and frame_count < 500:
while frame_count < 500:
    for _ in range(substeps):
        substep()
        if counter % (fps // 2) == 0:
            print("[INFO] Substep completed: ", counter)
        current_t += dt

    # if frame_count % (fps * 5 // 6) == 0 and frame_count > 0:
    #     add_drop()

    print("[INFO] Frame: ", frame_count)
    print("[DEBUG] Volume: ", current_volume[None])
    # render()

    # camera.position(1.8, 1.8, 1.8)
    # camera.lookat(0, 0, 0)
    # scene.set_camera(camera)
    # scene.ambient_light((0.5, 0.5, 0.5))
    # scene.point_light(pos=(3.0, 3.0, 3.0), color=(1.0, 1.0, 1.0))

    # scene.particles(particles, radius=0.005, color=(0.2, 0.2, 0.5))
    # rigid.render(scene)
    
    # canvas.scene(scene)
    # window.show()
    # window.save_image("output/{:05d}.png".format(frame_count))
    export(frame_count)
    frame_count += 1


# frame_count = 0
# while True:
#     for _ in range(substeps):
#         substep()
#     print("Frame:", frame_count)
#     export(frame_count)
#     frame_count += 1
#     if frame_count == 30:
#         break