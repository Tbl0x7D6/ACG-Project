import taichi as ti
from pathlib import Path

# ti.init(arch=ti.cuda, device_memory_fraction=0.95)
ti.init(arch=ti.cuda)

USE_REFLECTION = False

if Path("output").exists():
    import shutil
    shutil.rmtree("output")
Path("output").mkdir(parents=True, exist_ok=True)

# MAC grid
res = 512
dt = 5e-4
dtau = 0.001
substeps = int(1 / 60 // dt)

dx = 1.0 / res
rho = 1000.0
kappa = 0.3

u = ti.field(dtype=ti.f32, shape=(res + 1, res))
v = ti.field(dtype=ti.f32, shape=(res, res + 1))
u_new = ti.field(dtype=ti.f32, shape=(res + 1, res))
v_new = ti.field(dtype=ti.f32, shape=(res, res + 1))

p = ti.field(dtype=ti.f32, shape=(res, res))
r = ti.field(dtype=ti.f32, shape=(res, res))
p_cg = ti.field(dtype=ti.f32, shape=(res, res))
Ap = ti.field(dtype=ti.f32, shape=(res, res))

A_diag = ti.field(dtype=ti.f32, shape=(res, res))
A_plus_i = ti.field(dtype=ti.f32, shape=(res, res))
A_plus_j = ti.field(dtype=ti.f32, shape=(res, res))
b = ti.field(dtype=ti.f32, shape=(res, res))

Jx = ti.field(dtype=ti.f32, shape=(res, res))
Jy = ti.field(dtype=ti.f32, shape=(res, res))

ball_center = ti.Vector.field(2, dtype=ti.f32, shape=())
ball_radius = 0.1
ball_v = ti.Vector.field(2, dtype=ti.f32, shape=())
ball_m = 1.0

@ti.func
def ball_phi(i, j):
    x, y = (i + 0.5) * dx, (j + 0.5) * dx
    return ti.sqrt((x - ball_center[None].x)**2 + (y - ball_center[None].y)**2) - ball_radius

# level set
solid_phi = ti.field(dtype=ti.f32, shape=(res, res))
phi = ti.field(dtype=ti.f32, shape=(res, res))
phi_1 = ti.field(dtype=ti.f32, shape=(res, res))
phi_2 = ti.field(dtype=ti.f32, shape=(res, res))

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
    for i, j in ti.ndrange(res, res):
        count += heaviside(phi[i, j])
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
def interpolate_phi(phi, x, y):
    # MUST use sharp cubic interpolation to avoid mass loss
    i = ti.cast(x / dx - 0.5, ti.i32)
    j = ti.cast(y / dx - 0.5, ti.i32)

    i = ti.max(1, ti.min(res - 2, i))
    j = ti.max(1, ti.min(res - 2, j))

    fx = (x / dx - 0.5) - i
    fy = (y / dx - 0.5) - j
    fx = ti.max(0.0, ti.min(1.0, fx))
    fy = ti.max(0.0, ti.min(1.0, fy))

    col0 = cubic_interp(phi[i - 1, j - 1], phi[i, j - 1], phi[i + 1, j - 1], phi[i + 2, j - 1], fx)
    col1 = cubic_interp(phi[i - 1, j    ], phi[i, j    ], phi[i + 1, j    ], phi[i + 2, j    ], fx)
    col2 = cubic_interp(phi[i - 1, j + 1], phi[i, j + 1], phi[i + 1, j + 1], phi[i + 2, j + 1], fx)
    col3 = cubic_interp(phi[i - 1, j + 2], phi[i, j + 2], phi[i + 1, j + 2], phi[i + 2, j + 2], fx)

    return cubic_interp(col0, col1, col2, col3, fy)

@ti.func
def interpolate_solid_phi(x, y):
    i = ti.cast(x / dx - 0.5, ti.i32)
    j = ti.cast(y / dx - 0.5, ti.i32)

    i = ti.max(0, ti.min(res - 1, i))
    j = ti.max(0, ti.min(res - 1, j))

    fx = (x / dx - 0.5) - i
    fy = (y / dx - 0.5) - j
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

    for i, j in ti.ndrange(res, res):
        if i < res // 16 or i > res * 15 / 16 or j < res // 16 or j > res * 15 / 16:
            solid_phi[i, j] = -1.0
        else:
            solid_phi[i, j] = 1.0

    for i, j in ti.ndrange(res, res):
        # Example 1
        # phi[i, j] = max(i * dx - 0.6, j * dx - 0.8, -(i * dx - 0.2), -(j * dx - 0.1))
        # Example 2
        phi[i, j] = max(j * dx - 0.5, -(i * dx - 1.0/16), -(j * dx - 1.0/16), i * dx - 15.0/16)

    # Example 1
    # ball_center[None] = ti.Vector([0.75, 0.4])
    # Example 2
    ball_center[None] = ti.Vector([0.65, 0.8])
    ball_v[None] = ti.Vector([0.0, 0.0])

def init_volume():
    calc_volume()
    correct_volume[None] = current_volume[None]
    err[None] = 0.0
    err_int[None] = 0.0

@ti.kernel
def apply_gravity():
    g = -9.8
    for i, j in ti.ndrange(res, res + 1):
        if phi[i, j - 1] < 0 or phi[i, j] < 0 and solid_phi[i, j - 1] > 0 and solid_phi[i, j] > 0 and ball_phi(i, j - 1) > 0 and ball_phi(i, j) > 0:
            v[i, j] += g * dt

u_valid_old = ti.field(dtype=ti.i32, shape=(res + 1, res))
v_valid_old = ti.field(dtype=ti.i32, shape=(res, res + 1))
u_valid = ti.field(dtype=ti.i32, shape=(res + 1, res))
v_valid = ti.field(dtype=ti.i32, shape=(res, res + 1))

@ti.kernel
def extrapolate_mark_valid():
    for i, j in ti.ndrange(res + 1, res):
        if (i < res and solid_phi[i, j] > 0 and phi[i, j] < 0 and ball_phi(i, j) > 0) or (i > 0 and solid_phi[i - 1, j] > 0 and phi[i - 1, j] < 0 and ball_phi(i - 1, j) > 0):
            u_valid[i, j] = 1
        else:
            u_valid[i, j] = 0
    for i, j in ti.ndrange(res, res + 1):
        if (j < res and solid_phi[i, j] > 0 and phi[i, j] < 0 and ball_phi(i, j) > 0) or (j > 0 and solid_phi[i, j - 1] > 0 and phi[i, j - 1] < 0 and ball_phi(i, j - 1) > 0):
            v_valid[i, j] = 1
        else:
            v_valid[i, j] = 0
    # for i, j in ti.ndrange(res + 1, res):
    #     if ball_phi(i, j) < 0:
    #         u_valid[i, j] = 1
    #         u[i, j] = ball_v[None].x
    # for i, j in ti.ndrange(res, res + 1):
    #     if ball_phi(i, j) < 0:
    #         v_valid[i, j] = 1
    #         v[i, j] = ball_v[None].y

    for i in ti.grouped(u_new):
        u_new[i] = u[i]
    for i in ti.grouped(v_new):
        v_new[i] = v[i]

@ti.kernel
def extrapolate_iter():
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

def extrapolate():
    extrapolate_mark_valid()
    for _ in range(10):
        extrapolate_iter()

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
def advect_levelset():
    for i, j in ti.ndrange(res, res):
        pos = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx])
        back_pos = rk2_trace(pos.x, pos.y, -dt)
        phi_1[i, j] = interpolate_phi(phi, back_pos.x, back_pos.y)
    # for i, j in ti.ndrange(res, res):
    #     pos = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx])
    #     fwd_pos = rk2_trace(pos.x, pos.y, dt)
    #     phi_2[i, j] = interpolate_phi(phi_1, fwd_pos.x, fwd_pos.y)
    # for i, j in ti.ndrange(res, res):
    #     phi[i, j] = phi_1[i, j] + 0.5 * (phi[i, j] - phi_2[i, j])
    for i, j in ti.ndrange(res, res):
        phi[i, j] = phi_1[i, j]

@ti.kernel
def reinit_levelset_iter():
    for i, j in ti.ndrange(res, res):
        if i == 0 or i == res - 1 or j == 0 or j == res - 1:
            if i == 0:
                phi_2[i, j] = phi_1[i + 1, j]
            elif i == res - 1:
                phi_2[i, j] = phi_1[i - 1, j]
            elif j == 0:
                phi_2[i, j] = phi_1[i, j + 1]
            elif j == res - 1:
                phi_2[i, j] = phi_1[i, j - 1]
        else:
            phi_0 = phi[i, j]
            s = phi_0 / ti.sqrt(phi_0*phi_0 + dx*dx)
            # D_x^-
            dx_minus = (phi_1[i, j] - phi_1[i - 1, j]) / dx
            # D_x^+
            dx_plus  = (phi_1[i + 1, j] - phi_1[i, j]) / dx
            # D_y^-
            dy_minus = (phi_1[i, j] - phi_1[i, j - 1]) / dx
            # D_y^+
            dy_plus  = (phi_1[i, j + 1] - phi_1[i, j]) / dx

            grad_sq_x = 0.0
            grad_sq_y = 0.0

            if phi_0 > 0:
                term_xm = ti.max(dx_minus, 0.0)**2
                term_xp = ti.min(dx_plus,  0.0)**2
                grad_sq_x = ti.max(term_xm, term_xp)

                term_ym = ti.max(dy_minus, 0.0)**2
                term_yp = ti.min(dy_plus,  0.0)**2
                grad_sq_y = ti.max(term_ym, term_yp)
            else:
                term_xm = ti.min(dx_minus, 0.0)**2
                term_xp = ti.max(dx_plus,  0.0)**2
                grad_sq_x = ti.max(term_xm, term_xp)

                term_ym = ti.min(dy_minus, 0.0)**2
                term_yp = ti.max(dy_plus,  0.0)**2
                grad_sq_y = ti.max(term_ym, term_yp)

            grad_norm = ti.sqrt(grad_sq_x + grad_sq_y)

            phi_2[i, j] = phi_1[i, j] - dtau * s * (grad_norm - 1.0)

def reinit_levelset():
    phi_1.copy_from(phi)
    for _ in range(30):
        reinit_levelset_iter()
        phi_1.copy_from(phi_2)
    phi.copy_from(phi_1)

@ti.func
def curvature(i, j):
    return phi[i + 1, j] + phi[i - 1, j] + phi[i, j + 1] + phi[i, j - 1] - 4 * phi[i, j]

@ti.kernel
def advect(dt: float):
    for i, j in ti.ndrange(res + 1, res):
        x = i * dx
        y = (j + 0.5) * dx

        back = rk2_trace(x, y, -dt)
        # pos_aux = rk2_trace(back.x, back.y, dt)
        # back = back + 0.5 * (ti.Vector([x, y]) - pos_aux)
        u_new[i, j] = sample_u(back.x, back.y)

    for i, j in ti.ndrange(res, res + 1):
        x = (i + 0.5) * dx
        y = j * dx

        back = rk2_trace(x, y, -dt)
        # pos_aux = rk2_trace(back.x, back.y, dt)
        # back = back + 0.5 * (ti.Vector([x, y]) - pos_aux)
        v_new[i, j] = sample_v(back.x, back.y)

    for i, j in ti.ndrange(res + 1, res):
        u[i, j] = u_new[i, j]
    for i, j in ti.ndrange(res, res + 1):
        v[i, j] = v_new[i, j]

@ti.kernel
def constrain():
    for i, j in ti.ndrange(res, res):
        if i < res - 1 and solid_phi[i + 1, j] < 0 and u[i + 1, j] > 0:
            u[i + 1, j] = 0.0
        if i > 0 and solid_phi[i - 1, j] < 0 and u[i, j] < 0:
            u[i, j] = 0.0
        if j < res - 1 and solid_phi[i, j + 1] < 0 and v[i, j + 1] > 0:
            v[i, j + 1] = 0.0
        if j > 0 and solid_phi[i, j - 1] < 0 and v[i, j] < 0:
            v[i, j] = 0.0
        if i < res - 1 and ball_phi(i + 1, j) < 0 and u[i + 1, j] > ball_v[None].x:
            u[i + 1, j] = ball_v[None].x
        if i > 0 and ball_phi(i - 1, j) < 0 and u[i, j] < ball_v[None].x:
            u[i, j] = ball_v[None].x
        if j < res - 1 and ball_phi(i, j + 1) < 0 and v[i, j + 1] > ball_v[None].y:
            v[i + 1, j] = ball_v[None].y
        if j > 0 and ball_phi(i, j - 1) < 0 and v[i, j] < ball_v[None].y:
            v[i, j] = ball_v[None].y

@ti.kernel
def build_matrix(volume_correction: ti.types.f32):
    for i, j in ti.ndrange(res, res):
        b[i, j] = 0.0
    for i, j in ti.ndrange(res, res):
        A_diag[i, j] = 0.0
        A_plus_i[i, j] = 0.0
        A_plus_j[i, j] = 0.0
        Jx[i, j] = 0.0
        Jy[i, j] = 0.0
        # totally a piece of SHIT
        if phi[i, j] <= 0 and solid_phi[i, j] > 0 and ball_phi(i, j) > 0:
            if i > 0 and solid_phi[i - 1, j] > 0 and ball_phi(i - 1, j) > 0:
                A_diag[i, j] += 1.0
                b[i, j] += u[i, j]
            if i < res - 1 and solid_phi[i + 1, j] > 0 and ball_phi(i + 1, j) > 0:
                A_diag[i, j] += 1.0
                b[i, j] -= u[i + 1, j]
            if j > 0 and solid_phi[i, j - 1] > 0 and ball_phi(i, j - 1) > 0:
                A_diag[i, j] += 1.0
                b[i, j] += v[i, j]
            if j < res - 1 and solid_phi[i, j + 1] > 0 and ball_phi(i, j + 1) > 0:
                A_diag[i, j] += 1.0
                b[i, j] -= v[i, j + 1]

            b[i, j] += volume_correction

            # surface tension
            if (phi[i + 1, j] > 0 and solid_phi[i + 1, j] > 0) or \
               (phi[i - 1, j] > 0 and solid_phi[i - 1, j] > 0) or \
               (phi[i, j + 1] > 0 and solid_phi[i, j + 1] > 0) or \
               (phi[i, j - 1] > 0 and solid_phi[i, j - 1] > 0):
                b[i, j] -= (kappa / dx) * curvature(i, j)

            if i < res - 1 and phi[i + 1, j] <= 0 and solid_phi[i + 1, j] > 0 and ball_phi(i + 1, j) > 0:
                A_plus_i[i, j] = -1.0
            if j < res - 1 and phi[i, j + 1] <= 0 and solid_phi[i, j + 1] > 0 and ball_phi(i, j + 1) > 0:
                A_plus_j[i, j] = -1.0

            if ball_phi(i - 1, j) <= 0:
                Jx[i, j] -= dx
                b[i, j] += ball_v[None].x
            if ball_phi(i + 1, j) <= 0:
                Jx[i, j] += dx
                b[i, j] -= ball_v[None].x
            if ball_phi(i, j - 1) <= 0:
                Jy[i, j] -= dx
                b[i, j] += ball_v[None].y
            if ball_phi(i, j + 1) <= 0:
                Jy[i, j] += dx
                b[i, j] -= ball_v[None].y

@ti.kernel
def init_pressure_zero():
    for i, j in ti.ndrange(res, res):
        p[i, j] = 0.0

fx = ti.field(dtype=ti.f32, shape=())
fy = ti.field(dtype=ti.f32, shape=())

@ti.func
def compute_buoyancy(p):
    fx[None] = 0.0
    fy[None] = 0.0
    for i, j in ti.ndrange(res, res):
        fx[None] += Jx[i, j] * p[i, j]
        fy[None] += Jy[i, j] * p[i, j]

@ti.func
def apply_A(p):
    for i, j in ti.ndrange(res, res):
        val = A_diag[i, j] * p[i, j]
        if i < res - 1:
            val += A_plus_i[i, j] * p[i + 1, j]
        if i > 0:
            val += A_plus_i[i - 1, j] * p[i - 1, j]
        if j < res - 1:
            val += A_plus_j[i, j] * p[i, j + 1]
        if j > 0:
            val += A_plus_j[i, j - 1] * p[i, j - 1]
        Ap[i, j] = val

    compute_buoyancy(p)
    for i, j in ti.ndrange(res, res):
        Ap[i, j] += rho * (Jx[i, j] * fx[None] + Jy[i, j] * fy[None]) / ball_m

@ti.kernel
def compute_residual():
    apply_A(p)
    for i, j in ti.ndrange(res, res):
        r[i, j] = b[i, j] - Ap[i, j]

@ti.kernel
def compute_Ap():
    apply_A(p_cg)

@ti.kernel
def init_search_dirction():
    for i, j in ti.ndrange(res, res):
        p_cg[i, j] = r[i, j]

@ti.kernel
def dot_product_kernel(x: ti.template(), y: ti.template()) -> ti.f32:
    sum_val = 0.0
    for i, j in ti.ndrange(res, res):
        sum_val += x[i, j] * y[i, j]
    return sum_val

@ti.kernel
def update_pressure(alpha: ti.f32):
    for i, j in ti.ndrange(res, res):
        p[i, j] += alpha * p_cg[i, j]

@ti.kernel
def update_residual(alpha: ti.f32):
    for i, j in ti.ndrange(res, res):
        r[i, j] -= alpha * Ap[i, j]

@ti.kernel
def update_search_direction(beta: ti.f32):
    for i, j in ti.ndrange(res, res):
        p_cg[i, j] = r[i, j] + beta * p_cg[i, j]

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
        if rtr_new < res * res * 1e-9:
            break
        beta = rtr_new / (rtr_old + 1e-9)
        update_search_direction(beta)
        rtr_old = rtr_new

@ti.kernel
def apply_pressure_gradient():
    # extrapolate pressure into neighboring air cells
    # surface tension should not lead to surface expansion
    for i, j in ti.ndrange(res, res):
        if solid_phi[i, j] > 0 and phi[i, j] > 0:
            p_sum = 0.0
            count = 0
            if i < res - 1 and (phi[i + 1, j] <= 0 or solid_phi[i + 1, j] > 0):
                p_sum += p[i + 1, j]
                count += 1
            if i > 0 and (phi[i - 1, j] <= 0 or solid_phi[i - 1, j] > 0):
                p_sum += p[i - 1, j]
                count += 1
            if j < res - 1 and (phi[i, j + 1] <= 0 or solid_phi[i, j + 1] > 0):
                p_sum += p[i, j + 1]
                count += 1
            if j > 0 and (phi[i, j - 1] <= 0 or solid_phi[i, j - 1] > 0):
                p_sum += p[i, j - 1]
                count += 1
            if count > 0:
                p[i, j] = p_sum / count

    for i, j in ti.ndrange(res + 1, res):
        if i > 0 and i < res:
            if phi[i - 1, j] < 0 or phi[i, j] < 0:
                if solid_phi[i - 1, j] >= 0 and solid_phi[i, j] >= 0:
                    u[i, j] -= p[i, j] - p[i - 1, j]
    for i, j in ti.ndrange(res, res + 1):
        if j > 0 and j < res:
            if phi[i, j - 1] < 0 or phi[i, j] < 0:
                if solid_phi[i, j - 1] >= 0 and solid_phi[i, j] >= 0:
                    v[i, j] -= p[i, j] - p[i, j - 1]

@ti.kernel
def volume() -> int:
    count = 0
    for i, j in ti.ndrange(res, res):
        if phi[i, j] < 0 and solid_phi[i, j] > 0 and ball_phi(i, j) > 0:
            count += 1
    return count

def project():
    CG_solve(30)
    apply_pressure_gradient()

@ti.kernel
def substep_ball():
    compute_buoyancy(p)

    fx[None] *= dx * rho / dt
    fy[None] *= dx * rho / dt

    ball_v[None] += ti.Vector([fx[None], fy[None]]) / ball_m * dt + ti.Vector([0.0, -9.8]) * dt
    ball_center[None] += ball_v[None] * dt

    min_dist_boundary = ball_radius + dx * (res / 16 + 2)
    if ball_center[None].x < min_dist_boundary or ball_center[None].x > 1.0 - min_dist_boundary:
        ball_v[None].x = - ball_v[None].x
        if ball_center[None].x < min_dist_boundary:
            ball_center[None].x = min_dist_boundary
        else:
            ball_center[None].x = 1.0 - min_dist_boundary
    if ball_center[None].y < min_dist_boundary or ball_center[None].y > 1.0 - min_dist_boundary:
        ball_v[None].y = - ball_v[None].y
        if ball_center[None].y < min_dist_boundary:
            ball_center[None].y = min_dist_boundary
        else:
            ball_center[None].y = 1.0 - min_dist_boundary

u_half = ti.field(dtype=ti.f32, shape=(res + 1, res))
v_half = ti.field(dtype=ti.f32, shape=(res, res + 1))

counter = 0
def substep():
    global counter
    apply_gravity()
    extrapolate()
    advect_levelset()
    if counter % int(0.003 / dt) == 0:
        reinit_levelset()
    if USE_REFLECTION:
        advect(dt / 2)
        u_half.copy_from(u_new)
        v_half.copy_from(v_new)
        project()

        @ti.kernel
        def reflect():
            for i in ti.grouped(u):
                u[i] = 2 * u[i] - u_half[i]
            for i in ti.grouped(v):
                v[i] = 2 * v[i] - v_half[i]
        reflect()
        advect(dt / 2)
    else:
        advect(dt)
        project()
    constrain()

    substep_ball()

    counter += 1

init()
init_volume()

window = ti.ui.Window("2D Fluid Simulation", (512, 512))
canvas = window.get_canvas()
pixels = ti.field(dtype=ti.f32, shape=(res, res, 3))

@ti.kernel
def render():
    for i, j in ti.ndrange(res, res):
        pixels[i, j, 0] = 0.0
        pixels[i, j, 1] = 0.0
        pixels[i, j, 2] = 0.0

        if phi[i, j] < 0:
            pixels[i, j, 0] = 1.0
            pixels[i, j, 1] = 1.0
            pixels[i, j, 2] = 1.0

        if ball_phi(i, j) < 0.002:
            pixels[i, j, 0] = 0.8
            pixels[i, j, 1] = 0.2
            pixels[i, j, 2] = 0.2

        if solid_phi[i, j] < 0:
            pixels[i, j, 0] = 0.2
            pixels[i, j, 1] = 0.2
            pixels[i, j, 2] = 0.8

frame_count = 0
while window.running:
    for _ in range(substeps):
        substep()
    print("Frame:", frame_count, "Volume:", volume())
    render()
    canvas.set_image(pixels)
    window.save_image("output/{:05d}.png".format(frame_count))
    window.show()
    frame_count += 1