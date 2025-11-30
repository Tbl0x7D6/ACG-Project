import taichi as ti

# ti.init(arch=ti.cuda, device_memory_fraction=0.95)
ti.init(arch=ti.cuda)

USE_REFLECTION = False

# MAC grid
res = 256
dt = 5e-5
dtau = 0.001
substeps = int(1 / 60 // dt)

dx = 1.0 / res
rho = 1000.0

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

# level set
solid_phi = ti.field(dtype=ti.f32, shape=(res, res))
phi = ti.field(dtype=ti.f32, shape=(res, res))
phi_1 = ti.field(dtype=ti.f32, shape=(res, res))
phi_2 = ti.field(dtype=ti.f32, shape=(res, res))

# volume of fluid
vof = ti.field(dtype=ti.f32, shape=(res, res))
# vof_star = ti.field(dtype=ti.f32, shape=(res, res))
vof_1 = ti.field(dtype=ti.f32, shape=(res, res))

@ti.func
def heaviside(phi_val):
    epsilon = 1.5 * dx
    h = 0.0
    if phi_val < -epsilon:
        h = 1.0
    elif phi_val > epsilon:
        h = 0.0
    else:
        h = 0.5 * (1.0 - phi_val / epsilon)
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
        # r = ((i - res / 4) ** 2 + (j - res / 4) ** 2) ** 0.5 * dx
        # phi[i, j] = r - 0.1
        phi[i, j] = max(i * dx - 0.5, j * dx - 0.9, -(i * dx - 0.2), -(j * dx - 0.2))

    for i, j in ti.ndrange(res, res):
        vof[i, j] = heaviside(phi[i, j])

@ti.kernel
def apply_gravity():
    g = -9.8
    for i, j in ti.ndrange(res, res + 1):
        # if phi[i, j - 1] < 0 or phi[i, j] < 0 and solid_phi[i, j - 1] > 0 and solid_phi[i, j] > 0:
        if vof[i, j] > 1e-6 and solid_phi[i, j - 1] > 0 and solid_phi[i, j] > 0:
            v[i, j] += g * dt

u_valid_old = ti.field(dtype=ti.i32, shape=(res + 1, res))
v_valid_old = ti.field(dtype=ti.i32, shape=(res, res + 1))
u_valid = ti.field(dtype=ti.i32, shape=(res + 1, res))
v_valid = ti.field(dtype=ti.i32, shape=(res, res + 1))

@ti.kernel
def extrapolate_mark_valid():
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
    for _ in range(3):
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

# @ti.func
# def surface_normal(i, j):
#     # Youngs
#     nx = vof[i + 1, j + 1] + 2 * vof[i + 1, j] + vof[i + 1, j - 1] - vof[i - 1, j + 1] - 2 * vof[i - 1, j] - vof[i - 1, j - 1]
#     ny = vof[i + 1, j + 1] + 2 * vof[i, j + 1] + vof[i - 1, j + 1] - vof[i + 1, j - 1] - 2 * vof[i, j - 1] - vof[i - 1, j - 1]
#     n = ti.Vector([nx, ny])
#     return n.normalized()

# @ti.func
# def c_values(i, j, S):
#     # assume nx > ny, 0 - pi/4
#     normal = surface_normal(i, j)
#     y = ti.abs(normal.y)
#     x = ti.abs(normal.x)
#     t = ti.select(x > y, y / x, x / y)
#     S1 = 0.5 * t
#     S2 = 1 - S1
#     c1, c2, c3, c4 = 0.0, 0.0, 0.0, 0.0
#     if S < 1e-6:
#         c1, c2, c3, c4 = 0.0, 0.0, 0.0, 0.0
#     elif S < S1:
#         c1 = ti.sqrt(2 * S / t)
#         c2 = c1 * t
#         c3, c4 = 0.0, 0.0
#     elif S <= S2:
#         c2 = S - S1 + t
#         c4 = S - S1
#         c1, c3 = 1.0, 0.0
#     elif S < 1.0 - 1e-6:
#         c3 = ti.sqrt(2 * (1 - S) / t)
#         c4 = c3 * t
#         c3, c4 = 1.0 - c3, 1.0 - c4
#         c1, c2 = 1.0, 1.0
#     else:
#         c1, c2, c3, c4 = 1.0, 1.0, 1.0, 1.0

#     if 0 < normal.x and 0 < normal.y:
#         if x <= y:
#             c1, c2, c3, c4 = c2, c1, c4, c3
#     elif normal.x <= 0 < normal.y:
#         if x <= y:
#             c1, c2, c3, c4 = c4, c1, c2, c3
#         else:
#             c1, c2, c3, c4 = c3, c2, c1, c4
#     elif normal.x <= 0 and normal.y <= 0:
#         if x <= y:
#             c1, c2, c3, c4 = c3, c4, c1, c2
#         else:
#             c1, c2, c3, c4 = c4, c3, c2, c1
#     else:
#         if x <= y:
#             c1, c2, c3, c4 = c2, c3, c4, c1
#         else:
#             c1, c2, c3, c4 = c1, c4, c3, c2

#     return c1, c2, c3, c4

# @ti.func
# def advect_vof_intermediate():
#     for i, j in ti.ndrange(res, res):
#         if solid_phi[i, j] > 0:
#             u_right = u[i + 1, j]
#             # c_upwind_right = ti.select(u_right > 0.0, vof[i, j], vof[i + 1, j]) if i < res - 1 else vof[i, j]
#             c_upwind_right = ti.select(u_right > 0.0, c_values(i, j, vof[i, j])[0], c_values(i + 1, j, vof[i + 1, j])[2])
#             f_right = u_right * c_upwind_right * dt / dx

#             if solid_phi[i + 1, j] < 0:
#                 f_right = 0.0

#             u_left = u[i, j]
#             # c_upwind_left = ti.select(u_left > 0.0, vof[i - 1, j], vof[i, j]) if i > 0 else vof[i, j]
#             c_upwind_left = ti.select(u_left > 0.0, c_values(i - 1, j, vof[i - 1, j])[0], c_values(i, j, vof[i, j])[2])
#             f_left = u_left * c_upwind_left * dt / dx

#             if solid_phi[i - 1, j] < 0:
#                 f_left = 0.0

#             new_val = vof[i, j] * (1 + (u[i + 1, j] - u[i, j]) * dt / dx) + f_left - f_right
#             vof_star[i, j] = max(0.0, min(1.0, new_val))

@ti.kernel
def advect_vof():
    # advect_vof_intermediate()
    for i, j in ti.ndrange(res, res):
        if solid_phi[i, j] > 0:
            u_right = u[i + 1, j]
            c_upwind_right = ti.select(u_right > 0.0, vof[i, j], vof[i + 1, j]) if i < res - 1 else vof[i, j]
            # c_upwind_right = ti.select(u_right > 0.0, c_values(i, j, vof[i, j])[0], c_values(i + 1, j, vof[i + 1, j])[2])
            f_right = u_right * c_upwind_right * dt / dx

            if solid_phi[i + 1, j] < 0:
                f_right = 0.0

            u_left = u[i, j]
            c_upwind_left = ti.select(u_left > 0.0, vof[i - 1, j], vof[i, j]) if i > 0 else vof[i, j]
            # c_upwind_left = ti.select(u_left > 0.0, c_values(i - 1, j, vof[i - 1, j])[0], c_values(i, j, vof[i, j])[2])
            f_left = u_left * c_upwind_left * dt / dx

            if solid_phi[i - 1, j] < 0:
                f_left = 0.0

            v_up = v[i, j + 1]
            c_upwind_up = ti.select(v_up > 0.0, vof[i, j], vof[i, j + 1]) if j < res - 1 else vof[i, j]
            # c_upwind_up = ti.select(v_up > 0.0, c_values(i, j, vof_star[i, j])[1], c_values(i, j + 1, vof_star[i, j + 1])[3])
            f_up = v_up * c_upwind_up * dt / dx
            
            if solid_phi[i, j + 1] < 0:
                f_up = 0.0

            v_down = v[i, j]
            c_upwind_down = ti.select(v_down > 0.0, vof[i, j - 1], vof[i, j]) if j > 0 else vof[i, j]
            # c_upwind_down = ti.select(v_down > 0.0, c_values(i, j - 1, vof_star[i, j - 1])[1], c_values(i, j, vof_star[i, j])[3])
            f_down = v_down * c_upwind_down * dt / dx
            
            if solid_phi[i, j - 1] < 0:
                f_down = 0.0

            new_val = vof[i, j] + f_left - f_right + f_down - f_up
            vof_1[i, j] = max(0.0, min(1.0, new_val))

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
def clsvof():
    for i, j in ti.ndrange(res, res):
        c_val = vof[i, j]
        p_val = phi[i, j]

        if c_val > 0.01 and c_val < 0.99:
            epsilon = 1.5 * dx

            target_phi = epsilon * (1.0 - 2.0 * c_val)
            
            rate = 0.9  # relaxation factor
            phi[i, j] = p_val * (1 - rate) + target_phi * rate

        elif c_val >= 0.99 and phi[i, j] > 0:
            phi[i, j] = -0.5 * dx
        elif c_val <= 0.01 and phi[i, j] < 0:
            phi[i, j] = 0.5 * dx

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

@ti.kernel
def build_matrix():
    for i, j in ti.ndrange(res, res):
        b[i, j] = 0.0
    for i, j in ti.ndrange(res, res):
        A_diag[i, j] = 0.0
        A_plus_i[i, j] = 0.0
        A_plus_j[i, j] = 0.0
        if phi[i, j] <= 0 and solid_phi[i, j] > 0:
            if i > 0 and solid_phi[i - 1, j] > 0:
                A_diag[i, j] += 1.0
                b[i, j] += u[i, j]
            if i < res - 1 and solid_phi[i + 1, j] > 0:
                A_diag[i, j] += 1.0
                b[i, j] -= u[i + 1, j]
            if j > 0 and solid_phi[i, j - 1] > 0:
                A_diag[i, j] += 1.0
                b[i, j] += v[i, j]
            if j < res - 1 and solid_phi[i, j + 1] > 0:
                A_diag[i, j] += 1.0
                b[i, j] -= v[i, j + 1]

            if i < res - 1 and phi[i + 1, j] <= 0 and solid_phi[i + 1, j] > 0:
                A_plus_i[i, j] = -1.0
            if j < res - 1 and phi[i, j + 1] <= 0 and solid_phi[i, j + 1] > 0:
                A_plus_j[i, j] = -1.0

@ti.kernel
def init_pressure_zero():
    for i, j in ti.ndrange(res, res):
        p[i, j] = 0.0

@ti.kernel
def compute_residual():
    for i, j in ti.ndrange(res, res):
        Ap_val = A_diag[i, j] * p[i, j]
        if i < res - 1:
            Ap_val += A_plus_i[i, j] * p[i + 1, j]
        if i > 0:
            Ap_val += A_plus_i[i - 1, j] * p[i - 1, j]
        if j < res - 1:
            Ap_val += A_plus_j[i, j] * p[i, j + 1]
        if j > 0:
            Ap_val += A_plus_j[i, j - 1] * p[i, j - 1]
        r[i, j] = b[i, j] - Ap_val

@ti.kernel
def compute_Ap():
    for i, j in ti.ndrange(res, res):
        Ap_val = A_diag[i, j] * p_cg[i, j]
        if i < res - 1:
            Ap_val += A_plus_i[i, j] * p_cg[i + 1, j]
        if i > 0:
            Ap_val += A_plus_i[i - 1, j] * p_cg[i - 1, j]
        if j < res - 1:
            Ap_val += A_plus_j[i, j] * p_cg[i, j + 1]
        if j > 0:
            Ap_val += A_plus_j[i, j - 1] * p_cg[i, j - 1]
        Ap[i, j] = Ap_val

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
        if phi[i, j] <= 0 and solid_phi[i, j] > 0:
            p[i, j] += alpha * p_cg[i, j]

@ti.kernel
def update_residual(alpha: ti.f32):
    for i, j in ti.ndrange(res, res):
        if phi[i, j] <= 0 and solid_phi[i, j] > 0:
            r[i, j] -= alpha * Ap[i, j]

@ti.kernel
def update_search_direction(beta: ti.f32):
    for i, j in ti.ndrange(res, res):
        if phi[i, j] <= 0 and solid_phi[i, j] > 0:
            p_cg[i, j] = r[i, j] + beta * p_cg[i, j]

def CG_solve(max_iters=15):
    init_pressure_zero()
    build_matrix()
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
    # TODO: surface tension
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
    count = 0.0
    for i, j in ti.ndrange(res, res):
        count += vof[i, j]
    return count

def project():
    CG_solve()
    apply_pressure_gradient()

u_half = ti.field(dtype=ti.f32, shape=(res + 1, res))
v_half = ti.field(dtype=ti.f32, shape=(res, res + 1))

counter = 0
def substep():
    global counter
    apply_gravity()
    extrapolate()
    advect_vof()
    advect_levelset()
    phi.copy_from(phi_1)
    vof.copy_from(vof_1)
    clsvof()
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
    counter += 1

@ti.kernel
def add_drop():
    x = 0.5 + (ti.random() * 2 - 1) * 0.3
    y = 0.5
    for i, j in ti.ndrange(res, res):
        r = ti.sqrt(((i + 0.5) * dx - x) ** 2 + ((j + 0.5) * dx - y) ** 2)
        drop_phi = r - 0.05
        if drop_phi < phi[i, j]:
            phi[i, j] = drop_phi
            vof[i, j] = heaviside(phi[i, j])
        if r < 0.05:
            u[i + 1, j] = 0.0
            u[i, j] = 0.0
            v[i, j + 1] = 0.0
            v[i, j] = 0.0

init()

window = ti.ui.Window("2D Fluid Simulation", (res, res))
canvas = window.get_canvas()
pixels = ti.field(dtype=ti.f32, shape=(res, res, 3))

@ti.kernel
def gaussian_blur(phi_1: ti.template(), phi: ti.template()):
    for i, j in ti.ndrange(res, res):
        if i > 0 and i < res - 1 and j > 0 and j < res - 1:
            phi_1[i, j] = (4 * phi[i, j] + 2 * (phi[i - 1, j] + phi[i + 1, j] + phi[i, j - 1] + phi[i, j + 1]) + phi[i - 1, j - 1] + phi[i - 1, j + 1] + phi[i + 1, j - 1] + phi[i + 1, j + 1]) / 16.0

@ti.kernel
def render():
    for i, j in ti.ndrange(res, res):
        pixels[i, j, 0] = 1.0 if phi_1[i, j] < 0 else 0.0
        pixels[i, j, 1] = 1.0 if phi_1[i, j] < 0 else 0.0
        pixels[i, j, 2] = 1.0 if phi_1[i, j] < 0 else 0.0
        
        if solid_phi[i, j] < 0:
            pixels[i, j, 0] = 0.2
            pixels[i, j, 1] = 0.2
            pixels[i, j, 2] = 0.8

frame_count = 0
while window.running:
    if frame_count == 900:
        break
    for _ in range(substeps):
        substep()
    # if frame_count % 90 == 0 and frame_count != 0:
    #     add_drop()
    print("Volume:", volume())
    gaussian_blur(phi_1, phi)
    render()
    canvas.set_image(pixels)
    window.save_image("output/{:05d}.png".format(frame_count))
    window.show()
    frame_count += 1