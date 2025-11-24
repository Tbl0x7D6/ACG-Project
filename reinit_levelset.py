import taichi as ti

ti.init(arch=ti.cuda)

res = 512
dx = 1.0 / res
dtau = 0.001

phi = ti.field(dtype=ti.f32, shape=(res, res))
phi_1 = ti.field(dtype=ti.f32, shape=(res, res))
phi_2 = ti.field(dtype=ti.f32, shape=(res, res))

@ti.kernel
def init():
    for i, j in ti.ndrange(res, res):
        x = (i + 0.5) * dx
        y = (j + 0.5) * dx
        center = ti.Vector([0.5, 0.5])
        radius = 0.2
        # r = ti.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        phi[i, j] = (x - center[0])**2 + (y - center[1])**2 - radius**2

@ti.kernel
def reinit_iter():
    for i, j in ti.ndrange(res, res):
        # 简单的边界处理：保持边界数值不变（实际应用中建议做外推）
        if i == 0 or i == res - 1 or j == 0 or j == res - 1:
            # copy inner value
            if i == 0:
                phi_2[i, j] = phi_1[i + 1, j]
            elif i == res - 1:
                phi_2[i, j] = phi_1[i - 1, j]
            elif j == 0:
                phi_2[i, j] = phi_1[i, j + 1]
            elif j == res - 1:
                phi_2[i, j] = phi_1[i, j - 1]
        else:
            # 1. 获取原始 LevelSet 的符号 (用于固定界面)
            # 使用平滑符号函数，防止数值振荡
            phi_0 = phi[i, j]
            s = phi_0 / ti.sqrt(phi_0*phi_0 + dx * dx)

            # 2. 计算四个方向的单边导数 (Forward / Backward)
            # D_x^-
            dx_minus = (phi_1[i, j] - phi_1[i - 1, j]) / dx
            # D_x^+
            dx_plus  = (phi_1[i + 1, j] - phi_1[i, j]) / dx
            # D_y^-
            dy_minus = (phi_1[i, j] - phi_1[i, j - 1]) / dx
            # D_y^+
            dy_plus  = (phi_1[i, j + 1] - phi_1[i, j]) / dx

            # 3. Godunov Hamiltonian 格式
            grad_sq_x = 0.0
            grad_sq_y = 0.0

            if phi_0 > 0:
                # 外部：我们要找"上坡"来自哪里，需要 max(D-, 0) 和 min(D+, 0)
                # 取平方后也就是比较谁的幅度大
                term_xm = ti.max(dx_minus, 0.0)**2
                term_xp = ti.min(dx_plus,  0.0)**2
                grad_sq_x = ti.max(term_xm, term_xp)

                term_ym = ti.max(dy_minus, 0.0)**2
                term_yp = ti.min(dy_plus,  0.0)**2
                grad_sq_y = ti.max(term_ym, term_yp)
            
            else:
                # 内部 (phi < 0)：逻辑翻转
                # 我们需要 min(D-, 0) 和 max(D+, 0)
                # 这里的物理含义是：如果在内部，信息应该从"更接近0"的地方传过来
                term_xm = ti.min(dx_minus, 0.0)**2
                term_xp = ti.max(dx_plus,  0.0)**2
                grad_sq_x = ti.max(term_xm, term_xp) # 注意：这里依然是取 Max 幅度！

                term_ym = ti.min(dy_minus, 0.0)**2
                term_yp = ti.max(dy_plus,  0.0)**2
                grad_sq_y = ti.max(term_ym, term_yp) # 注意：这里依然是取 Max 幅度！

            grad_norm = ti.sqrt(grad_sq_x + grad_sq_y)
            
            # 4. 时间推进
            phi_2[i, j] = phi_1[i, j] - dtau * s * (grad_norm - 1.0)

    for i, j in ti.ndrange(res, res):
        phi_1[i, j] = phi_2[i, j]

window = ti.ui.Window("SDF", (512, 512))
canvas = window.get_canvas()
pixels = ti.field(dtype=ti.f32, shape=(res, res, 3))

init()

@ti.kernel
def render():
    for i, j in ti.ndrange(res, res):
        t = ti.abs(phi_1[i, j])
        pixels[i, j, 0] = t * 2.0
        pixels[i, j, 1] = t * 2.0
        pixels[i, j, 2] = t * 2.0
        
def reinit():
    phi_1.copy_from(phi)
    for _ in range(3):
        reinit_iter()
    phi.copy_from(phi_1)

phi_1.copy_from(phi)

frame = 0
while window.running:
    render()
    canvas.set_image(pixels)
    window.show()
    reinit_iter()
    frame += 1
    if frame % 60 == 0:
        print(f"Reinitialization step: {frame // 60}")