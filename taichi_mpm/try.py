import taichi as ti
ti.init(arch=ti.gpu)

# ---------------- simulation parameters ----------------
quality      = 1
n_grid       = 256 * quality             # 3-D grid resolution
n_particles  = 2000 * quality**3        # particle count
dim          = 3
dx, inv_dx   = 1 / n_grid, float(n_grid)
dt           = 1e-6 / quality

p_vol        = (dx * 0.5) ** 2           # volume of one particle  (0.5^3 × dx^3)
p_rho        = 1.0
p_mass       = p_vol * p_rho
E            = 10000.0                     # Young’s modulus
gravity_vec  = ti.Vector([0.0, -10.0, 0.0])

# ---------------- taichi fields ----------------
x  = ti.Vector.field(3, ti.f32,  n_particles)     # position
v  = ti.Vector.field(3, ti.f32,  n_particles)     # velocity
C  = ti.Matrix.field(3, 3, ti.f32, n_particles)   # affine velocity (APIC-MPM)
J  = ti.field(ti.f32, n_particles)                # determinant |F| (only scalar)

grid_v = ti.Vector.field(3, ti.f32, (n_grid, n_grid, n_grid))
grid_m = ti.field(ti.f32,               (n_grid, n_grid, n_grid))

# ---------------- weight helper ----------------
@ti.func
def quadratic_weights(fx):
    # returns length-3 list of 3-D vectors
    return [
        0.5 * ti.pow(1.5 - fx, 2),
        0.75 - ti.pow(fx - 1.0, 2),
        0.5 * ti.pow(fx - 0.5, 2)
    ]

# ---------------- core solver ----------------
@ti.kernel
def substep():
    # --- 清空网格
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.Vector.zero(ti.f32, 3)
        grid_m[I] = 0.0

    # -------- P2G  (particle ➜ grid) --------
    for p in x:
        Xp   = x[p] * inv_dx
        base = (Xp - 0.5).cast(int)
        fx   = Xp - base.cast(float)

        w = quadratic_weights(fx)                         # list[3] of vec3

        # 简单各向同性弹性应力（与原示例一致）
        stress  = -dt * 4.0 * E * p_vol * (J[p] - 1.0) / (dx * dx)
        affine  = ti.Matrix.identity(ti.f32, 3) * stress + p_mass * C[p]

        for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
            dpos   = (offset.cast(float) - fx) * dx
            weight = 1.0
            for i in ti.static(range(3)):
                weight *= w[offset[i]][i]                 # 与原算法相同

            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    # -------- 网格操作：归一化 + 重力 + 边界 --------
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:
            grid_v[I] = grid_v[I] / grid_m[I] + gravity_vec * dt

            # 简单吸收式边界
            for d in ti.static(range(3)):
                if (I[d] < 3 and grid_v[I][d] < 0) or \
                   (I[d] > n_grid - 3 and grid_v[I][d] > 0):
                    grid_v[I][d] = 0.0

    # -------- G2P (grid ➜ particle) --------
    for p in x:
        Xp   = x[p] * inv_dx
        base = (Xp - 0.5).cast(int)
        fx   = Xp - base.cast(float)

        w = quadratic_weights(fx)

        new_v = ti.Vector.zero(ti.f32, 3)
        new_C = ti.Matrix.zero(ti.f32, 3, 3)

        for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
            dpos   = (offset.cast(float) - fx) * dx
            weight = 1.0
            for i in ti.static(range(3)):
                weight *= w[offset[i]][i]

            g_v    = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx * inv_dx

        v[p] = new_v
        x[p] += dt * new_v
        J[p] *= 1.0 + dt * new_C.trace()          # 更新体积
        C[p]  = new_C

# ---------------- initialization ----------------
x_min = ti.Vector([0.1, 0.05, 0.3])
x_max = ti.Vector([0.2, 0.15, 0.4])

@ti.kernel
def reset_particles():
    for p in x:
        r    = ti.Vector([ti.random(), ti.random(), ti.random()])
        x[p] = x_min + r * (x_max - x_min)
        v[p] = ti.Vector.zero(ti.f32, 3)
        C[p] = ti.Matrix.zero(ti.f32, 3, 3)
        J[p] = 1.0

# ---------------- gui ----------------
def main():
    reset_particles()

    window = ti.ui.Window("MPM-3D (simple)", (800, 800))
    canvas = window.get_canvas()
    canvas.set_background_color((0.5, 0.5, 0.5))
    scene  = ti.ui.Scene()
    camera = ti.ui.Camera()

    camera.position(-2.0, 0.5, -2.0)
    camera.lookat(0.5, 0.5, 0.5)

    while window.running:
        for _ in range(100):      # smaller CFL → 多跑几步
            substep()

        # draw
        camera.track_user_inputs(window, movement_speed=0.003, hold_key=ti.ui.LMB)
        scene.set_camera(camera)
        scene.point_light((1.5, 1.5, 1.5), (1.1, 1.1, 1.1))
        scene.particles(x, radius=dx * 0.7, color=(0.8, 0.5, 1.0))

        canvas.scene(scene)
        window.show()

if __name__ == "__main__":
    main()
