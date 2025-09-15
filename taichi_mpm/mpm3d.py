import taichi as ti
ti.init(arch=ti.gpu)  # Try to run on GPU

quality = 1
n_particles, n_grid = 60000 * quality**3, 256 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5) ** 2, 1
p_mass = p_vol * p_rho
E, nu = 400, 0.4  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

x = ti.Vector.field(3, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(3, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)  # deformation gradient
material = ti.field(dtype=int, shape=n_particles)  # material id
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
grid_v = ti.Vector.field(3, dtype=float, shape=(n_grid, n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid))  # grid node mass
gravity = ti.Vector.field(3, dtype=float, shape=())

num_group = 3
x_min = ti.Vector.field(3, ti.f32, shape=())
x_max = ti.Vector.field(3, ti.f32, shape=())
x_min[None] = [0.1, 0.5, 0.3]          # 你想要的下界
x_max[None] = [0.8, 0.9, 0.95]         # 你想要的上界

@ti.kernel
def substep():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0, 0, 0]
        grid_m[i, j, k] = 0
    for p in range(n_particles):
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        #Quadratic kernels
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # deformation gradient update
        F[p] = (ti.Matrix.identity(float, 3) + dt * C[p]) @ F[p]
        #h = ti.max(0.1, ti.min(5, ti.exp(10 * (1.0 - Jp[p]))))
        h = 1
        mu, la = mu_0 * h, lambda_0 * h
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(3)):
            new_sig = sig[d, d]
            J *= new_sig
        # stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 3) * la * J * (
        #     J - 1
        # )
        #stress = (-dt * p_vol * 8 * inv_dx * inv_dx * inv_dx) * stress
        stress = (-dt * p_vol * 4 * E * inv_dx * inv_dx) * (J - 1)
        #stress = (-dt * p_vol * 8 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * dx

            # wx = [0.5*(1.5 - fx[0])**2,
            #     0.75-(fx[0]-1)**2,
            #     0.5*(fx[0]-0.5)**2]

            # wy = [0.5*(1.5 - fx[1])**2,
            #     0.75-(fx[1]-1)**2,
            #     0.5*(fx[1]-0.5)**2]

            # wz = [0.5*(1.5 - fx[2])**2,
            #     0.75-(fx[2]-1)**2,
            #     0.5*(fx[2]-0.5)**2]

            # P2G / G2P 内部
            # weight = wx[i] * wy[j] * wz[k]

            weight = w[i][0] * w[j][1] * w[k][2]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            grid_v[i, j, k] /= grid_m[i, j, k]
            grid_v[i, j, k] += gravity[None] * dt
            if i < 3 and grid_v[i, j, k][0] < 0:
                grid_v[i, j, k][0] = 0
            if j < 3 and grid_v[i, j, k][1] < 0:
                grid_v[i, j, k][1] = 0
            if k < 3 and grid_v[i, j, k][2] < 0:
                grid_v[i, j, k][2] = 0
            if i >= n_grid - 3 and grid_v[i, j, k][0] > 0:
                grid_v[i, j, k][0] = 0
            if j >= n_grid - 3 and grid_v[i, j, k][1] > 0:
                grid_v[i, j, k][1] = 0
            if k >= n_grid - 3 and grid_v[i, j, k][2] > 0:
                grid_v[i, j, k][2] = 0
    
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic kernels
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector([0.0, 0.0, 0.0])
        new_C = ti.Matrix.zero(float, 3, 3)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            dpos = (ti.Vector([i, j, k]).cast(float) - fx) * dx
            g_v = grid_v[base + ti.Vector([i, j, k])]

            # wx = [0.5*(1.5 - fx[0])**2,
            #     0.75-(fx[0]-1)**2,
            #     0.5*(fx[0]-0.5)**2]

            # wy = [0.5*(1.5 - fx[1])**2,
            #     0.75-(fx[1]-1)**2,
            #     0.5*(fx[1]-0.5)**2]

            # wz = [0.5*(1.5 - fx[2])**2,
            #     0.75-(fx[2]-1)**2,
            #     0.5*(fx[2]-0.5)**2]

            # P2G / G2P 内部
            # weight = wx[i] * wy[j] * wz[k]

            weight = w[i][0] * w[j][1] * w[k][2]
            new_v += weight * g_v
            new_C += inv_dx * inv_dx * 4 * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]


@ti.kernel
def reset():
    group_size = n_particles // num_group
    for i in range(n_particles):
        r = ti.Vector([ti.random(), ti.random(), ti.random()])
        x[i] = x_min[None] + r * (x_max[None] - x_min[None])
        v[i] = [0, 0, 0]
        F[i] = ti.Matrix.identity(float, 3)
        C[i] = ti.Matrix.zero(float, 3, 3)
        Jp[i] = 1.0

    #TODO: Set material types



reset()
gravity[None] = [0, -10, 0]

window = ti.ui.Window("Test for Drawing 3d-lines", (768, 768))
canvas = window.get_canvas()
scene = ti.ui.Scene()

camera = ti.ui.Camera()
camera.position(-2.0, 0.5, -2.0)  # x, y, z
camera.lookat(0.5, 0.5, 0.5)
#camera.up(0, 1, 0)
#camera.projection_mode(ti.ui.ProjectionMode.Perspective)
scene.set_camera(camera)

scene.point_light(pos=(1, 1, 1), color=(1.2, 1.2, 1.2))  # 明亮白光
scene.point_light(pos=(-2, 2, -2), color=(0.8, 0.8, 0.8))
canvas.set_background_color((0.5, 0.5, 0.5)) 
print(x.shape)

for frame in range(200000):
    # for s in range(int(2e-3 // dt)):
    #     substep()
    substep()
    scene.particles(x, radius=dx, color=(1.0, 0.6, 0.9))
    canvas.scene(scene)
    window.show()




