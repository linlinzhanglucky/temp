import taichi as ti
import math

ti.init(arch=ti.cpu)  # Use 'ti.cuda' if your GPU supports it

# Simulation resolution and parameters
n = 64
num_particles = 8192
dx = 1 / n
inv_dx = float(n)
dt = 1e-4
p_vol = (dx * 0.5) ** 2
p_rho = 1
p_mass = p_vol * p_rho
E = 400  # Young's modulus
nu = 0.2  # Poisson's ratio
mu_0 = E / (2 * (1 + nu))
la_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

# Fields
x = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)  # position
v = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=ti.f32, shape=num_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=ti.f32, shape=num_particles)  # deformation gradient
Jp = ti.field(dtype=ti.f32, shape=num_particles)  # plastic deformation

grid_v = ti.Vector.field(2, dtype=ti.f32, shape=(n, n))  # grid node velocity
grid_m = ti.field(dtype=ti.f32, shape=(n, n))  # grid node mass

@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = ti.Vector.zero(ti.f32, 2)
        grid_m[i, j] = 0

    for p in x:
        Xp = x[p] * inv_dx
        base = (Xp - 0.5).cast(int)
        fx = Xp - base.cast(float)

        w = [ti.Vector([0.5 * (1.5 - fx[i]) ** 2,
                        0.75 - (fx[i] - 1.0) ** 2,
                        0.5 * (fx[i] - 0.5) ** 2]) for i in range(2)]

        F[p] = (ti.Matrix.identity(ti.f32, 2) + dt * C[p]) @ F[p]
        h = ti.exp(10 * (1.0 - Jp[p]))
        mu, la = mu_0 * h, la_0 * h
        U, sig, V = ti.svd(F[p])
        J = sig[0, 0] * sig[1, 1]
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + \
                 ti.Matrix.identity(ti.f32, 2) * la * J * (J - 1)
        stress *= -dt * p_vol * 4 * inv_dx * inv_dx
        affine = stress + p_mass * C[p]

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[0][i] * w[1][j]
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass

    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
            grid_v[i, j][1] -= dt * 9.8

            if i < 3 and grid_v[i, j][0] < 0: grid_v[i, j][0] = 0
            if i > n - 3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0: grid_v[i, j][1] = 0
            if j > n - 3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0

    for p in x:
        Xp = x[p] * inv_dx
        base = (Xp - 0.5).cast(int)
        fx = Xp - base.cast(float)

        w = [ti.Vector([0.5 * (1.5 - fx[i]) ** 2,
                        0.75 - (fx[i] - 1.0) ** 2,
                        0.5 * (fx[i] - 0.5) ** 2]) for i in range(2)]

        new_v = ti.Vector.zero(ti.f32, 2)
        new_C = ti.Matrix.zero(ti.f32, 2, 2)

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[0][i] * w[1][j]
                g_v = grid_v[base + offset]
                new_v += weight * g_v
                # new_C += 4 * inv_dx * weight * ti.outer_product(g_v, dpos)
                #debug
                new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)


        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]

@ti.kernel
def initialize():
    for i in range(num_particles):
        x[i] = [ti.random() * 0.4 + 0.2, ti.random() * 0.2 + 0.5]
        v[i] = [0, -1]
        F[i] = ti.Matrix.identity(ti.f32, 2)
        Jp[i] = 1.0
        C[i] = ti.Matrix.zero(ti.f32, 2, 2)

initialize()

gui = ti.GUI("MPM Beam Simulation", res=512, background_color=0x112F41)
# while gui.running:
#     for s in range(50):
#         substep()
#     gui.circles(x.to_numpy(), radius=1.5, color=0x66ccff)
#     gui.show()

#debug: modified to save the video
import os

os.makedirs("frames", exist_ok=True)
frame = 0

while gui.running and frame < 300:
    for s in range(50):
        substep()
    gui.circles(x.to_numpy(), radius=1.5, color=0x66ccff)
    gui.show(f"frames/{frame:04d}.png")  # show and save
    frame += 1


import subprocess

print("Saving video with FFmpeg...")
subprocess.run([
    "ffmpeg", "-framerate", "30", "-i", "frames/%04d.png",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "beam_output.mp4"
])
print("Video saved as beam_output.mp4")
