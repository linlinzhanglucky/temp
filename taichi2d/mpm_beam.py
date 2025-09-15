import taichi as ti
import numpy as np
import os
import cv2
import subprocess

# ti.init(arch=ti.gpu)
ti.init(arch=ti.cpu)

os.makedirs("frames", exist_ok=True)
os.makedirs("frames_mask", exist_ok=True)

# Constants
n_particles_x, n_particles_y = 64, 16
n_particles = n_particles_x * n_particles_y
n_grid = 128
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 2e-4
p_vol, p_rho = (dx * 0.5) ** 2, 1
E_base = 400
nu = 0.2
mu_0 = E_base / (2 * (1 + nu))
lambda_0 = E_base * nu / ((1 + nu) * (1 - 2 * nu))

# Taichi fields
x = ti.Vector.field(2, ti.f32, shape=n_particles)
v = ti.Vector.field(2, ti.f32, shape=n_particles)
C = ti.Matrix.field(2, 2, ti.f32, shape=n_particles)
F = ti.Matrix.field(2, 2, ti.f32, shape=n_particles)
grid_v = ti.Vector.field(2, ti.f32, shape=(n_grid, n_grid))
grid_m = ti.field(ti.f32, shape=(n_grid, n_grid))
colors = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)

# Physical parameters
mass = ti.field(ti.f32, shape=n_particles)
stiffness = ti.field(ti.f32, shape=n_particles)
damping = ti.field(ti.f32, shape=n_particles)
fixed = ti.field(ti.i32, shape=n_particles)  # 1 if fixed

@ti.kernel
def initialize():
    for i, j in ti.ndrange(n_particles_x, n_particles_y):
        p = i * n_particles_y + j
        offset = ti.Vector([0.15, 0.2])
        spacing = 0.6 / max(n_particles_x, n_particles_y)
        x[p] = offset + spacing * ti.Vector([i, j])
        v[p] = [0, 0]
        F[p] = ti.Matrix.identity(ti.f32, 2)
        C[p] = ti.Matrix.zero(ti.f32, 2, 2)

        # Fix left edge
        if i < 2:
            fixed[p] = 1
        else:
            fixed[p] = 0

        # Region-based assignment by x
        if x[p].x < 0.35:
            mass[p] = 0.2
            stiffness[p] = 30000
            damping[p] = 400
            colors[p] = [1.0, 0.0, 0.0]  # Red = mass
        elif x[p].x < 0.55:
            mass[p] = 0.1
            stiffness[p] = 15000
            damping[p] = 150
            colors[p] = [0.0, 1.0, 0.0]  # Green = stiffness
        else:
            mass[p] = 0.05
            stiffness[p] = 8000
            damping[p] = 50
            colors[p] = [0.0, 0.0, 1.0]  # Blue = damping

@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0

    for p in x:
        Xp = x[p] * inv_dx
        base = (Xp - 0.5).cast(int)
        fx = Xp - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        F[p] = (ti.Matrix.identity(ti.f32, 2) + dt * C[p]) @ F[p]
        J = F[p].determinant()
        mu = mu_0 * (stiffness[p] / E_base)
        la = lambda_0 * (stiffness[p] / E_base)
        stress = 2 * mu * (F[p] - ti.Matrix.identity(ti.f32, 2)) @ F[p].transpose() + \
                 ti.Matrix.identity(ti.f32, 2) * la * ti.log(J) * J
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + mass[p] * C[p]

        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = (ti.Vector([i, j]) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + ti.Vector([i, j])] += weight * (mass[p] * v[p] + affine @ dpos)
            grid_m[base + ti.Vector([i, j])] += weight * mass[p]

    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
            grid_v[i, j][1] -= dt * 9.8
            if i < 3:
                grid_v[i, j] = [0, 0]

    for p in x:
        Xp = x[p] * inv_dx
        base = (Xp - 0.5).cast(int)
        fx = Xp - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        new_v = ti.Vector.zero(ti.f32, 2)
        new_C = ti.Matrix.zero(ti.f32, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)

        if fixed[p] == 1:
            v[p] = [0, 0]
        else:
            v[p] = new_v * ti.exp(-dt * damping[p] / 500.0)
        C[p] = new_C
        x[p] += dt * v[p]

# RGB to hex for GUI
def rgb_to_hex(rgb):
    rgb_int = (rgb * 255).astype(np.uint8)
    return (rgb_int[:, 0] << 16) + (rgb_int[:, 1] << 8) + rgb_int[:, 2]

# Run simulation
initialize()
gui = ti.GUI("MPM Beam", res=512, background_color=0x112F41)
frame_id = 0
while gui.running and frame_id < 300:
    for step in range(20):
        substep()
    color_hex = rgb_to_hex(colors.to_numpy())
    gui.circles(x.to_numpy(), radius=1.5, color=color_hex)
    gui.show(f"frames/{frame_id:04d}.png")
    frame_id += 1

# Generate video
print("Saving video with FFmpeg...")
subprocess.run([
    "ffmpeg", "-framerate", "30", "-i", "frames/%04d.png",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "test_new.mp4"
])

# Generate boundary mask for final frame
img = cv2.imread(f"frames/{frame_id-1:04d}.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(binary)
if contours:
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    cv2.imwrite(f"frames_mask/{frame_id-1:04d}.png", mask)

print("Video and mask saved!")
