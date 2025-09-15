import taichi as ti
import itertools
import os
import shutil
import subprocess
import numpy as np
import time

ROOT = os.path.dirname(os.path.abspath(__file__))   # 脚本所在的绝对路径

class MPMBeamSimulator:
    def __init__(
        self,
        n: int = 64,
        num_particles: int = 10000,
        dt: float = 1e-4,
        # frames: int = 120,
        # steps_per_frame: int = 50,
        frames: int = 20,
        steps_per_frame: int = 300,
        gui_res: int = 224,
        output_dir: str = "output",
        gravity: float = 9.8,
        create_video: bool = False,
    ):
        # Initialize Taichi
        #ti.init(arch=ti.cpu, random_seed=10)  # Use 'ti.cuda' if your GPU supports it
        ti.init(arch=ti.metal, kernel_profiler=True, random_seed=10)  # Use 'ti.cuda' if your GPU supports it

        # Simulation parameters
        self.n = n
        self.num_particles = num_particles
        self.dt = dt
        self.dx = 1.0 / n
        self.inv_dx = float(n)
        self.frames = frames
        self.steps_per_frame = steps_per_frame
        self.gui_res = gui_res
        self.output_dir = os.path.join(ROOT, output_dir)
        self.gravity = gravity
        self.p_vol = (self.dx * 0.5) ** 2
        self.create_video = create_video
        self.data_num = 0


        # Fields
        self.x = ti.Vector.field(2, ti.f32, shape=num_particles)
        self.v = ti.Vector.field(2, ti.f32, shape=num_particles)
        self.E_particle = ti.field(ti.f32, shape=num_particles)
        self.nu_particle = ti.field(ti.f32, shape=num_particles)
        self.mu_particle = ti.field(ti.f32, shape=num_particles)
        self.la_particle = ti.field(ti.f32, shape=num_particles)
        self.p_mass_particle = ti.field(ti.f32, shape=num_particles)
        self.C = ti.Matrix.field(2, 2, ti.f32, shape=num_particles)
        self.F = ti.Matrix.field(2, 2, ti.f32, shape=num_particles)
        self.Jp = ti.field(ti.f32, shape=num_particles)
        self.grid_v = ti.Vector.field(2, ti.f32, shape=(n, n))
        self.grid_m = ti.field(ti.f32, shape=(n, n))

        # Define kernels
        @ti.kernel
        def substep():
            # Reset grid
            for i, j in self.grid_m:
                self.grid_v[i, j] = ti.Vector.zero(ti.f32, 2)
                self.grid_m[i, j] = 0

            # P2G
            for p in self.x:
                Xp = self.x[p] * self.inv_dx
                base = (Xp - 0.5).cast(int)
                fx = Xp - base.cast(float)

                # Quadratic weights
                w = [
                    ti.Vector([0.5 * (1.5 - fx[i]) ** 2,
                               0.75 - (fx[i] - 1.0) ** 2,
                               0.5 * (fx[i] - 0.5) ** 2])
                    for i in range(2)
                ]

                # Update deformation gradient
                self.F[p] = (ti.Matrix.identity(ti.f32, 2) + self.dt * self.C[p]) @ self.F[p]
                h = ti.exp(10 * (1.0 - self.Jp[p]))
                mu = self.mu_particle[p] * h
                la = self.la_particle[p] * h
                U, sig, V = ti.svd(self.F[p])
                J = sig[0, 0] * sig[1, 1]
                stress = (
                    2 * mu * (self.F[p] - U @ V.transpose()) @ self.F[p].transpose()
                    + ti.Matrix.identity(ti.f32, 2) * la * J * (J - 1)
                )
                stress *= -self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx
                affine = stress + self.p_mass_particle[p] * self.C[p]

                # Scatter to grid
                for i in ti.static(range(3)):
                    for j in ti.static(range(3)):
                        offset = ti.Vector([i, j])
                        dpos = (offset.cast(float) - fx) * self.dx
                        weight = w[0][i] * w[1][j]
                        self.grid_v[base + offset] += (
                            weight * (self.p_mass_particle[p] * self.v[p] + affine @ dpos)
                        )
                        self.grid_m[base + offset] += weight * self.p_mass_particle[p]

            # Grid operations
            for i, j in self.grid_m:
                if self.grid_m[i, j] > 0:
                    self.grid_v[i, j] /= self.grid_m[i, j]
                    self.grid_v[i, j][1] -= self.dt * self.gravity

                    # Boundary conditions
                    if i < 2 or i > self.n - 3:
                        self.grid_v[i, j] = [0, 0]
                    if j < 3 and self.grid_v[i, j][1] < 0:
                        self.grid_v[i, j][1] = 0
                    if j > self.n - 3 and self.grid_v[i, j][1] > 0:
                        self.grid_v[i, j][1] = 0

            # G2P
            for p in self.x:
                Xp = self.x[p] * self.inv_dx
                base = (Xp - 0.5).cast(int)
                fx = Xp - base.cast(float)

                w = [
                    ti.Vector([0.5 * (1.5 - fx[i]) ** 2,
                               0.75 - (fx[i] - 1.0) ** 2,
                               0.5 * (fx[i] - 0.5) ** 2])
                    for i in range(2)
                ]

                new_v = ti.Vector.zero(ti.f32, 2)
                new_C = ti.Matrix.zero(ti.f32, 2, 2)

                for i in ti.static(range(3)):
                    for j in ti.static(range(3)):
                        offset = ti.Vector([i, j])
                        dpos = (offset.cast(float) - fx) * self.dx
                        weight = w[0][i] * w[1][j]
                        g_v = self.grid_v[base + offset]
                        new_v += weight * g_v
                        new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)

                self.v[p], self.C[p] = new_v, new_C
                self.x[p] += self.dt * self.v[p]

        @ti.kernel
        def initialize(
            E_tmp: ti.f32,
            nu_tmp: ti.f32,
            p_mass_tmp: ti.f32,
        ):
            for i in range(self.num_particles):
                #self.x[i] = [ti.random() * 0.8 + 0.0, ti.random() * 0.2 + 0.6]
                # Assign the position of the particles uniformly
                grid_nx = int(ti.sqrt(self.num_particles)) * 2
                grid_ny = int(self.num_particles / grid_nx)
                row = i // grid_nx
                col = i % grid_nx
                self.x[i] = [
                    col / grid_nx * 0.8 + 0.0 + 0.002* ti.random(),
                    row / grid_ny * 0.2 + 0.5 + 0.002* ti.random()
                ]
                self.E_particle[i] = E_tmp
                self.nu_particle[i] = nu_tmp
                self.mu_particle[i] = self.E_particle[i] / (2 * (1 + self.nu_particle[i]))
                self.la_particle[i] = (
                    self.E_particle[i] * self.nu_particle[i] /
                    ((1 + self.nu_particle[i]) * (1 - 2 * self.nu_particle[i]))
                )
                self.p_mass_particle[i] = p_mass_tmp
                self.v[i] = [0, -2]
                self.F[i] = ti.Matrix.identity(ti.f32, 2)
                self.Jp[i] = 1.0
                self.C[i] = ti.Matrix.zero(ti.f32, 2, 2)
        
        self.substep = substep
        self.initialize = initialize
    
    def norm_color(self, val, min_val, max_val, lo=0.2, hi=0.8):
        if max_val == min_val:
            return (lo + hi) / 2  # 避免除以0
        return lo + (val - min_val) / (max_val - min_val) * (hi - lo)
    def color_float_to_hex(self, r, g, b):
        #print(f"r: {r}, g: {g}, b: {b}")
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)
        return int((r << 16) + (g << 8) + b)

    

    def run(
        self,
        Es: list,
        nus: list,
        rhos: list,
    ):
        gui = ti.GUI("MPM Beam", res=self.gui_res, background_color=0x000000, show_gui=False)

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        self.raw_dir = os.path.join(self.output_dir, "raw")
        os.makedirs(self.raw_dir, exist_ok=True)
        self.input_dir = os.path.join(self.output_dir, "input")
        os.makedirs(self.input_dir, exist_ok=True)
        self.label_dir = os.path.join(self.output_dir, "label")
        os.makedirs(self.label_dir, exist_ok=True)
        self.video_dir = os.path.join(self.output_dir, "video")
        os.makedirs(self.video_dir, exist_ok=True)

        for idx, (E, rho, nu) in enumerate(itertools.product(Es, rhos, nus)):
            E_min, E_max = min(Es), max(Es)
            rho_min, rho_max = min(rhos), max(rhos)
            nu_min, nu_max = min(nus), max(nus)

            E_to_rho_min = E_min / rho_max
            E_to_rho_max = E_max / rho_min

            r = self.norm_color(E / rho, E_to_rho_min, E_to_rho_max)
            g = self.norm_color(rho, rho_min, rho_max)
            b = self.norm_color(nu, nu_min, nu_max)
            color = int(self.color_float_to_hex(r, g, b))

            tag = f"V{idx:06d}"

            dir_r = os.path.join(self.raw_dir, tag)
            os.makedirs(dir_r, exist_ok=True)
            dir_i = os.path.join(self.input_dir, tag)
            os.makedirs(dir_i, exist_ok=True)
            dir_l = os.path.join(self.label_dir, tag)
            os.makedirs(dir_l, exist_ok=True)

            vid = os.path.join(self.video_dir, f"beam_{tag}.mp4")
            # # Prepare frame folder
            # if os.path.exists(fdir):
            #     shutil.rmtree(fdir)
            # os.makedirs(fdir)

            # Initialize particles
            self.initialize(E, nu, rho * (self.dx * 0.5) ** 2)
            # Simulation loop
            for frame in range(self.frames):
                
                if frame == 0:
                    # Save initial frame
                    gui.circles(self.x.to_numpy(), radius=3.0, color=color)
                    gui.show(f"{dir_r}/f{frame:06d}.png")
                    gui.circles(self.x.to_numpy(), radius=3.0, color=color)
                    gui.show(f"{dir_l}/f{frame:06d}.png")
                    gui.circles(self.x.to_numpy(), radius=3.0, color = 0xFFFFFF)
                    gui.show(f"{dir_i}/f{frame:06d}.png")
                else:
                    gui.circles(self.x.to_numpy(), radius=3.0, color=color)
                    gui.show(f"{dir_r}/f{frame:06d}.png")
                    gui.circles(self.x.to_numpy(), radius=3.0, color = 0xFFFFFF)
                    gui.show(f"{dir_i}/f{frame:06d}.png")
                for _ in range(self.steps_per_frame):
                    self.substep()


            # Compose video via ffmpeg

            if self.create_video:
                print(f"[{tag}] Generating video...")
                subprocess.run([
                    "ffmpeg", "-loglevel", "error", "-framerate", "30",
                    "-i", f"{dir_r}/f%06d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", vid
                ])
                print(f"[{tag}] Saved to {vid}\n")
            else:
                # If not creating video, just copy the frames to the output directory
                pass
            

if __name__ == "__main__":
    simulator = MPMBeamSimulator(n=64, num_particles=4900, dt=1e-4, frames=20, steps_per_frame=350, output_dir="mpm_output_0527_decouple", create_video = False)
    Es = [12.0e4, 26.0e4]
    Nus = [0.01, 0.45]
    Rhos = [1.0, 4.0]
    Es   = np.linspace(12.0e4, 26.0e4, num=25).tolist()
    Nus  = np.linspace(0.01, 0.45, num=20).tolist()
    Rhos = np.linspace(1.0, 4.0, num=20).tolist()
    tic = time.time()
    simulator.run(Es, Nus, Rhos)
    toc = time.time()
    print(f"Total time: {toc - tic:.2f} seconds")
