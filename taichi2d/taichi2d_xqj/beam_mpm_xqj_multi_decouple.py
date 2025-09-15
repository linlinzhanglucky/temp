import taichi as ti
import itertools
import os
import shutil
import subprocess
import numpy as np
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
max_num_regions = 2
frames = 40
steps_per_frame = 200
output_dir = "mpm_output_0529_2"
decouple = True
E_min = 12.0e4 * 1
E_max = 26.0e4
Nu_min = 0.01
Nu_max = 0.45
Rho_min = 1.0 * 1
Rho_max = 4.0
E_to_Rho_min = E_min / Rho_max
E_to_Rho_max = E_max / Rho_min
x_min = 0.0
x_max = 0.8
y_min = 0.5
y_max = 0.7


n = 64  # Grid resolution
num_particles = 4900  # Number of particles
dt = 1e-4  # Time step
num_samples = 15000#10000  # Number of samples to generate
num_channels = 3  # RGB channels for color

if num_channels == 2:
    Rho_max = Rho_min
    E_min = 0.8 * E_min

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
        ti.init(arch=ti.metal, kernel_profiler=True)  # Use 'ti.cuda' if your GPU supports it

        # Simulation 
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
        self.color = ti.field(ti.i32, shape=num_particles)

        self.region_id = ti.field(ti.i32, shape=num_particles)
        self.region_E = ti.field(ti.f32, shape=max_num_regions)
        self.region_mass = ti.field(ti.f32, shape=max_num_regions)
        self.region_nu = ti.field(ti.f32, shape=max_num_regions)
        self.region_color = ti.field(ti.i32, shape=max_num_regions)
        self.center_field = ti.Vector.field(2, ti.f32, shape=max_num_regions)
        self.cur_regions = ti.field(ti.i32, shape=())



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
            # E_tmp: ti.f32,
            # nu_tmp: ti.f32,
            # p_mass_tmp: ti.f32,
        ):
            for i in range(self.num_particles):
                #self.x[i] = [ti.random() * 0.8 + 0.0, ti.random() * 0.2 + 0.6]
                # Assign the position of the particles uniformly
                grid_nx = int(ti.sqrt(self.num_particles)) * 2
                grid_ny = int(self.num_particles / grid_nx)
                row = i // grid_nx
                col = i % grid_nx
                self.x[i] = [
                    col / grid_nx * (x_max - x_min) + x_min + 0.002* ti.random(),
                    row / grid_ny * (y_max - y_min) + y_min + 0.002* ti.random()
                ]
                best_k = 0
                best_d2 = 1e8
                for k in range(max_num_regions):
                    if k >= self.cur_regions[None]:
                        break
                    d2 = (self.x[i][0] - self.center_field[k][0]) ** 2 + (self.x[i][1] - self.center_field[k][1]) ** 2
                    if d2 < best_d2:
                        best_d2 = d2
                        best_k = k
                self.region_id[i] = best_k
                E = self.region_E[best_k]
                nu = self.region_nu[best_k]
                mass = self.region_mass[best_k]
                self.E_particle[i] = E
                self.nu_particle[i] = nu
                self.mu_particle[i] = self.E_particle[i] / (2 * (1 + self.nu_particle[i]))
                self.la_particle[i] = (
                    self.E_particle[i] * self.nu_particle[i] /
                    ((1 + self.nu_particle[i]) * (1 - 2 * self.nu_particle[i]))
                )
                self.p_mass_particle[i] = mass
                self.color[i] = self.region_color[best_k]
                self.v[i] = [0, 0]
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
    
    def build_regions(self, num_regions):
        # ---------- CPU 端生成离散 Voronoi ----------
        self.cur_regions[None] = num_regions

        xmin, xmax = 0.0, 0.8
        ymin, ymax = 0.5, 0.7

        centers = np.stack([
            np.random.uniform(xmin, xmax, num_regions),
            np.random.uniform(ymin, ymax, num_regions)
        ], axis=-1).astype(np.float32)

        self.center_field.from_numpy(
            np.pad(centers, ((0, max_num_regions - num_regions), (0,0))).astype(np.float32)
        )

        for k in range(num_regions):
            E = np.random.uniform(E_min, E_max)
            nu = np.random.uniform(Nu_min, Nu_max)
            rho = np.random.uniform(Rho_min, Rho_max)
            self.region_E[k] = E
            self.region_nu[k] = nu
            self.region_mass[k] = rho * (self.dx * 0.5) ** 2

            if decouple:
                r = self.norm_color(E/rho, E_to_Rho_min, E_to_Rho_max)
            else:
                r = self.norm_color(E, E_min, E_max)
            # r = self.norm_color(E, 12.0e4, 26.0e4)
            if num_channels == 2:
                g = self.norm_color(2.0, 1.0, 4.0)
            else:
                g = self.norm_color(rho, 1.0, 4.0)
            b = self.norm_color(nu, 0.01, 0.45)

            self.region_color[k] = int(self.color_float_to_hex(r, g, b))
        
    

    def run(
        self,
        # Es: list,
        # nus: list,
        # rhos: list,
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

        for idx in range(num_samples):
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
            self.build_regions(np.random.randint(2, max_num_regions + 1))
            self.initialize()
            color = self.color.to_numpy()
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
    simulator = MPMBeamSimulator(n=n, num_particles=num_particles, dt=dt, frames=frames, steps_per_frame=steps_per_frame, output_dir=output_dir, create_video = False)
    # Es   = np.linspace(12.0e4, 26.0e4, num=25).tolist()
    # Nus  = np.linspace(0.01, 0.45, num=20).tolist()
    # Rhos = np.linspace(1.0, 4.0, num=20).tolist()

    param_info = {
        "max_num_regions": max_num_regions,
        "frames": frames,
        "steps_per_frame": steps_per_frame,
        "E_min": E_min,
        "E_max": E_max,
        "nu_min": Nu_min,
        "nu_max": Nu_max,
        "rho_min": Rho_min,
        "rho_max": Rho_max,
        "type": "beam",
        "decouple": decouple,
        "n": n,
        "num_particles": num_particles,
        "dt": dt,
        "num_samples": num_samples,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "E_to_Rho_min": E_to_Rho_min,
        "E_to_Rho_max": E_to_Rho_max,
        "num_channels": num_channels,

    }

    param_file = os.path.join(simulator.output_dir, "global_config.txt")
    os.makedirs(simulator.output_dir, exist_ok=True)
    with open(param_file, "w") as f:
        for key, val in param_info.items():
            f.write(f"{key}: {val}\n")

    tic = time.time()
    # simulator.run(Es, Nus, Rhos)
    simulator.run()
    toc = time.time()
    print(f"Total time: {toc - tic:.2f} seconds")
