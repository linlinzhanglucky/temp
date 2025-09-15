import argparse
import numpy as np
import taichi as ti
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import sys


@ti.data_oriented
class Beam:
    def __init__(self, 
                 N_row=4, 
                 N_column=8, 
                 mass_range=(0.05, 0.2),
                 stiffness_range=(10000.0, 25000.0),
                 damping_range=(50.0, 500.0),
                 mass_value=None, 
                 stiffness_value=None, 
                 damping_value=None, 
                 use_regions=True, 
                 seed=None):  # N_row x N_column grid
        # Grid parameters
        self.N_row = N_row
        self.N_column = N_column
        self.NV = N_row * N_column  # total number of vertices
        self.NC = (N_row-1) * (N_column-1)  # total number of cells
        # Edges: horizontal + vertical + diagonal1 + diagonal2
        self.NE = (N_row * (N_column-1) +  # horizontal springs
                  (N_row-1) * N_column +    # vertical springs
                  2 * (N_row-1) * (N_column-1))  # diagonal springs
        
        # Hyperparameters for mass and stiffness
        # Store input values
        self.mass_value = mass_value
        self.stiffness_value = stiffness_value
        self.kd = damping_value
        
        # Physical parameters
        self.mass_range = mass_range
        self.ks_range = stiffness_range
        self.kd_range = damping_range
        
        self.kf = 1.0e7  # fix point stiffness
        self.gravity = ti.Vector([0.0, -9.81])
        
        # Position and dynamics fields
        self.pos = ti.Vector.field(2, ti.f32, self.NV)
        self.initPos = ti.Vector.field(2, ti.f32, self.NV)
        self.vel = ti.Vector.field(2, ti.f32, self.NV)
        self.force = ti.Vector.field(2, ti.f32, self.NV)
        self.vel_1D = ti.ndarray(ti.f32, 2 * self.NV)
        self.force_1D = ti.ndarray(ti.f32, 2 * self.NV)
        self.b = ti.ndarray(ti.f32, 2 * self.NV)

        # Mass and stiffness matrices
        self.mass_matrix = ti.field(ti.f32, self.NV)
        self.stiffness_matrix = ti.field(ti.f32, self.NV)
        self.damping_matrix = ti.field(ti.f32, self.NV)

        self.spring = ti.Vector.field(2, ti.i32, self.NE)
        self.indices = ti.field(ti.i32, 2 * self.NE)
        self.Jx = ti.Matrix.field(2, 2, ti.f32, self.NE)
        self.Jv = ti.Matrix.field(2, 2, ti.f32, self.NE)
        self.rest_len = ti.field(ti.f32, self.NE)
        
        # Region information
        self.use_regions = use_regions
        self.region_grid = ti.field(ti.i32, shape=(self.N_row, self.N_column))
        # Initialize with placeholder values, will be resized in init_region_parameters
        self.region_mass = ti.field(ti.f32, shape=1)
        self.region_stiffness = ti.field(ti.f32, shape=1)
        self.region_damping = ti.field(ti.f32, shape=1)

        # Collision parameters
        self.collision_eps = 1e-3  # Collision epsilon
        self.collision_stiffness = 1e5  # Collision response stiffness
        spacing = min(0.8 / max(self.N_row, self.N_column), 0.1)  # Calculate spacing based on grid size
        self.self_collision_radius = spacing * 0.5  # Radius for self-collision detection
                
        # Initialize positions, edges, and parameters
        self.init_pos()
        self.init_edges()
        if self.use_regions:
            self.init_random_regions(min_regions=2, max_regions=2, seed=seed)
        else:
            self.init_unified_parameters()
        
        # Initialize sparse matrix builders
        self.MassBuilder = ti.linalg.SparseMatrixBuilder(2 * self.NV, 2 * self.NV, max_num_triplets=10000)
        self.DBuilder = ti.linalg.SparseMatrixBuilder(2 * self.NV, 2 * self.NV, max_num_triplets=10000)
        self.KBuilder = ti.linalg.SparseMatrixBuilder(2 * self.NV, 2 * self.NV, max_num_triplets=10000)
        self.init_mass_sp(self.MassBuilder)
        self.M = self.MassBuilder.build()
        
        # Fix the left side vertices
        self.fix_vertex = [i * N_column for i in range(N_row)]  # Left column indices
        self.Jf = ti.Matrix.field(2, 2, ti.f32, len(self.fix_vertex))

    @ti.kernel
    def init_pos(self):
        for i, j in ti.ndrange(self.N_row, self.N_column):
            idx = i * self.N_column + j
            # Adjust spacing based on grid size to maintain reasonable total size
            spacing = min(0.8 / max(self.N_row, self.N_column), 0.1)
            # Position beam horizontally: x varies with column (j), y varies with row (i)
            self.pos[idx] = ti.Vector([0.1 + j * spacing, 0.5 + i * spacing])
            self.initPos[idx] = self.pos[idx]
            self.vel[idx] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def init_edges(self):
        # Horizontal springs
        edge_idx = 0
        for i, j in ti.ndrange(self.N_row, self.N_column-1):
            idx = i * self.N_column + j
            self.spring[edge_idx] = ti.Vector([idx, idx + 1])
            self.rest_len[edge_idx] = (self.pos[idx] - self.pos[idx + 1]).norm()
            edge_idx += 1

        # Vertical springs
        for i, j in ti.ndrange(self.N_row-1, self.N_column):
            idx = i * self.N_column + j
            self.spring[edge_idx] = ti.Vector([idx, idx + self.N_column])
            self.rest_len[edge_idx] = (self.pos[idx] - self.pos[idx + self.N_column]).norm()
            edge_idx += 1

        # Diagonal springs (/)
        for i, j in ti.ndrange(self.N_row-1, self.N_column-1):
            idx = i * self.N_column + j
            self.spring[edge_idx] = ti.Vector([idx, idx + self.N_column + 1])
            self.rest_len[edge_idx] = (self.pos[idx] - self.pos[idx + self.N_column + 1]).norm()
            edge_idx += 1

        # Diagonal springs (\)
        for i, j in ti.ndrange(self.N_row-1, self.N_column-1):
            idx = i * self.N_column + j + 1
            self.spring[edge_idx] = ti.Vector([idx, idx + self.N_column - 1])
            self.rest_len[edge_idx] = (self.pos[idx] - self.pos[idx + self.N_column - 1]).norm()
            edge_idx += 1

    @ti.kernel
    def init_mass_sp(self, M: ti.types.sparse_matrix_builder()):
        for i in range(self.NV):
            M[2 * i + 0, 2 * i + 0] += self.mass_matrix[i]
            M[2 * i + 1, 2 * i + 1] += self.mass_matrix[i]

    @ti.func
    def clear_force(self):
        for i in self.force:
            self.force[i] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def compute_force(self):
        self.clear_force()
        for i in self.force:
            self.force[i] += self.gravity * self.mass_matrix[i]

        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            pos1, pos2 = self.pos[idx1], self.pos[idx2]
            dis = pos2 - pos1
            # Use average stiffness of connected nodes
            ks_avg = (self.stiffness_matrix[idx1] + self.stiffness_matrix[idx2]) * 0.5
            force = ks_avg * (dis.norm() - self.rest_len[i]) * dis.normalized()
            self.force[idx1] += force
            self.force[idx2] -= force

        # Fix constraint gradient for all left side vertices
        for i in range(self.N_row):
            idx = i * self.N_column
            self.force[idx] += self.kf * (self.initPos[idx] - self.pos[idx])

    @ti.kernel
    def compute_Jacobians(self):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            pos1, pos2 = self.pos[idx1], self.pos[idx2]
            dx = pos1 - pos2
            I = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
            dxtdx = ti.Matrix([[dx[0] * dx[0], dx[0] * dx[1]], [dx[1] * dx[0], dx[1] * dx[1]]])
            l = dx.norm()
            if l != 0.0:
                l = 1.0 / l
            # Use average stiffness of connected nodes
            ks_avg = (self.stiffness_matrix[idx1] + self.stiffness_matrix[idx2]) * 0.5
            # each spring has a different damping coefficient
            kd_avg = (self.damping_matrix[idx1] + self.damping_matrix[idx2]) * 0.5 # important
            self.Jx[i] = (I - self.rest_len[i] * l * (I - dxtdx * l**2)) * ks_avg
            self.Jv[i] = kd_avg * I

        # Fix point constraint hessian for all left side vertices
        for i in range(self.N_row):
            self.Jf[i] = ti.Matrix([[-self.kf, 0], [0, -self.kf]])

    @ti.kernel
    def assemble_K(self, K: ti.types.sparse_matrix_builder()):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            for m, n in ti.static(ti.ndrange(2, 2)):
                K[2 * idx1 + m, 2 * idx1 + n] -= self.Jx[i][m, n]
                K[2 * idx1 + m, 2 * idx2 + n] += self.Jx[i][m, n]
                K[2 * idx2 + m, 2 * idx1 + n] += self.Jx[i][m, n]
                K[2 * idx2 + m, 2 * idx2 + n] -= self.Jx[i][m, n]
        
        # Add fix constraint to K for all left side vertices
        for i in range(self.N_row):
            idx = i * self.N_column  # Left column indices
            for m, n in ti.static(ti.ndrange(2, 2)):
                K[2 * idx + m, 2 * idx + n] += self.Jf[i][m, n]

    @ti.kernel
    def assemble_D(self, D: ti.types.sparse_matrix_builder()):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            for m, n in ti.static(ti.ndrange(2, 2)):
                D[2 * idx1 + m, 2 * idx1 + n] -= self.Jv[i][m, n]
                D[2 * idx1 + m, 2 * idx2 + n] += self.Jv[i][m, n]
                D[2 * idx2 + m, 2 * idx1 + n] += self.Jv[i][m, n]
                D[2 * idx2 + m, 2 * idx2 + n] -= self.Jv[i][m, n]

    @ti.kernel
    def updatePosVel(self, h: ti.f32, dv: ti.types.ndarray()):
        for i in self.pos:
            self.vel[i] += ti.Vector([dv[2 * i], dv[2 * i + 1]])
            self.pos[i] += h * self.vel[i]

    @ti.kernel
    def copy_to(self, des: ti.types.ndarray(), source: ti.template()):
        for i in range(self.NV):
            des[2 * i] = source[i][0]
            des[2 * i + 1] = source[i][1]

    @ti.kernel
    def compute_b(self, b: ti.types.ndarray(), f: ti.types.ndarray(), Kv: ti.types.ndarray(), h: ti.f32):
        for i in range(2 * self.NV):
            b[i] = (f[i] + Kv[i] * h) * h

    @ti.func
    def line_segment_distance(self, p1: ti.template(), p2: ti.template(), p3: ti.template(), p4: ti.template()) -> ti.f32:
        """Compute the minimum distance between two line segments"""
        u = p2 - p1
        v = p4 - p3
        w = p1 - p3
        
        a = u.dot(u)
        b = u.dot(v)
        c = v.dot(v)
        d = u.dot(w)
        e = v.dot(w)
        
        D = a * c - b * b
        sc = 0.0
        tc = 0.0
        
        if D < self.collision_eps:  # Lines are almost parallel
            sc = 0.0
            if b > c:
                tc = d / b
            else:
                tc = e / c
        else:
            sc = (b * e - c * d) / D
            tc = (a * e - b * d) / D
            
        # Clamp sc and tc to [0,1]
        sc = ti.min(1.0, ti.max(0.0, sc))
        tc = ti.min(1.0, ti.max(0.0, tc))
        
        # Compute points of closest approach
        p_c1 = p1 + sc * u
        p_c2 = p3 + tc * v
        
        return (p_c1 - p_c2).norm()

    @ti.kernel
    def handle_collisions(self):
        # Self collision (vertex-vertex)
        for i in range(self.NV):
            for j in range(i + 1, self.NV):
                # Skip adjacent vertices that are connected by springs
                if abs(i - j) == 1 or abs(i - j) == self.N_column:
                    continue
                    
                rel_pos = self.pos[i] - self.pos[j]
                dist = rel_pos.norm()
                
                if dist < self.self_collision_radius:
                    # Compute penetration depth and direction
                    penetration = self.self_collision_radius - dist
                    normal = ti.Vector([0.0, 1.0])  # Default direction
                    if dist > self.collision_eps:
                        normal = rel_pos / dist
                        
                    # Apply equal and opposite forces
                    collision_force = normal * self.collision_stiffness * penetration
                    self.force[i] += collision_force
                    self.force[j] -= collision_force
                    
                    # Add damping for stability
                    rel_vel = self.vel[i] - self.vel[j]
                    # damping_force = self.kd * rel_vel
                    damping_force_i = self.damping_matrix[i] * rel_vel
                    damping_force_j = self.damping_matrix[j] * rel_vel
                    self.force[i] -= damping_force_i
                    self.force[j] += damping_force_j

        # Edge-edge collision
        for i in range(self.NE):
            for j in range(i + 1, self.NE):
                # Skip adjacent edges
                idx1_i, idx2_i = self.spring[i][0], self.spring[i][1]
                idx1_j, idx2_j = self.spring[j][0], self.spring[j][1]
                
                if idx1_i == idx1_j or idx1_i == idx2_j or idx2_i == idx1_j or idx2_i == idx2_j:
                    continue
                
                # Compute minimum distance between edges
                dist = self.line_segment_distance(
                    self.pos[idx1_i], self.pos[idx2_i],
                    self.pos[idx1_j], self.pos[idx2_j]
                )
                
                if dist < self.self_collision_radius:
                    # Compute edge midpoints
                    mid_i = (self.pos[idx1_i] + self.pos[idx2_i]) * 0.5
                    mid_j = (self.pos[idx1_j] + self.pos[idx2_j]) * 0.5
                    
                    # Compute collision response direction
                    dir = mid_i - mid_j
                    normal = ti.Vector([0.0, 1.0])  # Default direction
                    if dir.norm() > self.collision_eps:
                        normal = dir.normalized()
                    
                    # Apply forces to all four vertices
                    penetration = self.self_collision_radius - dist
                    force = normal * self.collision_stiffness * penetration * 0.25  # Distribute force among vertices
                    
                    self.force[idx1_i] += force
                    self.force[idx2_i] += force
                    self.force[idx1_j] -= force
                    self.force[idx2_j] -= force
                    
                    # Add damping
                    vel_i = (self.vel[idx1_i] + self.vel[idx2_i]) * 0.5
                    vel_j = (self.vel[idx1_j] + self.vel[idx2_j]) * 0.5
                    rel_vel = vel_i - vel_j
                    # damping_force = self.kd * rel_vel * 0.25
                    damping_force_i = self.damping_matrix[i] * rel_vel * 0.25
                    damping_force_j = self.damping_matrix[j] * rel_vel * 0.25
                    
                    self.force[idx1_i] -= damping_force_i
                    self.force[idx2_i] -= damping_force_i
                    self.force[idx1_j] += damping_force_j
                    self.force[idx2_j] += damping_force_j


    def update(self, h):
        self.compute_force()
        self.handle_collisions()
        self.compute_Jacobians()

        self.assemble_D(self.DBuilder)
        D = self.DBuilder.build()

        self.assemble_K(self.KBuilder)
        K = self.KBuilder.build()

        A = self.M - h * D - h**2 * K

        self.copy_to(self.vel_1D, self.vel)
        self.copy_to(self.force_1D, self.force)

        Kv = K @ self.vel_1D
        self.compute_b(self.b, self.force_1D, Kv, h)

        solver = ti.linalg.SparseSolver(solver_type="LDLT")
        solver.analyze_pattern(A)
        solver.factorize(A)
        dv = solver.solve(self.b)
        self.updatePosVel(h, dv)

    @ti.kernel
    def display_kernel(self, vertices: ti.template(), colors: ti.template()):
        """Kernel to prepare vertices and colors for rendering"""
        for i, j in ti.ndrange(self.N_row-1, self.N_column-1):
            # Calculate base index for this cell's vertices
            base_idx = (i * (self.N_column-1) + j) * 6
            
            # Get the four corners of this cell
            node_tl = i * self.N_column + j
            node_tr = i * self.N_column + (j + 1)
            node_br = (i + 1) * self.N_column + (j + 1)
            node_bl = (i + 1) * self.N_column + j
            
            # Set vertices for the two triangles
            vertices[base_idx + 0] = self.pos[node_tl]
            vertices[base_idx + 1] = self.pos[node_tr]
            vertices[base_idx + 2] = self.pos[node_br]
            vertices[base_idx + 3] = self.pos[node_tl]
            vertices[base_idx + 4] = self.pos[node_br]
            vertices[base_idx + 5] = self.pos[node_bl]
            
            # Initialize color components
            red = 0.0
            green = 0.0
            blue = 0.0
            
            # Check if we're using regions
            if self.use_regions:
                # Get region index for this cell (use top-left node)
                region_idx = self.region_grid[i, j]
                
                # Map region parameters to colors in range (0.2, 0.8)
                # Red channel: mass
                red = 0.2 + 0.6 * (self.region_mass[region_idx] - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0])
                
                # Green channel: stiffness
                green = 0.2 + 0.6 * (self.region_stiffness[region_idx] - self.ks_range[0]) / (self.ks_range[1] - self.ks_range[0])
                
                # Blue channel: damping
                blue = 0.2 + 0.6 * (self.region_damping[region_idx] - self.kd_range[0]) / (self.kd_range[1] - self.kd_range[0])
            else:
                # Map from [0, 1] to [0.1, 0.9]
                red = 0.2 + 0.6 * (self.mass_matrix[node_tl] - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0])
                green = 0.2 + 0.6 * (self.stiffness_matrix[node_tl] - self.ks_range[0]) / (self.ks_range[1] - self.ks_range[0])
                blue = 0.2 + 0.6 * (self.damping_matrix[node_tl] - self.kd_range[0]) / (self.kd_range[1] - self.kd_range[0])  
            
            # Set colors for all vertices of both triangles
            for k in range(6):
                colors[base_idx + k] = ti.Vector([red, green, blue])

    def display(self, window, canvas):
        """Display the beam using Taichi's Window and Canvas system"""
        # Calculate total number of vertices needed
        num_cells = (self.N_row-1) * (self.N_column-1)
        num_vertices = num_cells * 6  # 6 vertices per cell (2 triangles)
        
        # Create vertex and color buffers if they don't exist
        if not hasattr(self, 'vertices'):
            self.vertices = ti.Vector.field(2, dtype=ti.f32, shape=num_vertices)
            self.colors = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
            self.pos_np = ti.Vector.field(2, dtype=ti.f32, shape=self.NV)
        
        # Copy positions to field
        self.pos_np.copy_from(self.pos)
        
        # Update vertices and colors
        self.display_kernel(self.vertices, self.colors)
        
        # Draw the triangles
        canvas.triangles(self.vertices, per_vertex_color=self.colors)

    @ti.kernel
    def spring2indices(self):
        for i in self.spring:
            self.indices[2 * i + 0] = self.spring[i][0]
            self.indices[2 * i + 1] = self.spring[i][1]

    @ti.kernel
    def init_unified_parameters(self):
        # Use provided values or middle values for mass and stiffness
        mass_val = self.mass_value
        stiffness_val = self.stiffness_value
        damping_val = self.kd
        
        # If either value is None, use the middle of the range
        # if mass_val == 0.0:  # ti.kernel doesn't support None, so we use 0.0 as sentinel
        #     mass_val = (self.mass_range[0] + self.mass_range[1]) * 0.5
        # if stiffness_val == 0.0:
        #     stiffness_val = (self.ks_range[0] + self.ks_range[1]) * 0.5
        # if damping_val == 0.0:
        #     damping_val = (self.kd_range[0] + self.kd_range[1]) * 0.5
            
        for i in range(self.NV):
            self.mass_matrix[i] = mass_val
            self.stiffness_matrix[i] = stiffness_val
            self.damping_matrix[i] = damping_val

    def init_region_parameters(self, num_regions=5, seed=None):
        """Initialize parameters by randomly splitting the beam into regions
        
        Args:
            num_regions: Number of regions to create
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Create a grid to track which region each cell belongs to
        region_grid_np = np.zeros((self.N_row, self.N_column), dtype=int)
        
        # Define region centers (random points in the grid)
        region_centers = []
        for _ in range(num_regions):
            row = np.random.randint(0, self.N_row)
            col = np.random.randint(0, self.N_column)
            region_centers.append((row, col))
            
        # Assign each cell to the nearest region center
        for i in range(self.N_row):
            for j in range(self.N_column):
                # Find the closest region center
                min_dist = float('inf')
                closest_region = 0
                
                for r, (center_row, center_col) in enumerate(region_centers):
                    # Use Manhattan distance for simplicity
                    dist = abs(i - center_row) + abs(j - center_col)
                    if dist < min_dist:
                        min_dist = dist
                        closest_region = r
                        
                region_grid_np[i, j] = closest_region
        
        # Generate random mass and stiffness values for each region
        region_mass_np = np.random.uniform(self.mass_range[0], self.mass_range[1], num_regions).astype(np.float32)
        region_stiffness_np = np.random.uniform(self.ks_range[0], self.ks_range[1], num_regions).astype(np.float32)
        region_damping_np = np.random.uniform(self.kd_range[0], self.kd_range[1], num_regions).astype(np.float32)
        
        # Resize the Taichi fields to match the number of regions
        self.region_mass = ti.field(ti.f32, shape=num_regions)
        self.region_stiffness = ti.field(ti.f32, shape=num_regions)
        self.region_damping = ti.field(ti.f32, shape=num_regions)
        
        # Copy the numpy arrays to Taichi fields
        self.region_grid.from_numpy(region_grid_np)
        self.region_mass.from_numpy(region_mass_np)
        self.region_stiffness.from_numpy(region_stiffness_np)
        self.region_damping.from_numpy(region_damping_np)
        # Apply the region values to the mass and stiffness matrices
        self._apply_region_parameters()
        
    @ti.kernel
    def _apply_region_parameters(self):
        """Apply region parameters to the mass and stiffness matrices"""
        for i, j in ti.ndrange(self.N_row, self.N_column):
            idx = i * self.N_column + j
            region_idx = self.region_grid[i, j]
            self.mass_matrix[idx] = self.region_mass[region_idx]
            self.stiffness_matrix[idx] = self.region_stiffness[region_idx]
            self.damping_matrix[idx] = self.region_damping[region_idx]
            
    def init_random_regions(self, min_regions=3, max_regions=8, seed=None):
        """Initialize parameters with a random number of regions
        
        Args:
            min_regions: Minimum number of regions
            max_regions: Maximum number of regions
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Choose a random number of regions
        num_regions = np.random.randint(min_regions, max_regions + 1)
        self.init_region_parameters(num_regions=num_regions)

def save_video(beam, h, video_name="simulation", duration=1.0, fps=200, save_png=False, data_dir=None):
    """Save the simulation as a video using Taichi's Window and Canvas
    Args:
        beam: The beam object
        video_name: Name of the video file (without extension)
        duration: Duration of the video in seconds
        fps: Frames per second
    """
    # Calculate total frames needed
    total_frames = int(duration * fps) 
    
    # Create directories for this video
    raw_dir_vid = os.path.join(data_dir, 'raw', f"{video_name}")
    input_dir_vid = os.path.join(data_dir, 'input', f"{video_name}")
    label_dir_vid = os.path.join(data_dir, 'label', f"{video_name}")
    os.makedirs(raw_dir_vid, exist_ok=True)
    os.makedirs(input_dir_vid, exist_ok=True)
    os.makedirs(label_dir_vid, exist_ok=True)
    
    # Create a headless window for rendering
    window = ti.ui.Window("Beam Simulation", res=(224, 224), show_window=False)
    canvas = window.get_canvas()
    
    # Run simulation and collect frames in memory
    frames = []
    binary_frames = []
    
    # Run simulation and collect frames
    for frame in range(total_frames):
        # Update simulation
        beam.update(h)
        
        # Collect frames
        if frame % 10 == 0:  # Save every 10th frame
            # Render frame
            canvas.set_background_color((0, 0, 0))
            beam.display(window, canvas)
            
            # Process image in one step
            img_array = np.rot90((window.get_image_buffer_as_numpy() * 255).astype(np.uint8), k=1)
            img_binary = np.where(img_array > 0, 255, 0).astype(np.uint8)
            
            # Store frames in memory
            frames.append(img_array)
            binary_frames.append(img_binary)
    
    # After simulation completes successfully, save all frames
    for frame_idx, (img_array, img_binary) in enumerate(zip(frames, binary_frames)):
        # Save images
        Image.fromarray(img_array).save(os.path.join(raw_dir_vid, f"f{frame_idx:06d}.png"))
        Image.fromarray(img_binary).save(os.path.join(input_dir_vid, f"f{frame_idx:06d}.png"))
        
        # Save first frame to label directory
        if frame_idx == 0:
            Image.fromarray(img_array).save(os.path.join(label_dir_vid, f"f{frame_idx:06d}.png"))
    
    # Clear memory
    del frames
    del binary_frames

def simulate(beam, save_to_video=False, video_name="simulation", data_collection=False, data_dir=None):
    h = 0.01  # time step
    if data_collection:
        """data_collection version"""
        save_video(beam, h, video_name=video_name, save_png=data_collection, data_dir=data_dir)


    else:
        """Simulation version"""
        # Create window and canvas
        window = ti.ui.Window("Beam Simulation", res=(500, 500))
        canvas = window.get_canvas()
        
        # Initialize pause variable
        pause = False

        while window.running:
            if not pause:
                beam.update(h)

            # Clear canvas
            canvas.set_background_color((0, 0, 0))
            
            # Render beam
            beam.display(window, canvas)
            
            # Handle events
            if window.get_event(ti.ui.PRESS):
                if window.event.key == ti.ui.ESCAPE:
                    window.running = False
                elif window.event.key == ti.ui.SPACE:
                    pause = not pause
            
            window.show()

def init_taichi():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--arch",
        required=False,
        default="cpu",
        dest="arch",
        type=str,
        help="The arch (backend) to run this example on",
    )
    args, unknowns = parser.parse_known_args()
    arch = args.arch
    if arch in ["x64", "cpu", "arm64"]:
        ti.init(arch=ti.cpu, log_level=ti.ERROR)  # Set log_level to ERROR to suppress info/debug messages
    elif arch in ["cuda", "gpu"]:
        ti.init(arch=ti.cuda, log_level=ti.ERROR)  # Set log_level to ERROR to suppress info/debug messages
    else:
        raise ValueError("Only CPU and CUDA backends are supported for now.")

def run_simulation(param_idx, param, N_ROW, N_COLUMN, MASS_RANGE, STIFFNESS_RANGE, DAMPING_RANGE, dataset_dir, DATA_COLLECTION):
    """Run a single simulation with the given parameters"""
    try:
        # Initialize Taichi for this process
        ti.init(arch=ti.cpu, log_level=ti.ERROR)
        
        video_name = f"v{param_idx:06d}"

        if param == 0:
            # Create beam instance with region parameters
            beam = Beam(N_row=N_ROW, 
                        N_column=N_COLUMN, 
                        mass_range=MASS_RANGE,
                        stiffness_range=STIFFNESS_RANGE,
                        damping_range=DAMPING_RANGE,
                        use_regions=True,
                        seed=param_idx)
            
        else:
            # Create beam instance with unified parameters
            beam = Beam(N_row=N_ROW, 
                        N_column=N_COLUMN, 
                        mass_range=MASS_RANGE,
                        stiffness_range=STIFFNESS_RANGE,
                        damping_range=DAMPING_RANGE,
                        mass_value=param[0],
                        stiffness_value=param[1],
                        damping_value=param[2],
                        use_regions=False)
                    
        # Run simulation and save video
        simulate(beam, 
                save_to_video=True, 
                data_collection=DATA_COLLECTION,
                video_name=video_name,
                data_dir=dataset_dir)
        
        # Clean up
        del beam
        ti.reset()
        
        return True
    except Exception as e:
        print(f"Error in simulation {param_idx}: {e}")
        return False
    finally:
        # Ensure Taichi is reset even if an exception occurs
        try:
            ti.reset()
        except:
            pass

def main():
    MASS_RANGE = [0.2, 0.02]
    STIFFNESS_RANGE = [10000.0, 30000.0]
    DAMPING_RANGE = [50.0, 500.0]
    N_ROW = 8
    N_COLUMN = 20
    DATA_COLLECTION = True
    BEAM_TYPE = "region" # "region" or "uni"

    # Initialize Taichi for the main process
    init_taichi()
    
    # dataset directory
    if DATA_COLLECTION:
        if BEAM_TYPE == "region":
            dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset_beam_region')
        elif BEAM_TYPE == "uni":
            dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset_beam_uni')
        else:
            raise ValueError("Invalid beam type")
    else:
        dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset_temp')

    raw_dir = os.path.join(dataset_dir, 'raw')
    input_dir = os.path.join(dataset_dir, 'input')
    label_dir = os.path.join(dataset_dir, 'label')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    batch_size = 1000
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_idx", type=int, default=0)
    args = parser.parse_args()
    batch_idx = args.batch_idx
    index_list = list(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))

    if BEAM_TYPE == "uni":
        N_sample = 2 # N_sample ** 3 dataset
        mass_values = np.linspace(MASS_RANGE[0], MASS_RANGE[1], N_sample)
        stiffness_values = np.linspace(STIFFNESS_RANGE[0], STIFFNESS_RANGE[1], N_sample)
        damping_values = np.linspace(DAMPING_RANGE[0], DAMPING_RANGE[1], N_sample)
        param_list = list(product(mass_values, stiffness_values, damping_values))
        param_list_batch = param_list[batch_idx * batch_size:(batch_idx + 1) * batch_size]
    elif BEAM_TYPE == "region":
        param_list_batch = [0] * batch_size
    else:
        raise ValueError("Invalid beam type")

    # Use ProcessPoolExecutor to run simulations in parallel
    if DATA_COLLECTION:
        # Track failed simulations
        max_workers = 4
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = []
            for param_idx, param in zip(index_list, param_list_batch):
                future = executor.submit(
                    run_simulation, 
                    param_idx, 
                    param, 
                    N_ROW, 
                    N_COLUMN, 
                    MASS_RANGE, 
                    STIFFNESS_RANGE, 
                    DAMPING_RANGE, 
                    dataset_dir, 
                    DATA_COLLECTION
                )
                futures.append(future)
            
            # Process results with progress bar using as_completed
            from concurrent.futures import as_completed 
            completed = 0
            total = len(futures)
            
            with tqdm(total=total, desc="Simulations") as pbar:
                for future in as_completed(futures):
                    try:
                        success = future.result()
                        if not success:
                            print(f"Simulation failed")
                    except Exception as e:
                        print(f"Error processing result: {e}")
                    
                    completed += 1  
                    pbar.update(1)
        print(f"Batch {batch_idx} completed")

    else:
        for param_idx, param in zip(index_list, param_list_batch):
            run_simulation(param_idx, param, N_ROW, N_COLUMN, MASS_RANGE, STIFFNESS_RANGE, DAMPING_RANGE, dataset_dir, DATA_COLLECTION)

    # terminate the program
    sys.exit()

if __name__ == "__main__":
    main() 