from grid_3d_distribution_trian import BlobGenerator
import numpy as np
import taichi as ti
import open3d as o3d

#np_vertices = np_vertices - center
@ti.data_oriented
class voxelSobo:
    def __init__(self, np_vertices, np_edges, np_faces, np_face_ver, big_size = 1, real_size = 1):
        self.dt = 0.001      
        self.mass = 1.0   
        self.k_spring = 5000.0 
        self.damping = 2.0 
        self.gravity = ti.Vector([0.0, -9, 0.0])
        self.np_vertices = np_vertices
        self.np_edges = np_edges
        self.np_faces = np_faces
        self.np_face_ver = np_face_ver
        self.np_vertices_real = np_vertices / big_size * real_size
        self.np_center = np.mean(self.np_vertices_real, axis = 0)
        self.real_size = real_size

        self.num_ver = np_vertices.shape[0]
        self.num_edg = np_edges.shape[0]
        self.num_face = np_faces.shape[0]
        self.num_face_ver = np_face_ver.shape[0]

        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=self.num_ver)      
        self.vel = ti.Vector.field(3, dtype=ti.f32, shape=self.num_ver)     
        self.rest_length = ti.field(dtype=ti.f32, shape=self.num_edg)        
        self.edge_ends = ti.Vector.field(2, dtype=ti.i32, shape=self.num_edg)
        self.face_ends = ti.field(dtype=ti.i32, shape=self.num_face)  # 纯1D field
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=self.num_edg)
        self.springs = ti.Vector.field(3, dtype=ti.f32, shape = 2 * self.num_edg)
        self.face_ver = ti.field(dtype=ti.i32, shape=self.num_face_ver)
        self.center = ti.Vector.field(3, dtype=ti.f32, shape=1)
        self.sum_pos = ti.Vector.field(3, dtype=ti.f32, shape=1)

        self.pos.from_numpy(self.np_vertices_real)
        self.edge_ends.from_numpy(np_edges)
        self.face_ends.from_numpy(np_faces)
        self.face_ver.from_numpy(np_face_ver)
        self.face_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.num_face_ver)

    @ti.kernel
    def compute_rest_length(self):
        for e in range(self.num_edg):
            iA = self.edge_ends[e][0]
            iB = self.edge_ends[e][1]
            self.rest_length[e] = (self.pos[iA] - self.pos[iB]).norm()

    @ti.kernel
    def clear_vel(self):
        for i in range(self.num_ver):
            self.vel[i] = ti.Vector([0.0, 0.0, 0.0])
    #@ti.kernel
    def initialize(self):
        self.compute_rest_length()
        self.clear_vel()

        
    @ti.kernel
    def substep(self):
        # 1. 准备 force 数组(局部)
        for i in range(self.num_edg):
            self.force[i] = ti.Vector([0.0, 0.0, 0.0])

        # 2. 计算弹簧力
        for e in range(self.num_edg):
            iA = self.edge_ends[e][0]
            iB = self.edge_ends[e][1]
            pA = self.pos[iA]
            pB = self.pos[iB]
            current_len = (pB - pA).norm()
            # 如果 current_len == 0，需特殊处理，这里假设不会发生
            dirAB = (pB - pA) / current_len  # 单位方向

            # Hooke定律: F = -k*(L-L0)*dir
            elongation = current_len - self.rest_length[e]
            F = self.k_spring * elongation * dirAB

            # 作用于 A, 反作用于 B
            ti.atomic_add(self.force[iA],  F)
            ti.atomic_add(self.force[iB], -F)

        # 3. 加上重力和阻尼
        for i in range(self.num_ver):
            # 重力
            f_gravity = self.mass * self.gravity
            # 简单的速度阻尼: F_damp = -damping * vel[i]
            f_damp = -self.damping * self.vel[i]

            self.force[i] += f_gravity + f_damp

        # 4. 更新速度和位置 (显式Euler)
        for i in range(self.num_ver):
            acc = self.force[i] / self.mass
            self.vel[i] += acc * self.dt
            self.pos[i] += self.vel[i] * self.dt
            if self.pos[i][1] < 0:
                self.pos[i][1] = 0
                self.vel[i][1] = -1/2 * self.vel[i][1]

        for i in range(self.num_edg):
            iA = self.edge_ends[i][0]
            iB = self.edge_ends[i][1]
            self.springs[2*i]     = self.pos[iA]
            self.springs[2*i + 1] = self.pos[iB]

        #update center
        self.sum_pos[0] = ti.Vector([0.0, 0.0, 0.0])
        for i in range(self.num_ver):
            self.sum_pos[0] += self.pos[i]
        self.center[0] = self.sum_pos[0] / self.num_ver
        
        #update face_pos
        for i in range(self.num_face_ver):
            self.face_pos[i] = self.pos[self.face_ver[i]]

        

    def run(self):

        window = ti.ui.Window("Mass-Spring Demo (3D)", (1024, 768), vsync=True)
        canvas = window.get_canvas()
        canvas.set_background_color((0.2, 0.2, 0.2))

        scene = ti.ui.Scene()
        camera = ti.ui.Camera()
        camera.position(self.center[0][0] + self.real_size*3, self.center[0][1] + self.real_size, self.center[0][2] + self.real_size*3)
        camera.lookat(self.center[0][0], self.center[0][1], self.center[0][2])
        camera.up(0, 1, 0)

        # 可根据需要设置相机初始位置、目标、上方向

        while window.running:
            # 物理多步更新, 让系统稳定一些
            for _ in range(10):
                self.substep()
            
            # 更新相机 (可交互控制, 也可固定)
            camera.track_user_inputs(window, movement_speed=0.01, hold_key=ti.ui.RMB)

            # 设置相机到 scene
            scene.set_camera(camera)
            scene.ambient_light((0.5, 0.5, 0.5))
            scene.point_light(pos=(0, 50, 50), color=(1, 1, 1))

            # 渲染质点 (pos是ti.field(shape=N), 类型ti.Vector(3, f32))
            #scene.particles(self.pos, radius=0.01, color=(0.8, 0.2, 0.2))
            #scene.lines(self.springs, 2.0, color = (0.8, 0.8, 0.8))
            scene.mesh(self.pos,
                   indices=self.face_ends,
                   two_sided=False,
                   color=(0.2, 0.6, 1.0),
                   show_wireframe = True)


            # 把场景画到画布
            canvas.scene(scene)
            window.show()

if __name__ == "__main__":
    ti.init()
    blob = BlobGenerator()

    # create vertices and edges
    r = 20
    cov = np.diag([10.0, 10.0, 10.0])
    num_points = 50
    grid_data = blob.to_grid(r, cov, num_points)
    real_size = 3

    np_vertices, np_edges, np_faces, np_face_ver = blob.to_mesh(grid_data)
    voxelSobo_1 = voxelSobo(np_vertices = np_vertices, np_edges = np_edges, np_faces = np_faces, np_face_ver = np_face_ver, big_size=r, real_size = real_size)
    voxelSobo_1.initialize()
    voxelSobo_1.run()
    final_face_pos = voxelSobo_1.face_pos.to_numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_face_pos)
    alpha = 0.3
    mesh_alpha = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh_alpha.compute_vertex_normals()

    # 5. 显示重建网格
    o3d.visualization.draw([mesh_alpha])



# 