import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from shapely.geometry import Point, Polygon
import itertools
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, KDTree
#利用概率密度函数打点，找凸包
#如果要生成凹体，可以做凸包 与 凸包的补集的交集
class BlobGenerator:
    def __init__(self):
        pass

    def generate(self, mean = [0, 0, 0], cov = np.diag([3.0, 3.0, 3.0]), num_points = 40, rmax = 30):
        points = np.random.multivariate_normal(mean, cov, num_points)
        dist_sq = np.sum((points - mean) ** 2, axis = 1)
        in_range_mask = dist_sq <= (rmax**2)
        points = points[in_range_mask]
        hull = ConvexHull(points)
        return hull
    
    def check_avail(self, point, hull, tol = 1e-12):
        A = hull.equations[:, :3]
        B = hull. equations[:, 3]
        return np.all(A.dot(point) + B <= tol)


    def to_grid(self, r, cov, num_points):
        """
        Map the shape from [-1,1] → [0, shape_size], then center it
        in overall_size x overall_size. Returns grid with 1 = inside blob.
        """
        hull = self.generate(mean = [r, r, r], cov = cov, num_points = num_points, rmax = r)
        grid = np.zeros([int(2*r), int(2*r), int(2*r)])
        for idx in np.ndindex(grid.shape):
            if self.check_avail(idx, hull, tol = 1e-12) == 1:
                grid[idx] = 1
        return grid

    def to_mesh(self, grid):
        """
        Convert a binary grid to a mesh of vertices and edges.
        Returns vertices (Nx2) and edges (Mx2) as numpy arrays.
        """
        heights, rows, cols = grid.shape

        # The global vertex index for a point (r, c) in a (rows+1)x(cols+1) mesh.
        def vertex_id(hh, rr, cc):
            return hh * ((rows+1) * (cols + 1)) + rr * (cols+1) + cc

        edges_set = set()
        faces_set = set()
        trian_faces = [
            (0, 1, 5),  # Bottom
            (0, 4, 5),  # Top
            (0, 2, 6),  # Front
            (0, 4, 6),  # Back
            (2, 6, 7),  # Left
            (2, 3, 7),   # Right
            (1, 3, 7),
            (1, 5, 7),
            (0, 1, 3),
            (0, 2, 3),
            (4, 6, 7),
            (4, 5, 7)
        ]

        # 1) Collect edges from each filled cell
        for h in range(heights):
            for r in range(rows):
                for c in range(cols):
                    if grid[h, r, c] == 1:
                        v0 = vertex_id(h, r, c)
                        v1 = vertex_id(h, r,  c+1)
                        v2 = vertex_id(h, r+1, c)
                        v3 = vertex_id(h, r+1, c+1)
                        v4 = vertex_id(h+1, r, c)
                        v5 = vertex_id(h+1, r,  c+1)
                        v6 = vertex_id(h+1, r+1, c)
                        v7 = vertex_id(h+1, r+1, c+1)
                        corners = [v0, v1, v2, v3, v4, v5, v6, v7]
                        for vA, vB in itertools.combinations(corners, 2):
                            edges_set.add(tuple(sorted((vA, vB))))
                        for i, j, k in trian_faces:
                            face = tuple(sorted([corners[i], corners[j], corners[k]]))
                            if face in faces_set:
                                faces_set.remove(face)
                            else:
                                faces_set.add(face)


        # 2) Identify active vertices
        active_vertices = set()
        for (a, b) in edges_set:
            active_vertices.add(a)
            active_vertices.add(b)

        active_vertices_list = sorted(active_vertices)
        old_to_new = {old_id: i for i, old_id in enumerate(active_vertices_list)}

        # 3) Build the reindexed edges
        new_edges = []
        new_faces = []
        for (a, b) in edges_set:
            new_edges.append((old_to_new[a], old_to_new[b]))

        for (a, b, c) in faces_set:
            new_faces.append((old_to_new[a], old_to_new[b], old_to_new[c]))
            
        new_edges = np.array(new_edges, dtype=int)
        new_faces = np.array(new_faces, dtype = int)
        new_faces = new_faces.reshape(-1).astype(np.int32)
        new_face_ver = np.unique(new_faces)
        #这里的new_edges中起点和终点的序号是从0开始的，其实也就是将原来active_vertices_list的编号排了个序
        #例如new_edges中起点的序号是k，对应的就是new_vertices中的第k个点的坐标

        # 4) Construct the new vertex array (x=col, y=row, z = height)
        def id_to_hrc(v_id):
            return ((v_id % ((cols+1) * (rows + 1))) % (cols + 1), 
                    (v_id % ((cols+1) * (rows + 1))) // (cols + 1),
                    v_id // ((cols+1) * (rows + 1)))

        new_vertices = []
        for v_id in active_vertices_list:
            x, y, z = id_to_hrc(v_id)
            new_vertices.append((x, y, z))  # x=cc, y=rr

        new_vertices = np.array(new_vertices, dtype=float)

        return new_vertices, new_edges, new_faces, new_face_ver
    
    def assign_stiffness(self, points, num_samples, new_edges):
        num_points = points.shape[0]
        sampled_idx = [np.random.randint(num_points)]
        distances = np.full(num_points, np.inf)

        for _ in range(num_samples - 1):
            last_sampled = points[sampled_idx[-1]]
            dists = np.linalg.norm(points - last_sampled, axis=1)
            distances = np.minimum(distances, dists)
            next_idx = np.argmax(distances)
            sampled_idx.append(next_idx)
        
        sampled_points = points[sampled_idx]
        stiffness_value = np.random.uniform(1000, 100000, size = num_samples)
        kdtree = KDTree(sampled_points)
        k = 5
        distances, indices = kdtree.query(points, k = k)

        weights = 1.0 / (distances + 1e-6)
        weights /= np.sum(weights, axis = 1, keepdims = True)
        vertex_stiffness = np.sum(weights * stiffness_value[indices], axis = 1)

        # 计算每条边的刚度
        edge_stiffness = (vertex_stiffness[new_edges[:, 0]] + vertex_stiffness[new_edges[:, 1]]) / 2.0

        return edge_stiffness


if __name__ == "__main__":
    fig, axs = plt.subplots(1, 2, figsize=(8, 8), subplot_kw={'projection': '3d'})
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        # 1) Generate a 3D blob
        blob = BlobGenerator()

        # 2) Rasterize / to_mesh
        r = 20
        cov = np.diag([20.0, 20.0, 20.0])
        num_points = 40
        grid_data = blob.to_grid(r, cov, num_points)

        vertices, edges = blob.to_mesh(grid_data)

        # 3) 画 3D 图
        ax.set_title(f"Example {i+1}")

        # 取出所有点的坐标
        x_vals = vertices[:, 0]
        y_vals = vertices[:, 1]
        z_vals = vertices[:, 2]

        # 绘制所有边
        for (iA, iB) in edges:
            xA, yA, zA = vertices[iA]
            xB, yB, zB = vertices[iB]
            ax.plot([xA, xB], [yA, yB], [zA, zB], color='black', lw=0.5)

        # 若要绘制顶点（可选）
        # ax.scatter(x_vals, y_vals, z_vals, color='darkcyan', s=5)

        # 可选：设定 3D 坐标系的范围，让形状看起来更匀称
        ax.set_box_aspect((1, 1, 1))  # 让 x/y/z 轴保持 1:1:1 比例
        ax.set_xlim(x_vals.min(), x_vals.max())
        ax.set_ylim(y_vals.min(), y_vals.max())
        ax.set_zlim(z_vals.min(), z_vals.max())

    plt.tight_layout()
    plt.show()


