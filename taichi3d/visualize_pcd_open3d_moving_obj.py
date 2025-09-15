# Usage: python visualize_pcd.py
import open3d as o3d
import numpy as np
import imageio
import glob
import time
import os
import cv2
def create_ground_lines(plane_size=2.5, plane_y=0.0, grid_size=0.5):
    """
    用 LineSet 创建一个网格状的地面，不使用 TriangleMesh。
    grid_size: 网格的大小
    """
    import itertools

    # 生成网格线条的点
    num_lines = int(plane_size / grid_size) * 2
    points = []
    lines = []

    for i in range(-num_lines, num_lines + 1):
        x = i * grid_size
        z_min, z_max = -plane_size, plane_size
        x_min, x_max = -plane_size, plane_size
        points.append([x, plane_y, z_min])
        points.append([x, plane_y, z_max])
        points.append([x_min, plane_y, x])
        points.append([x_max, plane_y, x])

    # 生成线索引
    lines = [[i, i + 1] for i in range(0, len(points), 2)]
    
    # 创建 LineSet
    ground_lines = o3d.geometry.LineSet()
    ground_lines.points = o3d.utility.Vector3dVector(points)
    ground_lines.lines = o3d.utility.Vector2iVector(lines)
    
    return ground_lines

def create_ground_boundingbox(plane_size=1.5, plane_y=0.0):
    """
    用 BoundingBox 可视化地面（不会受网格误差影响）。
    """
    min_bound = np.array([0, plane_y, 0])
    max_bound = np.array([plane_size, plane_y - 0.01, plane_size])
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    bbox.color = (0.7, 0.7, 0.7)  # 灰色
    return bbox

def create_ground_plane(plane_size=1.0, plane_y=0.0):
    """
    创建一个矩形地面平面 (2 个三角形)，
    位于 y = plane_y 的位置，长度宽度都为 2 * plane_size。
    """
    import open3d as o3d
    import numpy as np
    
    # 4 个顶点，x,z 从 -plane_size 到 +plane_size，y 固定为 plane_y
    vertices = np.array([
        [-plane_size, plane_y, -plane_size],
        [-plane_size, plane_y,  plane_size],
        [ plane_size, plane_y,  plane_size],
        [ plane_size, plane_y, -plane_size],
    ], dtype=np.float32)

    # 两个三角面
    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=np.int32)

    # 构造 TriangleMesh
    plane_mesh = o3d.geometry.TriangleMesh()
    plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    plane_mesh.compute_vertex_normals()

    # （可选）给地面一个简单灰色或其他颜色
    plane_mesh.paint_uniform_color([0.7, 0.7, 0.7])

    return plane_mesh

def alpha_shape_from_ply(ply_file, alpha=0.3):
    """
    读取 .ply 点云 (而非网格)，进行 Alpha Shape 重建得到三角网格。
    """
    pcd = o3d.io.read_point_cloud(ply_file)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    # 手动做最近邻映射，把点云颜色映射到 mesh 顶点
    print(pcd.has_colors())
    if pcd.has_colors():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        pcd_points = np.asarray(pcd.points)
        pcd_colors = np.asarray(pcd.colors)
        print(pcd.colors)
        mesh_vertices = np.asarray(mesh.vertices)

        from scipy.spatial import cKDTree
        kdtree = cKDTree(pcd_points)
        _, idxs = kdtree.query(mesh_vertices, k=1)
        mesh_colors = pcd_colors[idxs]
        mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh

def visualize_mesh_fixed_camera(mesh, screenshot_path):
    """
    打开 Open3D 窗口显示 mesh
    相机位置 eye = [5,2,5], 看向 center = [1,1,1], 设缩放 = 两点距离。
    截屏保存到 screenshot_path 然后关闭窗口。
    """

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Fixed Camera Visualization",
        width=800,
        height=600,
        visible=True
    )

    #plane_mesh = create_ground_plane(plane_size=3.5, plane_y=0.0)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    ground_bbox = create_ground_boundingbox()
    ground_lines = create_ground_lines()

    ctr = vis.get_view_control()
    ctr.set_zoom(1.0)
    ctr.set_constant_z_near(0.1)
    ctr.set_constant_z_far(10.0)
    vis.poll_events()
    vis.update_renderer()

    camera_pos = np.array([4.5, 3.0, 1.0], dtype=np.float32)
    camera_lookat = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    camera_z = camera_lookat - camera_pos
    camera_z = camera_z / np.linalg.norm(camera_z)
    camera_x = np.array([0, 0, -1], dtype=np.float32)
    camera_y = np.cross(camera_z, camera_x)
    camera_y = camera_y / np.linalg.norm(camera_y)

    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, 0] = camera_x
    extrinsic_matrix[:3, 1] = camera_y
    extrinsic_matrix[:3, 2] = camera_z
    extrinsic_matrix[:3, 3] = camera_pos
    extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
    
    
    ctr.scale(1.0)
    ctr.set_constant_z_near(0.1)
    ctr.set_constant_z_far(10.0)
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    intrinsic_matrix = cam_params.intrinsic
    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic = intrinsic_matrix
    params.extrinsic = extrinsic_matrix
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    params2 = o3d.camera.PinholeCameraIntrinsic().intrinsic_matrix
    ctr_extrinsic_matrix = np.array(ctr.convert_to_pinhole_camera_parameters().extrinsic)
    ctr_intrinsic_matrix = np.array(params2)
    ctr.set_zoom(1.0)
    #print(f"[INFO] Camera extrinsic matrix (after update):\n{ctr_extrinsic_matrix}")
    #print(f"[INFO] Camera intrinsic matrix:\n{ctr_intrinsic_matrix}")


    # 相机固定参数
    # eye = np.array([2.5, 1.5, 2.5], dtype=np.float32)
    # center = np.array([1, 1, 1], dtype=np.float32)
    # up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # front = center - eye
    # front_norm = front / np.linalg.norm(front) if np.linalg.norm(front) > 1e-8 else [0,0,-1]

    # ctrl = vis.get_view_control()
    # #ctrl.set_lookat(center / np.linalg.norm(center))      # 看向中心
    # ctrl.set_front(front_norm)   # 相机正面向量
    # ctrl.set_up(up / np.linalg.norm(up))
    # distance = np.linalg.norm(center - eye)  # 距离
    #ctrl.set_zoom(distance*0.8)

    # cam_params = ctrl.convert_to_pinhole_camera_parameters()
    # extrinsic_matrix = np.array(cam_params.extrinsic, dtype=np.float32)
    # print(f"[INFO] Camera extrinsic matrix:\n{extrinsic_matrix}")

    # 截屏
    for i in range(10):
        vis.poll_events()
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    vis.add_geometry(ground_bbox)
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    vis.add_geometry(mesh)
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    vis.add_geometry(axis)
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    vis.poll_events()


    #vis.update_renderer()
    vis.capture_screen_image(screenshot_path)
    print(f"[INFO] Saved screenshot => {screenshot_path}")
    ctr_extrinsic_matrix = np.array(ctr.convert_to_pinhole_camera_parameters().extrinsic)
    print(f"[INFO] Camera extrinsic matrix (after update):\n{ctr_extrinsic_matrix}")

    # 关闭窗口
    vis.destroy_window()
    # 可选: 给一点延时，确保截图文件保存
    time.sleep(0.1)

def main():
    # 1) 假设你的 .ply 文件按 "pcd_0.ply", "pcd_1.ply", ... "pcd_9999.ply" 命名
    #    也可改用 glob 或其他方法收集
    num_frames = 300
    input_dir = "./pcd"
    output_dir = "./pic"
    os.makedirs(output_dir, exist_ok=True)

    # 2) 循环处理
    temp_imgs = []
    for i in range(num_frames):
        ply_file = os.path.join(input_dir, f"pcd_{i}.ply")
        if not os.path.exists(ply_file):
            print(f"[WARN] {ply_file} 不存在，跳过！")
            continue

        print(f"Processing {ply_file}...")
        # (a) Alpha Shape 重建
        mesh = alpha_shape_from_ply(ply_file, alpha=0.3)

        # (b) 可视化并截屏
        out_png = os.path.join(output_dir, f"frame_{i:05d}.png")
        visualize_mesh_fixed_camera(mesh, out_png)
        temp_imgs.append(out_png)

    # 3) 合成 GIF
    if temp_imgs:
        gif_path = "result.gif"
        with imageio.get_writer(gif_path, mode="I", duration=0.001) as writer:
            for png_file in temp_imgs:
                if os.path.exists(png_file):
                    frame = imageio.v2.imread(png_file)
                    writer.append_data(frame)
        print(f"[INFO] Saved GIF => {gif_path}")
    else:
        print("[INFO] No frames were generated. GIF not created.")


def images_to_gif(input_dir="./pic", output_gif="result.gif", duration=0.001):
    """
    从 input_dir 中读取所有名为 'frame_*.png' 的文件，
    按顺序合成为 GIF 并保存到 output_gif。

    :param input_dir: 存放 PNG 图片的文件夹
    :param output_gif: 输出 GIF 文件路径
    :param duration: 每帧显示时长（秒）
    """

    # 1) 收集所有符合 "frame_*.png" 格式的文件，按文件名排序
    png_files = sorted(glob.glob(os.path.join(input_dir, "frame_*.png")))
    
    if not png_files:
        print(f"[WARN] No 'frame_*.png' images found in '{input_dir}'!")
        return

    print(f"[INFO] Found {len(png_files)} PNG files. Merging into GIF...")

    # 2) 创建 GIF
    with imageio.get_writer(output_gif, mode="I", duration=duration) as writer:
        for png_file in png_files:
            frame = imageio.v2.imread(png_file)
            writer.append_data(frame)

    print(f"[INFO] GIF saved => {output_gif}")

if __name__ == "__main__":
    # 调用示例
    main()
    #images_to_gif(input_dir="./pic", output_gif="result.gif", duration=0.01)

