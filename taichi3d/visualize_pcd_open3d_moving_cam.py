import open3d as o3d
import numpy as np
import os
import imageio

# 参数配置
steps_per_circle = 360 * 5
image_dir = "./frames_static_obj"
video_path = "./rotation_video.mp4"

# 全局帧计数器
frame = 0
vis_ref = None  # 保存 vis 对象引用

def rotate_and_capture(vis):
    global frame, vis_ref
    ctr = vis.get_view_control()

    if frame < steps_per_circle:
        ctr.rotate(1.0, 0.0)
    elif frame < 2 * steps_per_circle:
        ctr.rotate(0.0, 1.0)
    else:
        print("Finished capturing frames.")
        return False  # 停止动画

    # 截图保存
    filename = os.path.join(image_dir, f"frame_{frame:04d}.png")
    vis.capture_screen_image(filename, do_render=True)

    frame += 1
    return False

def alpha_shape_from_ply(ply_file, alpha=0.3):
    pcd = o3d.io.read_point_cloud(ply_file)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
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

def generate_video():
    print("Generating video...")
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")])
    images = [imageio.imread(f) for f in image_files]
    imageio.mimsave(video_path, images, fps=30)
    print(f"Video saved to {video_path}")

def main():
    global vis_ref

    # 创建图像保存目录
    # os.makedirs(image_dir, exist_ok=True)

    # ply_file = "./pcd/pcd_250.ply"
    # mesh = alpha_shape_from_ply(ply_file, alpha=0.3)

    # vis_ref = o3d.visualization.draw_geometries_with_animation_callback(
    #     [mesh],
    #     rotate_and_capture
    # )

    # # # 所有帧保存完后生成视频
    generate_video()

if __name__ == "__main__":
    main()
