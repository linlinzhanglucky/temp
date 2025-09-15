import open3d as o3d
import numpy as np


# extrinsic_matrix = np.array([
#     [ 0.0,  0.0,  -1.0,  1.0],
#     [ -1.0,  0.0,  0.0,  0.0],
#     [ 0.0,  1.0,  0.0,  0.0],
#     [ 0.0,  0.0,  0.0,  1.0]
# ])

# extrinsic_matrix = np.linalg.inv(extrinsic_matrix)

camera_pos = np.array([2.5, 2.0, 1.5], dtype=np.float32)
camera_lookat = np.array([1.0, 1.0, 1.5], dtype=np.float32)
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


# 创建 Open3D 可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Fixed Camera Visualization", width=800, height=600, visible=True)

# 读取点云数据
# sample_ply_data = o3d.data.PLYPointCloud()
# pcd = o3d.io.read_point_cloud(sample_ply_data.path)
# vis.add_geometry(pcd)

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
vis.add_geometry(axis)
# 获取相机控制器
ctr = vis.get_view_control()

# 获取当前相机参数
cam_params = ctr.convert_to_pinhole_camera_parameters()
intrinsic_matrix = cam_params.intrinsic
params = o3d.camera.PinholeCameraParameters()
params.intrinsic = intrinsic_matrix
params.extrinsic = extrinsic_matrix

print(extrinsic_matrix)
# 更新相机
ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
#ctr.reset_camera_local_rotate()
ctr_extrinsic_matrix = np.array(ctr.convert_to_pinhole_camera_parameters().extrinsic)
# 刷新渲染
vis.poll_events()
vis.update_renderer()
print(f"[INFO] Camera extrinsic matrix (after update):\n{ctr_extrinsic_matrix}")



# 运行可视化
vis.run()
vis.destroy_window()