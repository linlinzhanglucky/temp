import os
import cv2
import numpy as np
def merge_similar_colors(color_set, threshold=10):
    """
    将 color_set 中彼此距离小于 threshold 的颜色合并，返回新的颜色列表。
    """
    color_list = list(color_set)
    merged_colors = []
    visited = [False] * len(color_list)

    for i in range(len(color_list)):
        if visited[i]:
            continue

        base_color = np.array(color_list[i])
        group = [base_color]
        visited[i] = True

        for j in range(i+1, len(color_list)):
            if visited[j]:
                continue

            dist = np.linalg.norm(base_color - np.array(color_list[j]))
            if dist < threshold:
                group.append(np.array(color_list[j]))
                visited[j] = True

        # 平均颜色作为代表色
        mean_color = np.mean(group, axis=0).astype(int)
        merged_colors.append(tuple(mean_color))

    return merged_colors

def get_unique_rgb_values(folder_path):
    unique_colors = set()

    for filename in os.listdir(folder_path):
        if filename.lower().startswith("gt") and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)  # BGR by default

            if img is None:
                print(f"Warning: Failed to read {img_path}")
                continue

            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Reshape and add to set
            pixels = img_rgb.reshape(-1, 3)
            for color in pixels:
                unique_colors.add(tuple(color))  # Convert np.array to tuple so it can be added to set

    return list(unique_colors)  # Optional: convert back to list

def replace_with_closest_color(img_rgb, robot_shape, color_list):
    """将每个像素替换为 color_list 中最接近的颜色"""
    h, w, _ = img_rgb.shape
    replaced_img = np.ones_like(img_rgb, dtype=np.uint8) * 255
    h_robot, w_robot = robot_shape.shape
    h_start = h / 8 * (8 - h_robot) // 2
    w_start = w / 8 * (8 - w_robot) // 2
    for i in range(h_robot):
        for j in range(w_robot):
            if robot_shape[i, j] == 0:
                continue
            voxel_img = img_rgb[int(h_start + i*h/8): int(h_start + (i+1)*h/8), int(w_start + j*w/8): int(w_start + (j+1)*w/8)]
            voxel_mean = np.mean(voxel_img, axis=(0, 1))
            nearest_color = min(color_list, key=lambda c: np.linalg.norm(np.array(c) - voxel_mean))
            replaced_img[int(h_start + i*h/8): int(h_start + (i+1)*h/8), int(w_start + j*w/8): int(w_start + (j+1)*w/8)] = nearest_color
    return replaced_img

def process_images_with_color_mapping(input_folder, output_folder, unique_color_list, robot_shape):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if "epoch125" in filename.lower() and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(input_folder, filename)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"Failed to read image: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            replaced_img_rgb = replace_with_closest_color(img_rgb, robot_shape, unique_color_list)

            # Convert back to BGR for saving
            replaced_img_bgr = cv2.cvtColor(replaced_img_rgb, cv2.COLOR_RGB2BGR)

            save_path_filtered = os.path.join(output_folder, "filtered_" + filename)
            save_path_unfiltered = os.path.join(output_folder, "unfiltered_" + filename)
            cv2.imwrite(save_path_filtered, replaced_img_bgr)
            cv2.imwrite(save_path_unfiltered, img_bgr)
            # print(f"Saved processed image: {save_path}")
            print(f"Processed and saved: {filename}")

# 示例用法
folder = "C:/tmp/evo_gym_dataset_0717/output_b1_tiny/images"
unique_colors = get_unique_rgb_values(folder)
unique_colors = merge_similar_colors(unique_colors)
print(unique_colors)
input_folder = "C:/tmp/evo_gym_dataset_0717/output_b1_tiny/images"
output_folder = "C:/tmp/evo_gym_dataset_0717/output_b1_tiny/filtered_img"
robot_shape = np.ones((3, 4))
robot_shape[1:3, 1:3] = 0

process_images_with_color_mapping(input_folder, output_folder, unique_colors, robot_shape)