import os
import cv2
import numpy as np

# 参数设置
image_dir = "C:/tmp/evo_gym_dataset_0717/output_b1_tiny/filtered_img"  # 替换成你的图片所在路径
output_file = "merged_result.png"
batch_ids = list(range(10))  # 从 0 到 9
img_names_template = [
    "gt_epoch1_batch{}.png",
    "unfiltered_pred_epoch125_batch{}.png",
    "filtered_pred_epoch125_batch{}.png",
    "unfiltered_pred_epoch178_batch{}.png",
    "filtered_pred_epoch178_batch{}.png",
]

# 加载一张图片并统一大小
def load_image(path, target_shape=None):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"File not found: {path}")
    if target_shape:
        img = cv2.resize(img, target_shape)
    return img

# 拼接图像列（每列对应一个 batch）
columns = []
for batch_id in batch_ids:
    paths = [os.path.join(image_dir, name.format(batch_id)) for name in img_names_template]
    images = [load_image(p) for p in paths]
    base_shape = (images[0].shape[1], images[0].shape[0])
    resized_images = [cv2.resize(img, base_shape) for img in images]
    column = np.vstack(resized_images)
    columns.append(column)

# 拼接整张大图
merged_image = np.hstack(columns)
cv2.imwrite(output_file, merged_image)
print(f"Saved merged image to {output_file}")