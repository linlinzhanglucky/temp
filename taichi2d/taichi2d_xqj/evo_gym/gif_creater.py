import os
# import imageio.v2 as imageio
from natsort import natsorted  # 自动按数字排序
from tqdm import tqdm
import numpy as np
import cv2
import imageio.v2 as imageio  # 确保使用imageio的v2版本

def create_gif_from_folder(folder_path, gif_path, duration=0.05):
    """从一个文件夹读取所有PNG图片，并保存为GIF"""
    images = []
    filenames = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    filenames = natsorted(filenames)  # 自动按f000000.png顺序排序

    for filename in filenames:
        img_path = os.path.join(folder_path, filename)
        images.append(imageio.imread(img_path))
    
    if images:
        imageio.mimsave(gif_path, images, duration=duration)


def batch_process_gif(root_folder, n=5, subfolder_type='input'):
    """批处理指定类型子目录（input或raw）下的前n个V文件夹"""
    target_dir = os.path.join(root_folder, subfolder_type)
    subfolders = natsorted([f for f in os.listdir(target_dir) if f.startswith('V')])
    
    for vname in tqdm(subfolders[:n], desc=f"Processing {subfolder_type}"):
        folder_path = os.path.join(target_dir, vname)
        gif_path = os.path.join(folder_path, 'preview.gif')
        create_gif_from_folder(folder_path, gif_path)

def load_frame_sequence(folder, seq_id, num_frames=40, repeat_first_only=False):
    frame_list = []
    seq_path = os.path.join(folder, f"V{seq_id:06d}")
    
    if repeat_first_only:
        img = cv2.imread(os.path.join(seq_path, "f000000.png"))
        frame_list = [img] * num_frames
    else:
        for i in range(num_frames):
            frame_name = f"f{i:06d}.png"
            img_path = os.path.join(seq_path, frame_name)
            img = cv2.imread(img_path)
            frame_list.append(img)
    
    return frame_list  # list of images

def generate_video_batch(raw_folder, input_folder, label_folder, output_path, num_sequences=8, num_frames=40, fps=10):
    all_rows_per_frame = []  # Will be a list of video frames

    for frame_idx in range(num_frames):
        row_images = []  # one image per column (i.e. V000000 to V00000n)

        for seq_id in range(num_sequences):
            raw_seq = load_frame_sequence(raw_folder, seq_id, num_frames)[frame_idx]
            input_seq = load_frame_sequence(input_folder, seq_id, num_frames)[frame_idx]
            label_seq = load_frame_sequence(label_folder, seq_id, num_frames, repeat_first_only=True)[frame_idx]

            col = np.vstack([raw_seq, input_seq, label_seq])
            row_images.append(col)

        full_frame = np.hstack(row_images)
        all_rows_per_frame.append(full_frame)

    # write to video
    height, width, _ = all_rows_per_frame[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in all_rows_per_frame:
        out.write(frame)
    
    out.release()
    print(f"Video saved to: {output_path}")

if __name__ == "__main__":
    # 将一个文件夹下的PNG转为GIF
    root_dir = "C:/tmp/evo_gym_dataset_0720"
    n = 10  # 只处理前n个文件夹

    batch_process_gif(root_dir, n=n, subfolder_type='input')
    batch_process_gif(root_dir, n=n, subfolder_type='raw')


    # # 读取多个文件夹下的图像，每个文件夹下的图像为一个视频序列，生成一个大的视频
    # generate_video_batch(
    #     raw_folder="C:/tmp/evo_gym_dataset_0717/raw",
    #     input_folder="C:/tmp/evo_gym_dataset_0717/input",
    #     label_folder="C:/tmp/evo_gym_dataset_0717/label",
    #     output_path="output.mp4",
    #     num_sequences=6,    # 列数
    #     num_frames=40,      # 每列帧数
    #     fps=10              # 可调节
    # )


    # 读取多个文件夹下的图像，生成一个大的图像
