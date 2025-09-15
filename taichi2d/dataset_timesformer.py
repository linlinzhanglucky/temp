# === dataset_timesformer.py ===
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class BWVideoColorizationDataset(Dataset):
    def __init__(self, input_root, label_root, num_frames=8, image_size=224):
        self.input_root = input_root
        self.label_root = label_root
        self.num_frames = num_frames
        self.image_size = image_size

        self.to_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor() # 会自动变成 [C, H, W]，灰度图是 [1, H, W]
        ])

        # ✅ 过滤掉无效的样本（必须保证 input 存在 + label/f000000.png 存在）
        self.videos = []
        for vid in sorted(os.listdir(input_root)):
            input_dir = os.path.join(input_root, vid)
            label_path = os.path.join(label_root, vid, "f000000.png")
            if not os.path.isdir(input_dir):
                continue
            if not os.path.exists(label_path):
                continue
            frame_list = [f for f in os.listdir(input_dir) if f.endswith(".png")]
            if len(frame_list) >= num_frames:
                self.videos.append(vid)

        if len(self.videos) == 0:
            raise RuntimeError(f"No valid videos found in {input_root} with {num_frames}+ frames and matching labels.")

    def __len__(self):
        return len(self.videos)

    # def __getitem__(self, idx):
    #     vid = self.videos[idx]
    #     input_folder = os.path.join(self.input_root, vid)
    #     label_path = os.path.join(self.label_root, vid, "f000000.png")

    #     frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])[:self.num_frames]
    #     gray_video = []
    #     for f in frame_files:
    #         img = Image.open(os.path.join(input_folder, f)).convert("L").convert("RGB")

    #         gray_video.append(self.to_tensor(img)) # [1, H, W]

    #     color_target = self.to_tensor(Image.open(label_path).convert("RGB"))

    #     return {
    #         "gray_video": torch.stack(gray_video),  # [T, 3, H, W]
    #         "target_frame": color_target,           # [3, H, W]
    #         "video_name": vid
    #     }

    # DEBUG：change to binary now
    def __getitem__(self, idx):
        vid = self.videos[idx]
        input_folder = os.path.join(self.input_root, vid)
        label_path = os.path.join(self.label_root, vid, "f000000.png")

        frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])[:self.num_frames]
        gray_video = []
        for f in frame_files:
            img = Image.open(os.path.join(input_folder, f)).convert("L")  # ✅ 改成灰度
            img = img.point(lambda p: 1.0 if p > 128 else 0.0)  # ✅ 二值化
            gray_video.append(self.to_tensor(img))  # [1, H, W]

        # target_img = Image.open(label_path).convert("L")  # ✅ 目标帧也改成灰度
        # target_img = target_img.point(lambda p: 1.0 if p > 128 else 0.0)  # ✅ 二值化
        # color_target = self.to_tensor(target_img)  # [1, H, W]
        color_target = self.to_tensor(Image.open(label_path).convert("RGB")) # [3, H, W]

        return {
            "gray_video": torch.stack(gray_video),  # [T, 1, H, W]
            # "target_frame": color_target,           # [1, H, W]
            "target_frame": color_target,           # [3, H, W]
            "video_name": vid
        }
