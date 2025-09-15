# === dataset_timesformer.py ===
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class BWVideoColorizationDataset(Dataset):
    def __init__(self, input_root, label_root,
                 num_frames=8, image_size=224,
                 preload=False, device = torch.device("cuda")):                 # ✨ 新增
        self.input_root = input_root
        self.label_root = label_root
        self.num_frames = num_frames
        self.image_size = image_size
        self.preload = preload

        self.to_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        # ------- 收集合法视频 id -------
        self.videos = []
        for vid in sorted(os.listdir(input_root)):
            in_dir  = os.path.join(input_root, vid)
            lbl_png = os.path.join(label_root, vid, "f000000.png")
            if not (os.path.isdir(in_dir) and os.path.exists(lbl_png)):
                continue
            if len([f for f in os.listdir(in_dir) if f.endswith(".png")]) >= num_frames:
                self.videos.append(vid)
        if not self.videos:
            raise RuntimeError("No valid samples found.")

        # ------- 可选：一次性预加载 -------
        self.cache = []
        if self.preload:
            print(f"🔄 Pre‐loading {len(self.videos)} videos into RAM …")
            for vid in self.videos:
                self.cache.append(self._load_one_video(vid))
            print("✅ Pre‐loading complete.")

        
    def _load_one_video(self, vid):
        in_dir  = os.path.join(self.input_root, vid)
        lbl_png = os.path.join(self.label_root, vid, "f000000.png")

        frames = sorted(f for f in os.listdir(in_dir) if f.endswith(".png"))[:self.num_frames]
        gray_video = []
        for f in frames:
            img = Image.open(os.path.join(in_dir, f)).convert("L")
            img = img.point(lambda p: 1.0 if p > 128 else 0.0)   # ⬅️ 二值化
            gray_video.append(self.to_tensor(img))               # [1,H,W]
        gray_video = torch.stack(gray_video)                     # [T,1,H,W]

        target = self.to_tensor(Image.open(lbl_png).convert("RGB"))  # [3,H,W]

        return {"gray_video": gray_video,
                "target_frame": target,
                "video_name": vid}

    # ----------------------------
    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.preload:
            return self.cache[idx]
        else:
            vid = self.videos[idx]
            return self._load_one_video(vid)
