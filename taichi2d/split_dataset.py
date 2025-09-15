# === 1. split_dataset.py ===
import os
import random
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="dataset_uni", help="dataset name")
args = parser.parse_args()

input_root = rf"{args.dataset}/input"
label_root = rf"{args.dataset}/label"
train_dir = rf"{args.dataset}/train"
val_dir = rf"{args.dataset}/val"
os.makedirs(train_dir + "/input", exist_ok=True)
os.makedirs(train_dir + "/label", exist_ok=True)
os.makedirs(val_dir + "/input", exist_ok=True)
os.makedirs(val_dir + "/label", exist_ok=True)

videos = sorted([v for v in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, v))])
random.seed(42)
random.shuffle(videos)

# complete dataset
split_idx = int(0.8 * len(videos))
train_videos = videos[:split_idx]
val_videos = videos[split_idx:]

# 5000 data experiment
# train_videos = videos[:8000]
# val_videos = videos[8000:10000]

for vid in train_videos:
    shutil.copytree(os.path.join(input_root, vid), os.path.join(train_dir, "input", vid))
    shutil.copytree(os.path.join(label_root, vid), os.path.join(train_dir, "label", vid))
for vid in val_videos:
    shutil.copytree(os.path.join(input_root, vid), os.path.join(val_dir, "input", vid))
    shutil.copytree(os.path.join(label_root, vid), os.path.join(val_dir, "label", vid))
print("✅ Dataset split complete!")