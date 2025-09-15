#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
blend_two_folders.py
把 folder_A 与 folder_B 中命名相同的 PNG/JPG 做 50/50 混合，
结果写到 out_dir, 并顺便用 ffmpeg 拼成 mp4。
"""

import os, cv2, argparse, subprocess, shutil
from pathlib import Path
import natsort

# ---------- 使用参数 ----------
parser = argparse.ArgumentParser()
parser.add_argument("--dir", default= "./mpm_output_0601",  help="第一组图片文件夹")
parser.add_argument("--ext",  default="png", help="图片扩展名 (png / jpg …)")
parser.add_argument("--alpha",type=float, default=0.5, help="A 占的权重 (0~1)")
parser.add_argument("--fps",  type=int,   default=30,  help="合成视频帧率")
args = parser.parse_args()
# Whether achieve the gt through reconstruction or directly from training data
gt_reconstruct = True
data_dir = args.dir
dirA   = os.path.join(data_dir, "val", "input")

all_entries = os.listdir(dirA)
subdirs_gt = [
    os.path.join(dirA, name)
    for name in sorted(all_entries)
    if os.path.isdir(os.path.join(dirA, name))
]

pred_dir = data_dir + "_predict"
epoch_dirs_pred = [
    os.path.join(pred_dir, name) for name in os.listdir(pred_dir)
    if os.path.isdir(os.path.join(pred_dir, name))
]

if gt_reconstruct:
    gt_dir = data_dir + "_gt"
    epoch_dirs_gt = [
        os.path.join(gt_dir, name) for name in os.listdir(gt_dir)
        if os.path.isdir(os.path.join(gt_dir, name))
    ]

for num_epoch in range(len(epoch_dirs_pred)):
    epoch_dir = epoch_dirs_pred[num_epoch]
    subdirs_pred = [
        os.path.join(epoch_dir, name, "input", "V000000") for name in os.listdir(epoch_dir)
        if os.path.isdir(os.path.join(epoch_dir, name))
    ]
    if gt_reconstruct:
        epoch_dir_gt = epoch_dirs_gt[num_epoch]
        subdirs_gt = [
            os.path.join(epoch_dir_gt, name, "input", "V000000") for name in os.listdir(epoch_dir_gt)
            if os.path.isdir(os.path.join(epoch_dir_gt, name))
        ]

    num_batch = len(subdirs_pred)
    subdirs_gt = subdirs_gt[:num_batch]

    for batch_num in range(num_batch):
        dirA = subdirs_gt[batch_num]
        dirB = subdirs_pred[batch_num]
        outdir = os.path.join(dirB, "..", "..", "video", "compare")
        os.makedirs(outdir, exist_ok=True)

        # ---------- 收集文件 ----------
        files = sorted([
            f for f in os.listdir(dirA)
            if os.path.isfile(os.path.join(dirA, f)) and f.lower().endswith(f".{args.ext}")
        ])

        if not files:
            raise RuntimeError(f"{dirA} 里找不到 *.{args.ext} 文件")

        print(f"共 {len(files)} 张图片，将输出到 {outdir}")
        print(files)

        # ---------- Blend 并保存 ----------
        alpha = args.alpha
        for idx, fA in enumerate(files):
            fB = os.path.join(dirB, fA)
            if not os.path.exists(fB):
                print(f"[警告] {fB} 不存在，跳过")
                continue
            fA = os.path.join(dirA, fA)
            
            imgA = cv2.imread(str(fA), cv2.IMREAD_UNCHANGED)
            imgB = cv2.imread(str(fB), cv2.IMREAD_UNCHANGED)

            if imgA.shape != imgB.shape:
                print(f"[警告] 尺寸不一致 ，跳过")
                continue

            blend = cv2.addWeighted(imgA, alpha, imgB, 1-alpha, 0)
            cv2.imwrite(str(os.path.join(outdir, f"frame{idx:06d}.png")), blend)

            if idx == 0:
                h, w = blend.shape[:2]
                print(f"分辨率 {w}×{h}")

        # ---------- 调 ffmpeg 合成视频 ----------
        mp4_path = os.path.join(outdir, "blend.mp4")
        ffmpeg_cmd = [
            "ffmpeg", "-loglevel", "error", "-y",
            "-framerate", str(args.fps),
            "-i", str(os.path.join(outdir, f"frame%06d.{args.ext}")),   # 如果文件名不是 f000001.png 可自行改
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(mp4_path)
        ]
        print("正在调用 ffmpeg 生成视频 ...")
        subprocess.run(ffmpeg_cmd, check=True)
        print("✅ 完成！视频保存到", mp4_path)
