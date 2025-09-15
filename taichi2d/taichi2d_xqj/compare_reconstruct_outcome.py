#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
blend_two_folders.py
把 folder_A 与 folder_B 中命名相同的 PNG/JPG 做 50/50 混合，
结果写到 out_dir, 并顺便用 ffmpeg 拼成 mp4。
"""

import os, cv2, argparse, subprocess, shutil
from pathlib import Path

# ---------- 使用参数 ----------
parser = argparse.ArgumentParser()
parser.add_argument("--dirA", default= "./mpm_output_0601_gt/raw/V000000",  help="第一组图片文件夹")
parser.add_argument("--dirB", default = "./mpm_output_0601_predict_initial/raw/V000000", help="第二组图片文件夹")
parser.add_argument("--out",  default = "./mpm_output_0601_compare_initial", help="Blend 后图片 & 视频输出文件夹")
parser.add_argument("--ext",  default="png", help="图片扩展名 (png / jpg …)")
parser.add_argument("--alpha",type=float, default=0.5, help="A 占的权重 (0~1)")
parser.add_argument("--fps",  type=int,   default=30,  help="合成视频帧率")
args = parser.parse_args()

dirA   = Path(args.dirA)
dirB   = Path(args.dirB)
outdir = Path(args.out)
outdir.mkdir(parents=True, exist_ok=True)

# ---------- 收集文件 ----------
files = sorted([f for f in dirA.iterdir() if f.suffix.lower().endswith(args.ext)])
if not files:
    raise RuntimeError(f"{dirA} 里找不到 *.{args.ext} 文件")

print(f"共 {len(files)} 张图片，将输出到 {outdir}")

# ---------- Blend 并保存 ----------
alpha = args.alpha
for idx, fA in enumerate(files):
    fB = dirB / fA.name
    if not fB.exists():
        print(f"[警告] {fB} 不存在，跳过")
        continue

    imgA = cv2.imread(str(fA), cv2.IMREAD_UNCHANGED)
    imgB = cv2.imread(str(fB), cv2.IMREAD_UNCHANGED)

    if imgA.shape != imgB.shape:
        print(f"[警告] 尺寸不一致 {fA.name}，跳过")
        continue

    blend = cv2.addWeighted(imgA, alpha, imgB, 1-alpha, 0)
    cv2.imwrite(str(outdir / fA.name), blend)

    if idx == 0:
        h, w = blend.shape[:2]
        print(f"分辨率 {w}×{h}")

# ---------- 调 ffmpeg 合成视频 ----------
mp4_path = outdir / "blend.mp4"
ffmpeg_cmd = [
    "ffmpeg", "-loglevel", "error", "-y",
    "-framerate", str(args.fps),
    "-i", str(outdir / f"f%06d.{args.ext}"),   # 如果文件名不是 f000001.png 可自行改
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    str(mp4_path)
]
print("正在调用 ffmpeg 生成视频 ...")
subprocess.run(ffmpeg_cmd, check=True)
print("✅ 完成！视频保存到", mp4_path)
