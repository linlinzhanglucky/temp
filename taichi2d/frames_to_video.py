import os
import cv2
from natsort import natsorted

frame_dir = "/home/jiong/Documents/code25/RoboVoxel/taichi2d/dataset_region/vis"
output_path = f"{frame_dir}/colorization_progress.mp4"

frames = natsorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])
frame_path = os.path.join(frame_dir, frames[0])
frame = cv2.imread(frame_path)
height, width, _ = frame.shape

video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))

for fname in frames:
    path = os.path.join(frame_dir, fname)
    frame = cv2.imread(path)
    video.write(frame)

video.release()
print(f"🎬 Video saved to {output_path}")


# to gif
import imageio

# Load the actual image data instead of using filenames
image_frames = []
for fname in frames:
    path = os.path.join(frame_dir, fname)
    img = cv2.imread(path)
    # Convert from BGR to RGB (OpenCV uses BGR, but imageio expects RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_frames.append(img)

imageio.mimsave(f'{frame_dir}/colorization_progress.gif', image_frames, duration=1000)
print(f"🎬 GIF saved to {frame_dir}/colorization_progress.gif")

