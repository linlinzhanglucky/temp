import cv2
import os
import numpy as np

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

def generate_video(raw_folder, input_folder, label_folder, output_path, num_sequences=8, num_frames=40, fps=10):
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
