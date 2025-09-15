# import os
# ROOT = os.path.dirname(os.path.abspath(__file__))
# pic_dir_1 = os.path.join(ROOT, "check_1","input", "V000000")
# pic_dir_2 = os.path.join(ROOT, "check_2","input", "V000000")

# pic_diff_out = os.path.join(ROOT, "pic_diff")

# if not os.path.exists(pic_diff_out):
#     os.makedirs(pic_diff_out)
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# def compare_images(pic1, pic2):
#     img1 = Image.open(pic1)
#     img2 = Image.open(pic2)

#     if img1.size != img2.size:
#         raise ValueError("Images must be the same size for comparison.")

#     arr1 = np.array(img1)
#     arr2 = np.array(img2)

#     diff = np.abs(arr1 - arr2)
#     diff_img = Image.fromarray(diff.astype(np.uint8))

#     return diff_img

# def save_comparison(pic1, pic2, output_path):
#     diff_img = compare_images(pic1, pic2)
#     diff_img.save(output_path)

# def main():
#     pic1_files = sorted(os.listdir(pic_dir_1))
#     pic2_files = sorted(os.listdir(pic_dir_2))

#     if len(pic1_files) != len(pic2_files):
#         raise ValueError("The number of images in both directories must be the same.")

#     for file1, file2 in zip(pic1_files, pic2_files):
#         pic1_path = os.path.join(pic_dir_1, file1)
#         pic2_path = os.path.join(pic_dir_2, file2)
#         output_path = os.path.join(pic_diff_out, f"diff_{file1}")

#         save_comparison(pic1_path, pic2_path, output_path)
#     print(f"Comparison images saved to {pic_diff_out}")
# if __name__ == "__main__":
#     main()
#     # Optionally, display the first comparison image
#     diff_files = sorted(os.listdir(pic_diff_out))
#     if diff_files:
#         first_diff_path = os.path.join(pic_diff_out, diff_files[0])
#         diff_img = Image.open(first_diff_path)
#         plt.imshow(diff_img)
#         plt.axis('off')
#         plt.show()
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import imageio

ROOT = os.path.dirname(os.path.abspath(__file__))      
pic_dir_1 = os.path.join(ROOT, "damping50_K_1", "input", "V000000")
pic_dir_2 = os.path.join(ROOT, "damping50_K_100", "input", "V000000")
pic_diff_out = os.path.join(ROOT, "damping50_K_1_100")
video_path = os.path.join(pic_diff_out, "diff_video.mp4")

if not os.path.exists(pic_diff_out):
    os.makedirs(pic_diff_out)

def compare_images(pic1, pic2):
    img1 = Image.open(pic1).convert("RGB")
    img2 = Image.open(pic2).convert("RGB")

    if img1.size != img2.size:
        raise ValueError("Images must be the same size for comparison.")

    arr1 = np.array(img1)
    arr2 = np.array(img2)

    diff = np.abs(arr1 - arr2)
    diff_img = Image.fromarray(diff.astype(np.uint8))

    return diff_img

def save_comparison(pic1, pic2, output_path):
    diff_img = compare_images(pic1, pic2)
    diff_img.save(output_path)

def create_video_from_images(image_dir, output_video_path, fps=10):
    files = sorted(f for f in os.listdir(image_dir) if f.endswith(".png"))
    images = [imageio.imread(os.path.join(image_dir, f)) for f in files]
    imageio.mimsave(output_video_path, images, fps=fps)
    print(f"Video saved to {output_video_path}")

def main():
    pic1_files = sorted(os.listdir(pic_dir_1))
    pic2_files = sorted(os.listdir(pic_dir_2))

    if len(pic1_files) != len(pic2_files):
        raise ValueError("The number of images in both directories must be the same.")

    for file1, file2 in zip(pic1_files, pic2_files):
        pic1_path = os.path.join(pic_dir_1, file1)
        pic2_path = os.path.join(pic_dir_2, file2)
        output_path = os.path.join(pic_diff_out, f"diff_{file1}")
        save_comparison(pic1_path, pic2_path, output_path)

    print(f"Comparison images saved to {pic_diff_out}")
    create_video_from_images(pic_diff_out, video_path, fps=10)

if __name__ == "__main__":
    main()

        