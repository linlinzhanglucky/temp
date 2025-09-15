import os
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import numpy as np
import glob
# === CONFIGURATION ===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="dataset_uni", help="dataset name")
parser.add_argument("--epoch", type=int, default=1, help="epoch number")
args = parser.parse_args()
output = "output_b1_tiny"

# === PATHS ===
#dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dataset)
dataset_dir = args.dataset
output_dir = os.path.join(dataset_dir, output, "images")
vis_dir = os.path.join(dataset_dir, "vis_b1")
os.makedirs(vis_dir, exist_ok=True)
save_path = os.path.join(vis_dir, f"{args.epoch}.png")
cols = 3

def get_pairs(num_pairs: int = 9, dataset_size: int = 4000, epoch_id: int = 1):
    # get batch indices
    epoch1_files = glob.glob(os.path.join(output_dir, "gt_epoch1_*.png"))
    bids = [os.path.basename(f).split("_")[2].split(".")[0] for f in epoch1_files]
    print(bids)
    
    pairs = []
    for bid in bids:
        pairs.append(("gt_epoch1_" + str(bid) + ".png", 
                      "pred_epoch" + str(epoch_id) + "_" + str(bid) + ".png"))
    return pairs[:num_pairs], bids[:num_pairs]

selected_pairs, bids = get_pairs(epoch_id=args.epoch, num_pairs=9)

rows = (len(selected_pairs) + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(12, 2.5 * rows))
axs = axs.flatten()

for i, (gt_file, pred_file) in enumerate(selected_pairs):
    gt_path = os.path.join(output_dir, gt_file)
    pred_path = os.path.join(output_dir, pred_file)

    if not os.path.exists(gt_path) or not os.path.exists(pred_path):
        print(f"⚠️ Missing pair: {gt_file}, {pred_file}")
        axs[i].axis("off")
        continue

    gt_img = Image.open(gt_path)
    pred_img = Image.open(pred_path)

    # Combine GT + Prediction
    combined = Image.new("RGB", (gt_img.width + pred_img.width, gt_img.height))
    combined.paste(gt_img, (0, 0))
    combined.paste(pred_img, (gt_img.width, 0))

    axs[i].imshow(combined)
    axs[i].axis("off")
    # axs[i].set_title(f"{gt_file} vs {pred_file}", fontsize=10)
    axs[i].set_title(f"Test {bids[i][-1:]}", fontsize=20)

# Hide extra plots
for i in range(len(selected_pairs), len(axs)):
    axs[i].axis("off")

plt.suptitle(f"GT VS Pred (Epoch {args.epoch})", fontsize=24)
plt.tight_layout(pad=2.5)
plt.savefig(save_path, dpi=300)
# plt.show()
print(f"✅ Saved comparison grid to: {save_path}")


"""plot three channels of the image"""
cols = 3
rows = 1 # og, red, green, blue
fig, axs = plt.subplots(rows, cols, figsize=(12, 2.5 * rows * 4))
axs = axs.flatten()

pairs = selected_pairs[:cols    ]

for i, (gt_file, pred_file) in enumerate(pairs):
    gt_path = os.path.join(output_dir, gt_file)
    pred_path = os.path.join(output_dir, pred_file)

    gt_img = Image.open(gt_path)
    pred_img = Image.open(pred_path)

    # to np array
    gt_img_np = np.array(gt_img)
    pred_img_np = np.array(pred_img)
    base_value = 0.0 * 255
    # base_value = 0

    gt_img_r = np.stack([gt_img_np[:, :, 0], 
                         np.ones_like(gt_img_np[:, :, 0]) * base_value, 
                         np.ones_like(gt_img_np[:, :, 0]) * base_value], axis=-1)
    gt_img_g = np.stack([np.ones_like(gt_img_np[:, :, 0]) * base_value, 
                         gt_img_np[:, :, 1], 
                         np.ones_like(gt_img_np[:, :, 0]) * base_value], axis=-1)
    gt_img_b = np.stack([np.ones_like(gt_img_np[:, :, 0]) * base_value, 
                         np.ones_like(gt_img_np[:, :, 0]) * base_value, 
                         gt_img_np[:, :, 2]], axis=-1)

    pred_img_r = np.stack([pred_img_np[:, :, 0], 
                         np.ones_like(pred_img_np[:, :, 0]) * base_value, 
                         np.ones_like(pred_img_np[:, :, 0]) * base_value], axis=-1)
    pred_img_g = np.stack([np.ones_like(pred_img_np[:, :, 0]) * base_value, 
                         pred_img_np[:, :, 1], 
                         np.ones_like(pred_img_np[:, :, 0]) * base_value], axis=-1)
    pred_img_b = np.stack([np.ones_like(pred_img_np[:, :, 0]) * base_value, 
                         np.ones_like(pred_img_np[:, :, 0]) * base_value, 
                         pred_img_np[:, :, 2]], axis=-1)

    print(gt_img_r.dtype, gt_img_r.min(), gt_img_r.max())
    # to PIL image
    gt_img_r = Image.fromarray(gt_img_r.astype(np.uint8))
    gt_img_g = Image.fromarray(gt_img_g.astype(np.uint8))
    gt_img_b = Image.fromarray(gt_img_b.astype(np.uint8))

    pred_img_r = Image.fromarray(pred_img_r.astype(np.uint8))
    pred_img_g = Image.fromarray(pred_img_g.astype(np.uint8))
    pred_img_b = Image.fromarray(pred_img_b.astype(np.uint8))


    combined = Image.new("RGB", (gt_img.width * 2, gt_img.height * 4))
    combined.paste(gt_img, (0, 0))
    combined.paste(pred_img, (gt_img.width, 0))
    combined.paste(gt_img_r, (0, gt_img.height))
    combined.paste(pred_img_r, (gt_img.width, gt_img.height))
    combined.paste(gt_img_g, (0, gt_img.height * 2))
    combined.paste(pred_img_g, (gt_img.width, gt_img.height * 2))
    combined.paste(gt_img_b, (0, gt_img.height * 3))
    combined.paste(pred_img_b, (gt_img.width, gt_img.height * 3))

    axs[i].imshow(combined)
    axs[i].axis("off")
    axs[i].set_title(f"Test {bids[i][-1:]}, og, r, g, b", fontsize=20)

save_path_channels = os.path.join(vis_dir, f"{args.epoch}_channels_base.png")
plt.suptitle(f"GT VS Pred (Epoch {args.epoch})", fontsize=24)
plt.tight_layout(pad=2.5)
plt.savefig(save_path_channels, dpi=300)
# plt.show()
print(f"✅ Saved comparison grid to: {save_path_channels}")
    
