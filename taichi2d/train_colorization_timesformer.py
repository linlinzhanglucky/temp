# === train_colorization_timesformer.py ===

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import argparse
import numpy as np
np.random.seed(2025)

sys.path.append(r"../../TimeSformer") # same level as robovoxel
from dataset_timesformer import BWVideoColorizationDataset
from timesformer.models.vit import TimeSformer

# === CONFIG ===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="dataset_uni", help="dataset name")
args = parser.parse_args()
train_input = rf"{args.dataset}/train/input"
train_label = rf"{args.dataset}/train/label"
val_input = rf"{args.dataset}/val/input"
val_label = rf"{args.dataset}/val/label"
output_dir = rf"{args.dataset}/output_b1_tiny"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

def main():
    # === HYPERPARAMETERS ===
    start_epoch = 1  # ✅ resume from epoch
    num_epochs = 200
    batch_size = 8
    lr = 1e-4
    num_frames = 20
    patience = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === DATASET & LOADER ===
    number_of_workers = 16 # lab computer
    train_dataset = BWVideoColorizationDataset(train_input, train_label, num_frames)
    val_dataset = BWVideoColorizationDataset(val_input, val_label, num_frames)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=number_of_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=number_of_workers)
    # val_image_ids = np.random.randint(0, len(val_dataset), 10)

    # === MODEL ===
    # Tiny: 192, 6, 3
    # Base: 768, 12, 12
    model = TimeSformer(
        img_size=224,
        num_classes=3 * 224 * 224, # ✅ 3 channels, output shape
        in_chans=1,  # ✅ binary, add to TimeSformer
        num_frames=num_frames,
        attention_type='divided_space_time',
        embed_dim=192,
        depth=6,
        num_heads=3,
        pretrained=False,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # === Resume if model exists ===
    if start_epoch > 1:
        ckpt_path = os.path.join(output_dir, f"model_epoch{start_epoch - 1}.pth")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path))
            print(f"🔁 Resumed from {ckpt_path}")
        else:
            print(f"⚠️ Warning: Resume failed. Checkpoint {ckpt_path} not found.")

    # === Logging ===
    log_file = open(os.path.join(output_dir, "train_log.txt"), "a")
    step_loss_path = os.path.join(output_dir, "step_loss_log.json")
    step_log = []
    best_val_loss = float("inf")
    no_improve = 0
    global_step = 0

    # === TRAIN LOOP ===
    print("✅ Starting training on", device)
    print(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")


    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        train_loss = 0
        progress = tqdm(train_loader, desc=f"[Epoch {epoch}/{num_epochs}]")

        for batch in progress:
            x = batch["gray_video"].to(device).permute(0, 2, 1, 3, 4)
            y = batch["target_frame"].to(device)
            out = model(x).view(x.size(0), 3, 224, 224)

            loss = criterion(out, y)
            loss_r = criterion(out[:, 0, :, :], y[:, 0, :, :])
            loss_g = criterion(out[:, 1, :, :], y[:, 1, :, :])
            loss_b = criterion(out[:, 2, :, :], y[:, 2, :, :])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.6f}"})
            step_log.append({"epoch": epoch, 
                            "step": global_step, 
                            "loss": loss.item(), 
                            "loss_r": loss_r.item(),
                            "loss_g": loss_g.item(),
                            "loss_b": loss_b.item(),
                            "type": "train"})
            global_step += 1

            # if global_step % 200 == 0:
            #     save_image(out[0], f"{output_dir}/pred_epoch{epoch}_step{global_step}.png")
            #     save_image(y[0], f"{output_dir}/gt_epoch{epoch}_step{global_step}.png")

        # === Validation ===
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bid, batch in enumerate(val_loader):
                x = batch["gray_video"].to(device).permute(0, 2, 1, 3, 4)
                y = batch["target_frame"].to(device)
                out = model(x).view(x.size(0), 3, 224, 224)
                loss = criterion(out, y)
                loss_r = criterion(out[:, 0, :, :], y[:, 0, :, :])
                loss_g = criterion(out[:, 1, :, :], y[:, 1, :, :])
                loss_b = criterion(out[:, 2, :, :], y[:, 2, :, :])
                val_loss += loss.item()
                step_log.append({"epoch": epoch, 
                                "step": global_step, 
                                "loss": loss.item(), 
                                "loss_r": loss_r.item(),
                                "loss_g": loss_g.item(),
                                "loss_b": loss_b.item(),
                                "type": "val"})

                # save first 10 images
                if bid < 10: #bid in val_image_ids:
                    save_image(out[0], f"{output_dir}/images/pred_epoch{epoch}_batch{bid}.png")
                    if epoch == 1:
                        save_image(y[0], f"{output_dir}/images/gt_epoch{epoch}_batch{bid}.png")

        train_avg = train_loss / len(train_loader)
        val_avg = val_loss / len(val_loader)
        log_file.write(f"[Epoch {epoch}] Train Loss: {train_avg:.6f} | Val Loss: {val_avg:.6f}\n")
        log_file.flush()
        print(f"📉 Epoch {epoch}: Train={train_avg:.6f} | Val={val_avg:.6f}")

        # === Save model ===
        model_path = os.path.join(output_dir, "models", f"model_epoch{epoch}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"💾 Saved model: {model_path}")

        with open(step_loss_path, "w") as f:
            json.dump(step_log, f, indent=2)

        # # === Early stopping ===
        # if val_avg < best_val_loss - 1e-4:
        #     best_val_loss = val_avg
        #     no_improve = 0
        # else:
        #     no_improve += 1
        #     print(f"⏸️ No improvement {no_improve}/{patience}")
        #     if no_improve >= patience:
        #         print("⛔ Early stopping triggered.")
        #         break

        # === EARLY STOPPING MONITORING (non-breaking) ===
        if val_avg < best_val_loss - 1e-6:
            best_val_loss = val_avg
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "models", "best_model.pth"))
            print(f"🏅 Saved best model at epoch {epoch}")
        else:
            no_improve += 1
            print(f"⏸️ No improvement {no_improve}/{patience}")
            if no_improve >= patience:
                print("⛔ Early stopping would have triggered, but continuing training to full epoch count.")
                
    log_file.close()
    print("✅ Training complete")

if __name__ == "__main__":
    main()
    print("✅ Main function complete")