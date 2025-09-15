#train_colorization_timesformer_tiny.py
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.append(r"C:/Users/linli/OneDrive/Desktop/linlinzhanglab/TimeSformer")
from dataset_timesformer import BWVideoColorizationDataset
from timesformer.models.vit import TimeSformer

# === CONFIG ===
train_input = "dataset_region/train/input"
train_label = "dataset_region/train/label"
val_input = "dataset_region/val/input"
val_label = "dataset_region/val/label"
output_dir = "output_tiny"
os.makedirs(output_dir, exist_ok=True)

# === HYPERPARAMETERS ===
start_epoch = 1
num_epochs = 30
batch_size = 1
lr = 2e-4
num_frames = 20
patience = 3
max_train_samples = 1000  # ✅ Limit training data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === DATASET ===
train_dataset = BWVideoColorizationDataset(train_input, train_label, num_frames, max_samples=max_train_samples)
val_dataset = BWVideoColorizationDataset(val_input, val_label, num_frames)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# === MODEL (ViT-Tiny Config) ===
# model = TimeSformer(
#     img_size=224,
#     num_classes=3 * 224 * 224,
#     num_frames=num_frames,
#     attention_type='divided_space_time',
#     embed_dim=192,     # 👈 Smaller than base (768)
#     depth=12,
#     num_heads=3
# ).to(device)
model = TimeSformer(
    img_size=224,
    # num_classes=3 * 224 * 224, 
    num_classes=1 * 224 * 224, #change to binary now
    num_frames=num_frames,
    attention_type='divided_space_time',
    embed_dim=192,
    depth=12,
    num_heads=3,
    # pretrained_model='',  # ✅ 禁止加载 ViT-Base 预训练权重
    pretrained=False  # ✅ 禁用预训练加载
).to(device)

#NOTE DEGUG（not work）:
# 在 TimeSformer.__init__() 里加一行：
# self.pretrained = False  # ✅ 禁止调用 load_pretrained
#NOTE DEBUG (work):
# actually 我在vit.py的TimeSformer class 里加了一个参数pretrained=True, 所以需要改成pretrained=False



optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# === Resume if model exists ===
ckpt_path = os.path.join(output_dir, f"model_epoch{start_epoch - 1}.pth")
if start_epoch > 1 and os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path))
    print(f"🔁 Resumed from {ckpt_path}")

# === Logging ===
log_file = open(os.path.join(output_dir, "train_log2.txt"), "a")
step_loss_path = os.path.join(output_dir, "step_loss_log2.json")
step_log = []
best_val_loss = float("inf")
no_improve = 0
global_step = 0

print("✅ Starting training on", device)
print(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

# === TRAINING LOOP ===
for epoch in range(start_epoch, num_epochs + 1):
    model.train()
    train_loss = 0
    progress = tqdm(train_loader, desc=f"[Epoch {epoch}/{num_epochs}]")

    for batch in progress:
        x = batch["gray_video"].to(device).permute(0, 2, 1, 3, 4)
        y = batch["target_frame"].to(device)
        # out = model(x).view(x.size(0), 3, 224, 224) 
        out = model(x).view(x.size(0), 1, 224, 224) #change to binary now   

        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        step_log.append({"epoch": epoch, "step": global_step, "loss": loss.item(), "type": "train"})
        progress.set_postfix({"loss": f"{loss.item():.4f}"})
        global_step += 1

        if global_step % 200 == 0:
            save_image(out[0], f"{output_dir}/sample_pred_step{global_step}.png")
            save_image(y[0], f"{output_dir}/sample_gt_step{global_step}.png")

    # === Validation ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch["gray_video"].to(device).permute(0, 2, 1, 3, 4)
            y = batch["target_frame"].to(device)
            out = model(x).view(x.size(0), 3, 224, 224)
            loss = criterion(out, y)
            val_loss += loss.item()
            step_log.append({"epoch": epoch, "step": global_step, "loss": loss.item(), "type": "val"})

    train_avg = train_loss / len(train_loader)
    val_avg = val_loss / len(val_loader)
    log_file.write(f"[Epoch {epoch}] Train Loss: {train_avg:.4f} | Val Loss: {val_avg:.4f}\n")
    log_file.flush()
    print(f"📉 Epoch {epoch}: Train={train_avg:.4f} | Val={val_avg:.4f}")

    # Save model each epoch
    torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch{epoch}.pth"))
    with open(step_loss_path, "w") as f:
        json.dump(step_log, f, indent=2)

    # Best model save
    if val_avg < best_val_loss - 1e-4:
        best_val_loss = val_avg
        no_improve = 0
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
        print(f"🏅 Saved best model at epoch {epoch}")
    else:
        no_improve += 1
        print(f"⏸️ No improvement {no_improve}/{patience}")
        if no_improve >= patience:
            print("⚠️ Early stopping would trigger, continuing anyway.")

log_file.close()
print("✅ Training complete.")
