# === 4. plot_loss_curve.py ===
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

# === CONFIGURATION ===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="mpm_output", help="dataset name")
args = parser.parse_args()
output = "output_b1_tiny"

# Load the data from the JSON file
with open(f"{args.dataset}/{output}/step_loss_log.json") as f:
    data = json.load(f)

# Aggregate loss by epoch by averaging the loss values per epoch
epoch_train_losses = {}
epoch_val_losses = {}
epoch_val_losses_red = {}
epoch_val_losses_green = {}
epoch_val_losses_blue = {}

# Group by epoch and calculate average loss for each
for d in data:
    epoch = d["epoch"]
    loss = d["loss"]
    loss_r = d["loss_r"]
    loss_b = d["loss_b"]
    loss_g = d["loss_g"]
    if d["type"] == "train":
        if epoch not in epoch_train_losses:
            epoch_train_losses[epoch] = []
        epoch_train_losses[epoch].append(loss)
    elif d["type"] == "val":
        if epoch not in epoch_val_losses:
            epoch_val_losses[epoch] = []
        epoch_val_losses[epoch].append(loss)
        if epoch not in epoch_val_losses_red:
            epoch_val_losses_red[epoch] = []
        epoch_val_losses_red[epoch].append(loss_r)
        if epoch not in epoch_val_losses_green: 
            epoch_val_losses_green[epoch] = []
        epoch_val_losses_green[epoch].append(loss_g)
        if epoch not in epoch_val_losses_blue:
            epoch_val_losses_blue[epoch] = []
        epoch_val_losses_blue[epoch].append(loss_b)
    

# Compute average loss for each epoch
avg_train_losses = [np.mean(epoch_train_losses[epoch]) for epoch in sorted(epoch_train_losses.keys())]
avg_val_losses = [np.mean(epoch_val_losses.get(epoch, [])) for epoch in sorted(epoch_train_losses.keys())]
avg_val_losses_red = [np.mean(epoch_val_losses_red.get(epoch, [])) for epoch in sorted(epoch_train_losses.keys())]
avg_val_losses_green = [np.mean(epoch_val_losses_green.get(epoch, [])) for epoch in sorted(epoch_train_losses.keys())]
avg_val_losses_blue = [np.mean(epoch_val_losses_blue.get(epoch, [])) for epoch in sorted(epoch_train_losses.keys())]

# Plotting the loss curves by epoch
epochs = sorted(epoch_train_losses.keys())

plt.figure(figsize=(10, 6))

# Plot average train loss
plt.plot(epochs, avg_train_losses, label="Train Loss", marker='o', linestyle='-', color='orange', alpha=1.0)

# Plot average validation loss only if there is validation data
if avg_val_losses:
    plt.plot(epochs, avg_val_losses, label="Validation Loss", marker='o', linestyle='-', color='purple', alpha=1.0)
    # plt.plot(epochs, avg_val_losses_red, label="Validation Loss Red - Mass", linestyle='--', color='red', alpha=1.0)
    # plt.plot(epochs, avg_val_losses_green, label="Validation Loss Green - Stiffness", linestyle='--', color='green', alpha=1.0)
    # plt.plot(epochs, avg_val_losses_blue, label="Validation Loss Blue - Damping", linestyle='--', color='blue', alpha=1.0)

# Add labels and title
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Learning Curve (Average Loss per Epoch)", fontsize=14)

# Add grid and legend
plt.grid(True)
plt.legend()

# Tight layout for proper spacing
plt.tight_layout()

# Save the plot
plt.savefig(f"{args.dataset}/{output}_loss_curve.png")

# Display the plot
plt.show()

print("✅ plot complete")


# log plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, np.log10(avg_train_losses), label="Train Loss", marker='o', linestyle='-', color='orange', alpha=1.0)
plt.plot(epochs, np.log10(avg_val_losses), label="Validation Loss", marker='o', linestyle='-', color='purple', alpha=1.0)
# plt.plot(epochs, np.log10(avg_val_losses_red), label="Validation Loss Red - Mass", linestyle='--', color='red', alpha=1.0)
# plt.plot(epochs, np.log10(avg_val_losses_green), label="Validation Loss Green - Stiffness", linestyle='--', color='green', alpha=1.0)
# plt.plot(epochs, np.log10(avg_val_losses_blue), label="Validation Loss Blue - Damping", linestyle='--', color='blue', alpha=1.0)


plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Log Loss", fontsize=12)
plt.title("Log Learning Curve (Average Loss per Epoch)", fontsize=14)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(f"{args.dataset}/{output}_loss_curve_log.png")

plt.show()
print("✅ log plot complete")