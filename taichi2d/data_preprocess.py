import os
import sys
import re
import shutil
import argparse

# === CONFIGURATION ===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="dataset_uni", help="dataset name")
args = parser.parse_args()

def find_empty_folders(root_dir):
    """
    Find all empty folders in the given directory and its subdirectories.
    """
    empty_folders = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if the current directory is empty (no files and no subdirectories)
        if not dirnames and not filenames:
            empty_folders.append(dirpath)
    
    return empty_folders

def extract_indices(folder_paths):
    """
    Extract the indices from folder paths.
    """
    indices = {"input": [], "label": [], "raw": []}
    
    for path in folder_paths:
        # Extract the directory name (input, label, or raw)
        dir_name = os.path.basename(os.path.dirname(path))
        
        # Extract the index using regex
        match = re.search(r'v(\d+)', os.path.basename(path))
        if match and dir_name in indices:
            index = int(match.group(1))
            indices[dir_name].append(index)
    
    # Sort indices for each directory
    for dir_name in indices:
        indices[dir_name].sort()
    
    return indices

def save_indices_to_file(indices, dataset_dir):
    """
    Save the indices to a file in the dataset directory.
    """
    # Create a file to save the indices
    file_path = os.path.join(dataset_dir, "empty_indices.txt")
    
    with open(file_path, 'w') as f:
        f.write("Empty folder indices in the dataset:\n\n")
        
        for dir_name, idx_list in indices.items():
            f.write(f"{dir_name}/: {len(idx_list)} empty folders\n")
            f.write(f"Indices: {idx_list}\n\n")
    
    print(f"Saved empty indices to {file_path}")

def delete_empty_folders(empty_folders):
    """
    Delete all empty folders.
    """
    count = 0
    for folder in empty_folders:
        try:
            os.rmdir(folder)
            count += 1
        except Exception as e:
            print(f"Error deleting {folder}: {e}")
    
    return count

# Path to the dataset directory
dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dataset)

# Check if the directory exists
if not os.path.exists(dataset_dir):
    print(f"Error: Directory '{dataset_dir}' does not exist.")

# Find empty folders
empty_folders = find_empty_folders(dataset_dir)

# find total number of folders in the dataset
total_folders = len(os.listdir(dataset_dir + "/input"))
print(f"Total number of folders in {dataset_dir}: {total_folders}")

# Extract indices
indices = extract_indices(empty_folders)

# Print results
print(f"Found {len(empty_folders)} empty folder(s) in {dataset_dir}:")

# Print indices only once
all_indices = set()
for idx_list in indices.values():
    all_indices.update(idx_list)

print(f"Total unique indices: {len(all_indices)}")
print(f"Indices: {sorted(list(all_indices))}")

# Save indices to file
save_indices_to_file(indices, dataset_dir)

# Ask for confirmation before deleting
print("\nAbout to delete all empty folders. This action cannot be undone.")
confirmation = input("Do you want to proceed? (yes/no): ")

if confirmation.lower() == 'yes':
    # Delete empty folders
    deleted_count = delete_empty_folders(empty_folders)
    print(f"Deleted {deleted_count} empty folders.")
else:
    print("Deletion cancelled.")