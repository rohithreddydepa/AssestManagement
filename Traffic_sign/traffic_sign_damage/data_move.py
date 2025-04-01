import os
import shutil
import random

# Set paths
dataset_path = os.getcwd()
train_img_dir = os.path.join(dataset_path, "train", "images")
train_lbl_dir = os.path.join(dataset_path, "train", "labels")
valid_img_dir = os.path.join(dataset_path, "valid", "images")
valid_lbl_dir = os.path.join(dataset_path, "valid", "labels")
test_img_dir = os.path.join(dataset_path, "test", "images")
test_lbl_dir = os.path.join(dataset_path, "test", "labels")

# Create destination directories if they don't exist
for folder in [valid_img_dir, valid_lbl_dir, test_img_dir, test_lbl_dir]:
    os.makedirs(folder, exist_ok=True)

# Get list of all training images
train_images = sorted([f for f in os.listdir(train_img_dir) if f.endswith(".jpg") or f.endswith(".png")])

# Shuffle dataset
random.seed(42)  # For reproducibility
random.shuffle(train_images)

# Split dataset
total_images = len(train_images)
valid_count = int(0.2 * total_images)  # 20% for validation
test_count = int(0.1 * total_images)   # 10% for testing

valid_images = train_images[:valid_count]
test_images = train_images[valid_count:valid_count + test_count]

# Function to move images & labels
def move_files(image_list, src_img_dir, src_lbl_dir, dest_img_dir, dest_lbl_dir):
    for img_file in image_list:
        label_file = img_file.replace(".jpg", ".txt").replace(".png", ".txt")
        
        # Move image
        shutil.move(os.path.join(src_img_dir, img_file), os.path.join(dest_img_dir, img_file))
        
        # Move label
        if os.path.exists(os.path.join(src_lbl_dir, label_file)):
            shutil.move(os.path.join(src_lbl_dir, label_file), os.path.join(dest_lbl_dir, label_file))

# Move files to validation and test sets
move_files(valid_images, train_img_dir, train_lbl_dir, valid_img_dir, valid_lbl_dir)
move_files(test_images, train_img_dir, train_lbl_dir, test_img_dir, test_lbl_dir)

print(f"✅ Moved {valid_count} images to validation set.")
print(f"✅ Moved {test_count} images to test set.")
print(f"✅ Remaining in train: {total_images - valid_count - test_count} images.")
