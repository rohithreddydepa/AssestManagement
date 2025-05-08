import os
import pandas as pd
import shutil

# Load the CSV
df = pd.read_csv("Train.csv")

# Output directories for YOLO format
base_dir = "GTSDB_YOLO"
image_dir = os.path.join(base_dir, "images", "train")
label_dir = os.path.join(base_dir, "labels", "train")
os.makedirs(image_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

for _, row in df.iterrows():
    w, h = row["Width"], row["Height"]
    x1, y1, x2, y2 = row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"]
    class_id = int(row["ClassId"])
    rel_img_path = row["Path"].replace("\\", "/")
    img_name = os.path.basename(rel_img_path)

    # ✅ Fixed path: don't prepend "archive/"
    src_img_path = os.path.join(rel_img_path)
    dst_img_path = os.path.join(image_dir, img_name)

    # Copy image to YOLO directory
    try:
        shutil.copyfile(src_img_path, dst_img_path)
    except FileNotFoundError:
        print(f"❌ Image not found: {src_img_path}")
        continue

    # Normalize bounding box for YOLO
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    box_width = (x2 - x1) / w
    box_height = (y2 - y1) / h

    # Write label file in YOLO format
    label_file = os.path.join(label_dir, img_name.replace(".png", ".txt"))
    with open(label_file, "a") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

print("\n✅ All annotations converted to YOLO format!")
