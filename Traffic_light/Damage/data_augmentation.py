import os
import cv2

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def augment_dataset(input_dir, output_dir, angles):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path)
            for angle in angles:
                rotated = rotate_image(image, angle)
                output_filename = f"{os.path.splitext(filename)[0]}_rot{angle}.jpg"
                angle_folder = os.path.join(output_dir, str(angle))
                os.makedirs(angle_folder, exist_ok=True)
                cv2.imwrite(os.path.join(angle_folder, output_filename), rotated)

# Use smaller angle steps for higher accuracy
augment_dataset("original_images", "augmented_images", range(-60, 65, 5))
