import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO

# ======================================================================
# Auto-Detect Device (GPU or CPU)
# ======================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔍 Using device: {DEVICE}")

# Adjust batch size based on available hardware
BATCH_SIZE = 16 if DEVICE == "cuda" else 4  # Lower batch size for CPU

# ======================================================================
# Auto-Detect Dataset Path
# ======================================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # Get script directory
DATASET_DIR = os.path.abspath(os.path.dirname(__file__))  # Auto-detect dataset folder
MODEL_NAME = "yolov8m"  # Define model type

# Ensure dataset exists
if not os.path.exists(os.path.join(DATASET_DIR, "train/images")):
    raise FileNotFoundError(f"❌ Dataset not found in: {DATASET_DIR}")

# Generate `data.yaml` automatically
DATA_YAML_PATH = os.path.join(DATASET_DIR, "data.yaml")
with open(DATA_YAML_PATH, "w") as f:
    f.write(f"""path: {DATASET_DIR}
train: train/images
val: valid/images
test: test/images
nc: 5
names: ['Bending', 'Damage', 'Healthy', 'Vandalism', 'Wear']
""")
print(f"✅ Dataset YAML created at {DATA_YAML_PATH}")

# ======================================================================
# Set Unique Training Directory
# ======================================================================
TRAIN_DIR = os.path.join(BASE_DIR, "runs", MODEL_NAME)
os.makedirs(TRAIN_DIR, exist_ok=True)  # Ensure directory exists

# ======================================================================
# Configuration
# ======================================================================
class Config:
    MODEL_NAME = "yolov8m"  # Using YOLOv8m for better efficiency on small datasets
    PRETRAINED = True  # Use pretrained weights
    
    # Dataset
    DATA_YAML = DATA_YAML_PATH  
    
    # Training
    IMG_SIZE = 736  
    EPOCHS = 150  
    BATCH = BATCH_SIZE  
    OPTIMIZER = "AdamW"  
    LR0 = 0.0005  
    LRF = 0.01  
    WEIGHT_DECAY = 0.0005
    DROPOUT = 0.1  

    # Training output isolation
    PROJECT = TRAIN_DIR  # Unique output directory for each model

    # Augmentation
    AUGMENT = True
    CUSTOM_AUG = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.RandomShadow(p=0.3),
        A.ChannelShuffle(p=0.1),
        A.GaussNoise(p=0.2),
        A.MotionBlur(p=0.2),  
        A.CLAHE(p=0.2),  
        A.RandomFog(alpha_coef=0.1, p=0.2),
        A.CoarseDropout(max_holes=8, min_holes=1, max_height=16, max_width=16, min_height=4, min_width=4, p=0.2),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    # Advanced Features
    MIXUP = 0.1  
    PATIENCE = 10  

# ======================================================================
# Custom Albumentations Wrapper
# ======================================================================
class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, labels):
        annotations = {'image': image, 'bboxes': labels[:, :4], 'class_labels': labels[:, 4]}
        augmented = self.transform(**annotations)
        return augmented['image'], torch.tensor(augmented['bboxes']), torch.tensor(augmented['class_labels'])

# ======================================================================
# Model Training
# ======================================================================
def train_model():
    # Load model
    model = YOLO(f"{Config.MODEL_NAME}.pt" if Config.PRETRAINED else f"{Config.MODEL_NAME}.yaml")
    
    # Add Albumentations
    albumentations_transform = AlbumentationsTransform(Config.CUSTOM_AUG)
    model.add_callback("on_preprocess", albumentations_transform)
    
    # Train with optimized settings
    results = model.train(
        data=Config.DATA_YAML,
        epochs=Config.EPOCHS,
        imgsz=Config.IMG_SIZE,
        batch=Config.BATCH,
        optimizer=Config.OPTIMIZER,
        lr0=Config.LR0,
        lrf=Config.LRF,
        weight_decay=Config.WEIGHT_DECAY,
        dropout=Config.DROPOUT,
        augment=Config.AUGMENT,
        mixup=Config.MIXUP,
        patience=Config.PATIENCE,
        device=DEVICE,  # Auto-detects CPU or GPU
        pretrained=Config.PRETRAINED,
        val=True,
        project=Config.PROJECT,  # 💡 Unique training output directory
        name="experiment",  # Creates runs/yolov8m/experiment/
        exist_ok=True,  # Avoid overwriting old runs
        mosaic=0.5,  
        fliplr=0.5,  
        flipud=0.1,  
        degrees=10.0,  
        translate=0.1,  
        scale=0.3,  
        shear=2.0,  
        perspective=0.001,  
        hsv_h=0.02,  
        hsv_s=0.5,  
        hsv_v=0.3,  
        single_cls=False,
        verbose=True
    )
    
    return model

# ======================================================================
# Main Execution
# ======================================================================
if __name__ == "__main__":
    # Enable CUDA debugging (optional)
    torch.set_float32_matmul_precision('high')  
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  
    
    # Train and save model
    trained_model = train_model()
    trained_model.export(format="onnx")  
    
    # Validate on Test Set
    metrics = trained_model.val(
        data=Config.DATA_YAML,
        split="test",
        conf=0.4,  
        iou=0.5
    )
    
    print(f"✅ Final mAP@0.5: {metrics.box.map:.3f}")
