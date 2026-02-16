import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.transforms import functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# --- CONFIGURATION ---
# Path to your PROCESSED data (the 640x640 pngs and .txt labels)
DATA_ROOT = r"D:\Projects\DermaScanAI\datasets\face_detection\processed_640"
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
LABELS_DIR = os.path.join(DATA_ROOT, "labels")

# Training Settings
BATCH_SIZE = 16          # 16 is very safe for SSDLite on 6GB VRAM
EPOCHS = 15
LEARNING_RATE = 0.005    # SSDLite likes a slightly higher LR start
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# --- 1. CUSTOM RAM-EFFICIENT DATASET ---
class FaceDetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        # We only store filenames (strings), which take almost zero RAM
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. Lazy Load Image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Read with OpenCV
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w = image.shape[:2]

        # 2. Lazy Load Label
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_name)
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                # YOLO Format: class cx cy w h (Normalized 0-1)
                parts = list(map(float, line.strip().split()))
                
                # SSDLite needs: x1, y1, x2, y2 (Absolute Pixels)
                cx, cy, wn, hn = parts[1], parts[2], parts[3], parts[4]
                
                w_box = wn * w
                h_box = hn * h
                x1 = (cx * w) - (w_box / 2)
                y1 = (cy * h) - (h_box / 2)
                x2 = x1 + w_box
                y2 = y1 + h_box
                
                # Clamp to image boundaries to prevent NaN loss
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # Valid box check (width & height must be > 0)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(1) # Class 1 = Face

        # Convert to Tensor
        img_tensor = F.to_tensor(image) # Normalizes to [0, 1] automatically
        
        target = {}
        if len(boxes) > 0:
            target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.tensor(labels, dtype=torch.int64)
        else:
            # Handle negative samples (no faces) safely
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0), dtype=torch.int64)
            
        return img_tensor, target

def collate_fn(batch):
    """Required for object detection because images have different numbers of boxes"""
    return tuple(zip(*batch))

# --- 2. THE OPTIMIZED TRAINING LOOP ---
def main():
    print(f"--- INIT PURE PYTORCH TRAINING ON {DEVICE} ---")
    
    # Dataset
    dataset = FaceDetectionDataset(IMAGES_DIR, LABELS_DIR)
    
    # Split Train/Val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Dataloader
    # num_workers=2 is the sweet spot for Windows. Too high = RAM explosion.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=2, collate_fn=collate_fn, pin_memory=True)
    
    print(f"Training Images: {len(train_ds)} | Batch Size: {BATCH_SIZE}")

    # Model: SSDLite320 (MobileNetV3 Backbone)
    # This model is designed for speed. We load default weights to speed up convergence.
    print("Loading SSDLite320 (MobileNetV3)...")
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
    
    # Modify the Head for 2 classes (Background + Face)
    # The default has 91 classes (COCO). We shrink it to save compute.
    model.head.classification_head.num_classes = 2 
    
    model.to(DEVICE)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    
    # Learning Rate Scheduler (Drop LR every 5 epochs)
    lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # Mixed Precision Scaler (Saves VRAM)
    scaler = torch.cuda.amp.GradScaler()

    # --- TRAINING LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, targets in loop:
            # Move to GPU
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            # Forward Pass (with AMP for speed)
            with torch.cuda.amp.autocast():
                # torchvision models return a dict of losses during training
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # Backward Pass
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += losses.item()
            
            # Update Progress Bar
            loop.set_postfix(loss=losses.item())
            
            # OPTIONAL: Explicit cache clear if you still hit OOM (usually not needed with AMP)
            # torch.cuda.empty_cache()

        lr_scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")
        
        # Save Checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"face_detector_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved: face_detector_epoch_{epoch+1}.pth")

    # Final Save
    torch.save(model.state_dict(), "face_detector_final.pth")
    print("--- DONE. Model saved as 'face_detector_final.pth' ---")

if __name__ == "__main__":
    main()