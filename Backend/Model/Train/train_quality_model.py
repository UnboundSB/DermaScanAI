# File: D:\Projects\DermaScanAI\Backend\Model\Train\train_quality_model.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- CONFIGURATION ---
# Base directory for raw data (CSV location)
BASE_RAW_DIR = r"D:\Projects\DermaScanAI\datasets\quality\raw\gfiqa-20k\GFIQA-20k"
CSV_PATH = os.path.join(BASE_RAW_DIR, "mos_val_rating.csv")

# Base directory for processed data (Image location)
PROCESSED_IMG_DIR = r"D:\Projects\DermaScanAI\datasets\quality\processed\gfiqa-224-png\image"
SAVE_PATH = "iqa_gatekeeper_b0.pth"

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 15
IMG_SIZE = 224
NUM_WORKERS = 0  # Set to 0 for Windows compatibility

# --- DEVICE CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# --- DATASET CLASS ---
class IQADataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # CSV contains original filenames (likely .jpg)
        original_fname = str(row['filename'])
        score = float(row['mos'])
        
        # Map to processed filenames (.png)
        fname_root = os.path.splitext(original_fname)[0]
        png_fname = fname_root + ".png"
        
        img_path = os.path.join(self.img_dir, png_fname)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            # Fallback: create a blank image to prevent training crash
            # In production, logging this error to a file is recommended
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(score, dtype=torch.float32)

# --- TRAINING PIPELINE ---
def train_model():
    # 1. Load and Clean Data
    print(f"Loading CSV from {CSV_PATH}...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    
    # Standardize column names
    # Assumes column 0 is filename and column 1 is score
    df.rename(columns={df.columns[0]: 'filename', df.columns[1]: 'mos'}, inplace=True)
    
    # Validation Split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training Samples: {len(train_df)} | Validation Samples: {len(val_df)}")

    # 2. Transforms
    # Images are already resized to 224x224 during preprocessing, 
    # but Resize is kept here for safety.
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 3. Data Loaders
    train_dataset = IQADataset(train_df, PROCESSED_IMG_DIR, transform=train_transforms)
    val_dataset = IQADataset(val_df, PROCESSED_IMG_DIR, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    # 4. Model Initialization (EfficientNet-B0)
    print("Initializing EfficientNet-B0...")
    # Use standard weights
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    
    # Modify Classifier Head for Regression
    # The default classifier is a Sequential block; we replace it.
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 1)
    )
    
    model = model.to(device)

    # 5. Optimization
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler: Reduce LR if validation loss plateaus for 2 epochs
    # Removed verbose=True due to TypeError in newer PyTorch versions
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Mixed Precision Scaler
    scaler = GradScaler()

    # 6. Training Loop
    best_val_loss = float('inf')

    print(f"Starting training for {EPOCHS} epochs...")
    print("-" * 60)

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        
        # Training Phase
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        
        for images, scores in loop:
            images = images.to(device)
            scores = scores.to(device).unsqueeze(1)

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, scores)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, scores in val_loader:
                images = images.to(device)
                scores = scores.to(device).unsqueeze(1)
                
                # Autocast during validation saves VRAM
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, scores)
                
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        # Scheduler Step
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # End of Epoch Reporting
        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch+1} Completed | Duration: {epoch_duration:.0f}s")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"--> Best model saved to {SAVE_PATH}")
            
        print("-" * 60)

    print("Training Complete.")

if __name__ == "__main__":
    if os.path.exists(PROCESSED_IMG_DIR):
        train_model()
    else:
        print(f"CRITICAL ERROR: Processed directory not found: {PROCESSED_IMG_DIR}")
        print("Please run the preprocessing script first.")