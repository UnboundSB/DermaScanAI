import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
UTKFACE_DIR = r"D:\Projects\DermaScanAI\datasets\skin_ageing_symptoms\UTKFace_resized\UTKFace_resized"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, "binary_skin_tone_effnetb0.pth")

BATCH_SIZE = 32
EPOCHS = 8          # Binary tasks converge faster, 8 is plenty
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CUSTOM BINARY DATASET LOADER ---
class UTKFaceBinaryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        print(f"Scanning UTKFace for Binary classification: {root_dir}...")
        for filename in os.listdir(root_dir):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                parts = filename.split('_')
                if len(parts) >= 3:
                    try:
                        race = int(parts[2])
                        # Map to Binary targets:
                        # 0 (White) and 2 (Asian) -> 0 (Light Skin)
                        if race in [0, 2]:
                            self.image_paths.append(os.path.join(root_dir, filename))
                            self.labels.append(0.0) # Float required for BCE loss
                        # 1 (Black) and 3 (Indian) -> 1 (Dark Skin)
                        elif race in [1, 3]:
                            self.image_paths.append(os.path.join(root_dir, filename))
                            self.labels.append(1.0)
                        # We ignore 4 (Other) to keep the contrast pure
                    except ValueError:
                        continue 
                        
        print(f"Found {len(self.image_paths)} valid images for Binary training.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def train_model():
    print(f"--- STARTING BINARY EFFNET-B0 ON {DEVICE} ---")
    
    # 1. Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 2. Load Dataset
    full_dataset = UTKFaceBinaryDataset(UTKFACE_DIR, transform=data_transforms['train'])
    
    if len(full_dataset) == 0:
        print("[Error] No images found.")
        return

    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = data_transforms['val'] 

    # num_workers=0 avoids the Windows multiprocessing hang
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    }
    dataset_sizes = {'train': train_size, 'val': val_size}

    # 3. Model Setup (EfficientNet-B0)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # EXACTLY 1 OUTPUT NODE for Sigmoid binary classification
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1) 
    model = model.to(DEVICE)

    # 4. Optimizer & Loss (BCEWithLogits combines Sigmoid + CrossEntropy)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 5. Training Loop
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            loop = tqdm(dataloaders[phase], desc=f"{phase.capitalize()}")
            for inputs, labels in loop:
                inputs = inputs.to(DEVICE)
                
                # Reshape labels to match output shape [batch_size, 1]
                labels = labels.view(-1, 1).to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Calculate Accuracy. 
                # Since we use Logits, an output > 0 means the Sigmoid probability is > 0.5.
                preds = (outputs > 0.0).float()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
                
                loop.set_postfix(loss=loss.item())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save the best model based on validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f'\nTraining complete! Best Val Acc: {best_acc:4f}')

    # 6. Save Model
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Binary model successfully saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()