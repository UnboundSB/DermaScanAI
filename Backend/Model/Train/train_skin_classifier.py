import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing

# --- CONFIGURATION ---
BASE_DIR = r"D:\Projects\DermaScanAI\datasets\skin_ageing_symptoms"
DATA_DIR = os.path.join(BASE_DIR, "dataset_augmented_224_png")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "symptom_classifier_phased.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "Phased_Train_Results")

NUM_CLASSES = 5
BATCH_SIZE = 32
PHASE_1_EPOCHS = 10 # Frozen Backbone
PHASE_2_EPOCHS = 5  # Unfrozen Fine-tuning
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATA PREP (Same 75:20:5 Stratified Split) ---
class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform: x = self.transform(x)
        return x, y
    def __len__(self): return len(self.subset)

def prepare_datasets():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    full_dataset = datasets.ImageFolder(root=DATA_DIR)
    targets = np.array(full_dataset.targets)
    train_idx, val_idx, test_idx = [], [], []
    np.random.seed(42)
    for c in range(NUM_CLASSES):
        idx = np.where(targets == c)[0]
        np.random.shuffle(idx)
        n_test = max(1, int(0.05 * len(idx)))
        n_val = max(1, int(0.20 * len(idx)))
        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test : n_test + n_val])
        train_idx.extend(idx[n_test + n_val :])
    
    train_targets = targets[train_idx]
    class_weights = 1.0 / torch.tensor([list(train_targets).count(c) for c in range(NUM_CLASSES)], dtype=torch.float)
    sampler = WeightedRandomSampler(weights=[class_weights[t] for t in train_targets], num_samples=len(train_idx), replacement=True)

    train_loader = DataLoader(DatasetWrapper(Subset(full_dataset, train_idx), transform=train_transforms), batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(DatasetWrapper(Subset(full_dataset, val_idx), transform=val_test_transforms), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(DatasetWrapper(Subset(full_dataset, test_idx), transform=val_test_transforms), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader, full_dataset.classes

def run_epoch(model, dataloader, criterion, optimizer, phase, device):
    if phase == 'train': model.train()
    else: model.eval()
    running_loss, running_corrects = 0.0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            if phase == 'train':
                loss.backward()
                optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    return running_loss / len(dataloader.dataset), (running_corrects.double() / len(dataloader.dataset)).item()

def main():
    multiprocessing.freeze_support()
    train_loader, val_loader, test_loader, class_names = prepare_datasets()
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # --- PHASE 1: FREEZE BACKBONE ---
    print("\n--- PHASE 1: TRAINING HEAD ONLY (Backbone Frozen) ---")
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3) # Higher LR for head
    
    best_acc = 0.0
    for epoch in range(PHASE_1_EPOCHS):
        t_loss, t_acc = run_epoch(model, train_loader, criterion, optimizer, 'train', DEVICE)
        v_loss, v_acc = run_epoch(model, val_loader, criterion, optimizer, 'val', DEVICE)
        print(f"P1 Epoch {epoch+1:02d} | Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f}")
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # --- PHASE 2: UNFREEZE & FINE-TUNE ---
    print("\n--- PHASE 2: GLOBAL FINE-TUNING (All Layers Unfrozen) ---")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    for param in model.parameters():
        param.requires_grad = True
    
    # Lower LR for fine-tuning to prevent destroying pre-trained weights
    optimizer = optim.AdamW(model.parameters(), lr=1e-5) 
    
    for epoch in range(PHASE_2_EPOCHS):
        t_loss, t_acc = run_epoch(model, train_loader, criterion, optimizer, 'train', DEVICE)
        v_loss, v_acc = run_epoch(model, val_loader, criterion, optimizer, 'val', DEVICE)
        print(f"P2 Epoch {epoch+1:02d} | Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f}")
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("\nTraining Complete. Model Saved.")

if __name__ == "__main__":
    main()