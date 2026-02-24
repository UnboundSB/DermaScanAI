import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "symptom_classifier_production.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "Train_Results")

NUM_CLASSES = 5
BATCH_SIZE = 32
NUM_EPOCHS = 30 
LEARNING_RATE = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform: 
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def prepare_datasets():
    print(f"--- INITIALIZING 75:20:5 STRATIFIED SPLIT ---")
    
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
    
    # 75:20:5 Stratified Logic
    for c in range(NUM_CLASSES):
        idx = np.where(targets == c)[0]
        np.random.shuffle(idx)
        
        n_total = len(idx)
        n_test = max(1, int(0.05 * n_total))
        n_val = max(1, int(0.20 * n_total))
        
        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test : n_test + n_val])
        train_idx.extend(idx[n_test + n_val :])

    print(f"Split Summary: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Balance Sampler for the Train subset
    train_targets = targets[train_idx]
    class_counts = [list(train_targets).count(c) for c in range(NUM_CLASSES)]
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(DatasetWrapper(Subset(full_dataset, train_idx), transform=train_transforms), 
                              batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(DatasetWrapper(Subset(full_dataset, val_idx), transform=val_test_transforms), 
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(DatasetWrapper(Subset(full_dataset, test_idx), transform=val_test_transforms), 
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, full_dataset.classes

def plot_telemetry(history, all_labels, all_preds, classes):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. ACCURACY & LOSS CURVES
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'dodgerblue', label='Train Loss', lw=2)
    plt.plot(epochs, history['val_loss'], 'crimson', label='Val Loss', lw=2)
    plt.title('Convergence Map: Loss Dynamics', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'dodgerblue', label='Train Acc', lw=2)
    plt.plot(epochs, history['val_acc'], 'crimson', label='Val Acc', lw=2)
    plt.title('Performance Map: Accuracy Growth', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_metrics.png"))
    plt.close()

    # 2. NORMALIZED CONFUSION MATRIX
    
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Normalized - 5% Vault)', fontweight='bold')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted Symptom')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

def main():
    multiprocessing.freeze_support()
    train_loader, val_loader, test_loader, class_names = prepare_datasets()

    print(f"\nMounting EfficientNet-B0 on {DEVICE}...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0

    print(f"\n--- INITIATING UNCHAINED TRAINING ---")
    for epoch in range(NUM_EPOCHS):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss, running_corrects = 0.0, 0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
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

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = (running_corrects.double() / len(dataloader.dataset)).item()
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)
            
            if phase == 'val':
                print(f'Epoch {epoch+1:02d} | Val Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc:.4f}')
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("\nTraining Complete. Finalizing Telemetry...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    plot_telemetry(history, all_labels, all_preds, class_names)
    print("\n--- FINAL VAULT CLASSIFICATION REPORT ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print(f"All plots saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()