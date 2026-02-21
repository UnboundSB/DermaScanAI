import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing

# --- CONFIGURATION ---
BASE_DIR = r"D:\Projects\DermaScanAI\datasets\skin_ageing_symptoms"
DATA_DIR = os.path.join(BASE_DIR, "dataset_ready_for_training")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "symptom_classifier_optimized.pth")
PLOT_DIR = os.path.join(BASE_DIR, "Training_Plots")

NUM_CLASSES = 5
BATCH_SIZE = 32
NUM_EPOCHS = 20 # Bumped up to 20 because early stopping will catch it if it finishes early
LEARNING_RATE = 1e-4
TEST_SAMPLES_PER_CLASS = 200
EARLY_STOPPING_PATIENCE = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- GLOBAL DATASET WRAPPER (Windows Multiprocessing Fix) ---
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
    print("--- PREPARING 200-IMAGE BALANCED TEST SPLIT ---")
    
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=DATA_DIR)
    targets = np.array(full_dataset.targets)
    
    test_indices = []
    train_val_indices = []

    np.random.seed(42) 
    for c in range(NUM_CLASSES):
        class_idx = np.where(targets == c)[0]
        np.random.shuffle(class_idx)
        test_indices.extend(class_idx[:TEST_SAMPLES_PER_CLASS])
        train_val_indices.extend(class_idx[TEST_SAMPLES_PER_CLASS:])

    np.random.shuffle(train_val_indices)
    split_point = int(0.8 * len(train_val_indices))
    train_indices = train_val_indices[:split_point]
    val_indices = train_val_indices[split_point:]

    train_data = Subset(full_dataset, train_indices)
    val_data = Subset(full_dataset, val_indices)
    test_data = Subset(full_dataset, test_indices)

    train_dataset = DatasetWrapper(train_data, transform=train_transforms)
    val_dataset = DatasetWrapper(val_data, transform=val_test_transforms)
    test_dataset = DatasetWrapper(test_data, transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True)

    return train_loader, val_loader, test_loader, full_dataset.classes, targets[train_indices]

def plot_training_curves(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "optimized_training_curves.png"))
    plt.close()
    print(f"Saved Optimized Learning Curves to {save_dir}")

def plot_confusion_matrix(y_true, y_pred, classes, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Balanced 200-Image Test Set)')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "optimized_confusion_matrix.png"))
    plt.close()

def evaluate_on_test_set(model, test_loader, classes):
    print("\n--- RUNNING FINAL TEST ON BALANCED VAULT ---")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    plot_confusion_matrix(all_labels, all_preds, classes, PLOT_DIR)
    print("\nOptimized Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

def main():
    multiprocessing.freeze_support()
    
    train_loader, val_loader, test_loader, class_names, train_targets = prepare_datasets()
    
    print("\nCalculating CSL Weights for Training Set...")
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_targets), y=train_targets)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    print("\nLoading EfficientNet-B0...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Fixed line for PyTorch 2.2+
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # --- NEW: Early Stopping Trackers ---
    best_loss = float('inf')
    best_acc = 0.0
    epochs_no_improve = 0

    print(f"\n--- STARTING OPTIMIZED TRAINING ON {DEVICE} ---")
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        
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

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            if phase == 'val':
                # Step the scheduler based on validation loss
                scheduler.step(epoch_loss)
                
                # Check Early Stopping
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                    # Save model if it's the best accuracy we've seen
                    if epoch_acc >= best_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), MODEL_SAVE_PATH)
                        print(f"*** New Best Model Saved (Val Loss: {epoch_loss:.4f}) ***")
                else:
                    epochs_no_improve += 1
                    print(f"Early Stopping Counter: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

        # Break out of the outer epoch loop if patience is exceeded
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n[!] Early stopping triggered. Validation loss hasn't improved in {EARLY_STOPPING_PATIENCE} epochs.")
            break

    print("\nTraining complete. Generating plots...")
    plot_training_curves(history, PLOT_DIR)
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    evaluate_on_test_set(model, test_loader, class_names)

if __name__ == "__main__":
    main()