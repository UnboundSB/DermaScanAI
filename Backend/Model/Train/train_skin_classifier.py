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
PHASE_1_EPOCHS = 10  # Frozen Backbone
PHASE_2_EPOCHS = 5   # Unfrozen Fine-tuning
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- DATA WRAPPER ---
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
    # FIX: Removed Grayscale — color is critical for skin condition classification
    # acne = redness, dark spots = pigmentation, puffy eyes = discoloration
    # Throwing away color was destroying discriminative features
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # augment color
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

    train_idx, val_idx, test_idx = [], [], []
    np.random.seed(42)

    for c in range(NUM_CLASSES):
        idx = np.where(targets == c)[0]
        np.random.shuffle(idx)
        n_test = max(1, int(0.05 * len(idx)))
        n_val  = max(1, int(0.20 * len(idx)))
        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test: n_test + n_val])
        train_idx.extend(idx[n_test + n_val:])

    train_targets = targets[train_idx]
    class_counts  = [list(train_targets).count(c) for c in range(NUM_CLASSES)]
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_idx), replacement=True)

    train_loader = DataLoader(
        DatasetWrapper(Subset(full_dataset, train_idx), transform=train_transforms),
        batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        DatasetWrapper(Subset(full_dataset, val_idx), transform=val_test_transforms),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        DatasetWrapper(Subset(full_dataset, test_idx), transform=val_test_transforms),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"\n[Dataset] Classes      : {full_dataset.classes}")
    print(f"[Dataset] Class counts : {class_counts}")
    print(f"[Dataset] Train / Val / Test : {len(train_idx)} / {len(val_idx)} / {len(test_idx)}\n")

    return train_loader, val_loader, test_loader, full_dataset.classes


def run_epoch(model, dataloader, criterion, optimizer, phase, device):
    if phase == 'train':
        model.train()
    else:
        model.eval()

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

        running_loss     += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc  = (running_corrects.double() / len(dataloader.dataset)).item()
    return epoch_loss, epoch_acc


def evaluate_model(model, test_loader, class_names, device):
    """Runs full evaluation with confusion matrix and classification report."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n--- TEST SET CLASSIFICATION REPORT ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix — Test Set")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    print(f"[*] Confusion matrix saved to {RESULTS_DIR}")


def main():
    multiprocessing.freeze_support()
    train_loader, val_loader, test_loader, class_names = prepare_datasets()

    # Build model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

    # -----------------------------------------------------------------------
    # PHASE 1: Train head only (backbone frozen)
    # -----------------------------------------------------------------------
    print("=" * 55)
    print("  PHASE 1: TRAINING HEAD ONLY (Backbone Frozen)")
    print("=" * 55)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE_1_EPOCHS)

    best_acc = 0.0
    for epoch in range(PHASE_1_EPOCHS):
        t_loss, t_acc = run_epoch(model, train_loader, criterion, optimizer, 'train', DEVICE)
        v_loss, v_acc = run_epoch(model, val_loader,   criterion, optimizer, 'val',   DEVICE)
        scheduler.step()
        print(f"  P1 Epoch {epoch+1:02d}/{PHASE_1_EPOCHS} | "
              f"Train Loss: {t_loss:.4f}  Acc: {t_acc:.4f} | "
              f"Val Loss: {v_loss:.4f}  Acc: {v_acc:.4f}")
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"    ↑ Best model saved (val_acc={best_acc:.4f})")

    # -----------------------------------------------------------------------
    # PHASE 2: Unfreeze everything and fine-tune with low LR
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  PHASE 2: GLOBAL FINE-TUNING (All Layers Unfrozen)")
    print("=" * 55)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE_2_EPOCHS)

    for epoch in range(PHASE_2_EPOCHS):
        t_loss, t_acc = run_epoch(model, train_loader, criterion, optimizer, 'train', DEVICE)
        v_loss, v_acc = run_epoch(model, val_loader,   criterion, optimizer, 'val',   DEVICE)
        scheduler.step()
        print(f"  P2 Epoch {epoch+1:02d}/{PHASE_2_EPOCHS} | "
              f"Train Loss: {t_loss:.4f}  Acc: {t_acc:.4f} | "
              f"Val Loss: {v_loss:.4f}  Acc: {v_acc:.4f}")
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"    ↑ Best model saved (val_acc={best_acc:.4f})")

    print(f"\n[✓] Training complete. Best val accuracy: {best_acc:.4f}")
    print(f"[✓] Model saved to: {MODEL_SAVE_PATH}")

    # Final test evaluation
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    evaluate_model(model, test_loader, class_names, DEVICE)


if __name__ == "__main__":
    main()