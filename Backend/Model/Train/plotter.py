import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
BASE_DIR = r"D:\Projects\DermaScanAI\datasets\skin_ageing_symptoms"
DATA_DIR = os.path.join(BASE_DIR, "dataset_augmented_224_png")
MODEL_PATH = os.path.join(BASE_DIR, "symptom_classifier_phased.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "Final_Audit_Results")

NUM_CLASSES = 5
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_test_loader():
    """Reconstructs the exact 5% test vault used during training."""
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=DATA_DIR)
    targets = np.array(full_dataset.targets)
    
    test_idx = []
    np.random.seed(42) # MUST match the trainer's seed to see the same 'unseen' data
    
    for c in range(NUM_CLASSES):
        idx = np.where(targets == c)[0]
        np.random.shuffle(idx)
        n_test = max(1, int(0.05 * len(idx)))
        test_idx.extend(idx[:n_test])

    test_dataset = Subset(full_dataset, test_idx)
    # Applying transforms via a wrapper if needed, but for plotting, standard Subset is fine
    # We redefine the transform on the dataset object for the test run:
    full_dataset.transform = test_transforms
    
    return DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False), full_dataset.classes

def plot_visuals(all_labels, all_preds, classes):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. NORMALIZED CONFUSION MATRIX
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='magma', xticklabels=classes, yticklabels=classes)
    plt.title('Final Audit: Normalized Confusion Matrix\n(Unseen 5% Vault)', fontweight='bold', fontsize=14)
    plt.ylabel('Ground Truth (Actual Symptom)', fontsize=12)
    plt.xlabel('AI Prediction', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "final_confusion_matrix.png"), dpi=300)
    print(f"[*] Confusion Matrix saved to {RESULTS_DIR}")

    # 2. CLASS PRECISION BAR CHART
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    precision = [report[cls]['precision'] for cls in classes]
    recall = [report[cls]['recall'] for cls in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, precision, width, label='Precision', color='skyblue')
    ax.bar(x + width/2, recall, width, label='Recall', color='salmon')

    ax.set_ylabel('Score')
    ax.set_title('Clinical Precision & Recall per Symptom', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(RESULTS_DIR, "precision_recall_audit.png"), dpi=300)
    print(f"[*] Precision/Recall chart saved to {RESULTS_DIR}")

def main():
    print(f"--- INITIATING FINAL MODEL AUDIT ON {DEVICE} ---")
    
    # Load Model
    model = models.efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(DEVICE)
    model.eval()

    # Load Data
    test_loader, class_names = get_test_loader()
    
    all_preds = []
    all_labels = []

    print("Running inference on unseen vault images...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate Report
    print("\n--- FINAL CLASSIFICATION REPORT ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Plot
    plot_visuals(all_labels, all_preds, class_names)
    print("\nAudit Complete. You are officially ready for deployment.")

if __name__ == "__main__":
    main()