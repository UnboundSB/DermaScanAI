import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from tqdm import tqdm

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: Pointing to the ready_for_training folder to test on original ground truth
DATA_DIR = r"D:\Projects\DermaScanAI\datasets\skin_ageing_symptoms\dataset_ready_for_training"

# THE FOUR FINALISTS
MODELS_TO_TEST = [
    os.path.join(SCRIPT_DIR, "symptom_classifier_final1.pth"), 
    os.path.join(SCRIPT_DIR, "symptom_classifier_final2.pth"), 
    os.path.join(SCRIPT_DIR, "symptom_classifier_final3.pth"),
    os.path.join(SCRIPT_DIR, "symptom_classifier_final4.pth")
]

NUM_CLASSES = 5
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- GLOBAL DATASET WRAPPER ---
class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform: x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

def get_dynamic_test_loader():
    print("--- SCANNING DATASET FOR CLINICAL BALANCING ---")
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=DATA_DIR)
    targets = np.array(full_dataset.targets)
    
    class_counts = {c: np.sum(targets == c) for c in range(NUM_CLASSES)}
    min_samples = min(class_counts.values())
    
    print(f"Minimum images found in a single class: {min_samples}")
    print(f"Locking comparison test set to exactly {min_samples} images per class.")

    test_indices = []
    np.random.seed(42) 
    
    for c in range(NUM_CLASSES):
        class_idx = np.where(targets == c)[0]
        np.random.shuffle(class_idx)
        test_indices.extend(class_idx[:min_samples])

    test_data = Subset(full_dataset, test_indices)
    test_dataset = DatasetWrapper(test_data, transform=val_test_transforms)
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True)
    return test_loader, full_dataset.classes

def plot_individual_confusion_matrix(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='magma', xticklabels=classes, yticklabels=classes)
    plt.title(f'Normalized Confusion Matrix: {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('AI Prediction')
    plt.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, f"cm_{model_name}.png")
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model_path, model, test_loader, classes):
    if not os.path.exists(model_path):
        print(f"[!] Warning: {os.path.basename(model_path)} not found. Skipping.")
        return None, None
        
    model_name = os.path.basename(model_path).replace('.pth', '')
    print(f"\nEvaluating: {model_name} ...")
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Inference", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    plot_individual_confusion_matrix(all_labels, all_preds, classes, model_name)
            
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    acc = accuracy_score(all_labels, all_preds)
    
    flat_results = []
    for cls in classes:
        flat_results.append({
            "Model": model_name,
            "Class": cls,
            "Precision": report[cls]['precision'],
            "Recall": report[cls]['recall'],
            "F1-Score": report[cls]['f1-score']
        })
        
    return flat_results, {"Model": model_name, "Overall Accuracy": acc}

def generate_comparison_plots(df_metrics, df_acc):
    print("\nRendering comparative visualizations...")
    sns.set_theme(style="whitegrid")
    
    # Accuracy Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_acc, x="Model", y="Overall Accuracy", palette="viridis")
    plt.title("Master Accuracy Audit", fontweight='bold')
    plt.ylim(0.5, 1.0)
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "audit_0_overall_accuracy.png"))
    plt.close()

    metrics_to_plot = ["Precision", "Recall", "F1-Score"]
    
    # Class-wise Metrics
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.figure(figsize=(14, 7))
        sns.barplot(data=df_metrics, x="Class", y=metric, hue="Model", palette="tab10")
        plt.title(f"Diagnostic Breakdown: {metric}", fontsize=16, fontweight='bold')
        plt.ylim(0.5, 1.0)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(SCRIPT_DIR, f"audit_{i}_{metric.lower()}.png"))
        plt.close()

def main():
    multiprocessing.freeze_support()
    
    test_loader, class_names = get_dynamic_test_loader()
    
    print("\nReconstructing Base Architecture...")
    base_model = models.efficientnet_b0(weights=None)
    base_model.classifier[1] = nn.Linear(base_model.classifier[1].in_features, NUM_CLASSES)
    base_model = base_model.to(DEVICE)

    all_class_metrics = []
    all_accuracies = []
    
    for model_path in MODELS_TO_TEST:
        class_metrics, acc_metrics = evaluate_model(model_path, base_model, test_loader, class_names)
        if class_metrics and acc_metrics:
            all_class_metrics.extend(class_metrics)
            all_accuracies.append(acc_metrics)
            
    if not all_class_metrics:
        print("\n[!] Error: No brains detected for the audit.")
        return

    df_metrics = pd.DataFrame(all_class_metrics)
    df_acc = pd.DataFrame(all_accuracies)
    
    print("\n" + "="*80)
    print(" FINAL AUDIT REPORT: MULTI-GENERATIONAL PERFORMANCE ")
    print("="*80)
    print("\n--- GLOBAL ACCURACY SCORES ---")
    print(df_acc.to_string(index=False))
    
    print("\n--- F1-SCORE GRID (Symptom Detection Stability) ---")
    pivot_df = df_metrics.pivot(index='Class', columns='Model', values='F1-Score')
    print(pivot_df.to_string())
    print("="*80)
    
    generate_comparison_plots(df_metrics, df_acc)
    print(f"\n[SUCCESS] Comparative charts and confusion matrices saved to: {SCRIPT_DIR}")

if __name__ == "__main__":
    main()