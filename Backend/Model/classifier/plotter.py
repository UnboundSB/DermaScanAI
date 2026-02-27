import os
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "plots", "model4")

def generate_telemetry_plot():
    # --- PHASE 1 DATA (Epochs 1-10: Backbone Frozen) ---
    p1_train = [0.7310, 0.8181, 0.8241, 0.8442, 0.8333, 0.8542, 0.8548, 0.8651, 0.8692, 0.8751]
    p1_val   = [0.8299, 0.8411, 0.8585, 0.8605, 0.8778, 0.8758, 0.8727, 0.8697, 0.8829, 0.8839]

    # --- PHASE 2 DATA (Epochs 11-15: Global Fine-Tuning) ---
    p2_train = [0.8882, 0.8988, 0.9107, 0.9289, 0.9267]
    p2_val   = [0.9043, 0.9084, 0.9175, 0.9196, 0.9236]

    # Combine into full arrays
    train_acc = p1_train + p2_train
    val_acc   = p1_val + p2_val
    epochs    = np.arange(1, len(train_acc) + 1)

    # 1. Ensure output directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 2. Plot Setup
    plt.figure(figsize=(12, 7))
    plt.plot(epochs, train_acc, label='Training Accuracy', color='#007acc', linewidth=3, marker='o', markersize=6)
    plt.plot(epochs, val_acc, label='Validation Accuracy', color='#d62728', linewidth=3, marker='s', markersize=6)

    # --- THE UNFREEZE LINE ---
    # Placing the line at 10.5 to sit between Phase 1 and Phase 2
    plt.axvline(x=10.5, color='#333333', linestyle='--', linewidth=2, alpha=0.9)
    
    # Text Annotations
    plt.text(10.3, 0.75, 'UNFREEZE BACKBONE', color='#333333', fontweight='bold', rotation=90, verticalalignment='center')
    plt.text(5.5, 0.72, 'PHASE 1: HEAD ONLY', color='#007acc', fontsize=11, fontweight='bold', ha='center')
    plt.text(13, 0.72, 'PHASE 2: GLOBAL TUNING', color='#d62728', fontsize=11, fontweight='bold', ha='center')

    # Styling the Graph
    plt.title('DermaScanAI Model 4: Accuracy Convergence (Phased Training)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Training Epochs (Sequential)', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (0.0 - 1.0)', fontsize=12, fontweight='bold')
    
    plt.xticks(epochs)
    plt.ylim(0.7, 1.0) # Focus on the high-performance zone
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='lower right', frameon=True, shadow=True, fontsize=11)
    
    # 3. Save to the specified folder
    save_filename = "accuracy_curve_model4.png"
    save_path = os.path.join(RESULTS_DIR, save_filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    
    print("\n" + "="*50)
    print(f"[SUCCESS] Telemetry plot saved to:\n{save_path}")
    print("="*50)
    plt.show()

if __name__ == "__main__":
    generate_telemetry_plot()