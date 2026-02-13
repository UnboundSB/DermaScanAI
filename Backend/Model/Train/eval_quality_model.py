import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_RAW_DIR = r"D:\Projects\DermaScanAI\datasets\quality\raw\gfiqa-20k\GFIQA-20k"
CSV_PATH = os.path.join(BASE_RAW_DIR, "mos_val_rating.csv")
PROCESSED_IMG_DIR = r"D:\Projects\DermaScanAI\datasets\quality\processed\gfiqa-224-png\image"
MODEL_PATH = "iqa_gatekeeper_b0.pth"
IMG_SIZE = 224
BATCH_SIZE = 32

# --- DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET CLASS (Reused) ---
class IQADataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname_root = os.path.splitext(str(row['filename']))[0]
        png_fname = fname_root + ".png"
        img_path = os.path.join(self.img_dir, png_fname)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))

        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(float(row['mos']), dtype=torch.float32)

def load_inference_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = models.efficientnet_b0(weights=None) # No internet weights needed, loading local
    
    # Recreate the exact same head architecture
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 1)
    )
    
    # Load weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

def run_quantitative_test(model):
    print("\n--- Running Quantitative Validation ---")
    
    # Load Data
    df = pd.read_csv(CSV_PATH)
    df.rename(columns={df.columns[0]: 'filename', df.columns[1]: 'mos'}, inplace=True)
    
    # Use same random state to get the exact same validation set
    _, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_loader = DataLoader(
        IQADataset(val_df, PROCESSED_IMG_DIR, transform=val_transforms),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    predictions = []
    actuals = []
    
    with torch.no_grad():
        for images, scores in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            preds = model(images).cpu().squeeze().tolist()
            scores = scores.tolist()
            
            # Handle single-item batches
            if isinstance(preds, float):
                predictions.append(preds)
                actuals.append(scores)
            else:
                predictions.extend(preds)
                actuals.extend(scores)

    # Calculate Metrics
    predictions = torch.tensor(predictions)
    actuals = torch.tensor(actuals)
    
    mse = nn.MSELoss()(predictions, actuals).item()
    mae = nn.L1Loss()(predictions, actuals).item()
    
    print(f"\nFinal Results on {len(val_df)} Validation Images:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Interpretation: On average, predictions are off by {mae:.4f} points.")
    
    # Generate Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.3, s=5)
    plt.plot([0, 1], [0, 1], 'r--', label="Perfect Prediction") # Diagonal line
    plt.xlabel("Actual Human Score (MOS)")
    plt.ylabel("Model Prediction")
    plt.title(f"IQA Model Accuracy (MAE: {mae:.4f})")
    plt.legend()
    plt.grid(True)
    plt.savefig("evaluation_scatter_plot.png")
    print("Saved 'evaluation_scatter_plot.png'")

def run_stress_test(model):
    print("\n--- Running Qualitative Stress Test (Blur Check) ---")
    
    # Find a random high-quality image from the processed folder
    all_files = os.listdir(PROCESSED_IMG_DIR)
    test_file = all_files[0] # Just pick the first one
    img_path = os.path.join(PROCESSED_IMG_DIR, test_file)
    
    # Load Image
    original_img = Image.open(img_path).convert("RGB")
    
    # Create a Blurry Version
    blurred_img = original_img.filter(ImageFilter.GaussianBlur(radius=5))
    
    # Transform
    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    tensor_orig = preprocess(original_img).unsqueeze(0).to(device)
    tensor_blur = preprocess(blurred_img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        score_orig = model(tensor_orig).item()
        score_blur = model(tensor_blur).item()
        
    print(f"Test Image: {test_file}")
    print(f"Original Score: {score_orig:.4f}")
    print(f"Blurred Score:  {score_blur:.4f}")
    
    if score_blur < score_orig:
        print("PASS: Model successfully penalized the blurred image.")
    else:
        print("FAIL: Model did not penalize the blur.")
        
    # Save visual comparison
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_img)
    ax[0].set_title(f"Original\nScore: {score_orig:.2f}")
    ax[0].axis('off')
    
    ax[1].imshow(blurred_img)
    ax[1].set_title(f"Artificially Blurred\nScore: {score_blur:.2f}")
    ax[1].axis('off')
    
    plt.savefig("stress_test_comparison.png")
    print("Saved 'stress_test_comparison.png'")

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH) and os.path.exists(PROCESSED_IMG_DIR):
        model = load_inference_model()
        run_quantitative_test(model)
        run_stress_test(model)
    else:
        print("Error: Could not find model or dataset path.")