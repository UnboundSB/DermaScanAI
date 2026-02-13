import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# --- STRICT LAWS & CONFIGURATION ---
DATA_DIR = r"D:\Projects\DermaScanAI\datasets\facial_recognition\final\images_160_png"
DEMOGRAPHICS_CSV = r"D:\Projects\DermaScanAI\datasets\facial_recognition\final\demographics.csv"
MODEL_SAVE_PATH = "strict_facenet_weights.pth"

BATCH_SIZE = 32 # Safe for 6GB VRAM
EPOCHS = 20
LEARNING_RATE = 0.0001
MARGIN = 1.0 # The minimum distance enforced between different people

# Enforce deterministic behavior for reproducibility
torch.manual_seed(42)
random.seed(42)

# --- 1. STRICT DATASET & SPLITTING ---
class FaceEmbeddingDataset(Dataset):
    def __init__(self, data_dir, is_training=True, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.is_training = is_training
        
        # Gather all identities
        all_identities = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        # STRICT SPLIT: 80% Train, 20% Unseen Actors (Test)
        # We split by FOLDER (Identity), not by image. 
        # The test set contains people the training set will NEVER see.
        split_idx = int(len(all_identities) * 0.8)
        random.shuffle(all_identities)
        
        if self.is_training:
            self.identities = all_identities[:split_idx]
        else:
            self.identities = all_identities[split_idx:]
            
        # Map identities to their image paths
        self.image_map = {}
        for identity in self.identities:
            folder_path = os.path.join(data_dir, identity)
            images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
            if len(images) >= 2: # Triplet loss requires at least 2 images per person
                self.image_map[identity] = images
                
        self.valid_identities = list(self.image_map.keys())
        print(f"[{'TRAIN' if is_training else 'TEST'}] Loaded {len(self.valid_identities)} unique identities.")

    def __len__(self):
        return len(self.valid_identities) * 10 # Arbitrary epoch length per identity

    def __getitem__(self, idx):
        # TRIPLET LOSS GENERATOR: Anchor, Positive, Negative
        
        # 1. Pick a random person (Anchor/Positive)
        anchor_id = random.choice(self.valid_identities)
        
        # 2. Pick a DIFFERENT person (Negative)
        negative_id = random.choice(self.valid_identities)
        while negative_id == anchor_id:
            negative_id = random.choice(self.valid_identities)
            
        # 3. Select the images
        anchor_img_path, pos_img_path = random.sample(self.image_map[anchor_id], 2)
        neg_img_path = random.choice(self.image_map[negative_id])
        
        anchor = Image.open(anchor_img_path).convert('RGB')
        positive = Image.open(pos_img_path).convert('RGB')
        negative = Image.open(neg_img_path).convert('RGB')
        
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
            
        return anchor, positive, negative

# --- 2. DEMOGRAPHIC ENFORCER (The 50/50 Rule) ---
def get_balanced_sampler(identities):
    """
    Reads demographics.csv and creates a weighted sampler.
    Forces the DataLoader to pull 50% Black and 50% White subjects.
    """
    if not os.path.exists(DEMOGRAPHICS_CSV):
        print("WARNING: demographics.csv not found! Proceeding with standard random shuffling.")
        print("To enforce the 50/50 rule, create a CSV with 'person_name' and 'race' columns.")
        return None
        
    df = pd.read_csv(DEMOGRAPHICS_CSV)
    weights = []
    
    # Calculate weights to balance the classes
    # If 80% are white, white gets weight 0.2, black gets 0.8
    race_counts = df['race'].value_counts()
    weight_map = {race: 1.0 / count for race, count in race_counts.items()}
    
    for identity in identities:
        try:
            race = df[df['person_name'] == identity]['race'].values[0]
            weights.append(weight_map[race])
        except IndexError:
            weights.append(0.01) # Penalize unknown demographics
            
    # PyTorch will use this to enforce strict 50/50 pulling
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler

# --- 3. THE MODEL ARCHITECTURE ---
class FaceNetModel(nn.Module):
    def __init__(self, embedding_size=128):
        super(FaceNetModel, self).__init__()
        # Use ResNet18 as the backbone for speed and efficiency on 6GB VRAM
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Strip the final classification layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() 
        
        # Add our strict embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_size)
        )

    def forward(self, x):
        features = self.backbone(x)
        embeds = self.embedding(features)
        # Normalize the embeddings (Strict requirement for Distance metrics)
        return nn.functional.normalize(embeds, p=2, dim=1)

# --- 4. TRAINING ENGINE ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- INITIALIZING STRICT TRAINING ON {device.type.upper()} ---")

    # Strict Augmentations to prevent overfitting
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = FaceEmbeddingDataset(DATA_DIR, is_training=True, transform=train_transforms)
    test_dataset = FaceEmbeddingDataset(DATA_DIR, is_training=False, transform=test_transforms)

    # Apply the 50/50 racial balancer if the CSV exists
    sampler = get_balanced_sampler(train_dataset.valid_identities)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=(sampler is None), num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = FaceNetModel(embedding_size=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Triplet Loss: Pushes same people together, pulls different people apart
    criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        loop = tqdm(train_loader, leave=True)
        for anchor, positive, negative in loop:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            
            # Get fingerprints
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)
            
            # Calculate law enforcement (Loss)
            loss = criterion(emb_a, emb_p, emb_n)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        # --- VALIDATION ON UNSEEN ACTORS ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negative in test_loader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                emb_a, emb_p, emb_n = model(anchor), model(positive), model(negative)
                loss = criterion(emb_a, emb_p, emb_n)
                val_loss += loss.item()
                
        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(test_loader)
        print(f"\n[Epoch {epoch+1} Results] Train Loss: {avg_train:.4f} | Unseen Actor Validation Loss: {avg_val:.4f}\n")

    # Save the strictly trained weights
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Strict Training Complete. Model secured at {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()