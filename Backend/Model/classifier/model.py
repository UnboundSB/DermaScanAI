import os
import torch
import torch.nn as nn
from torchvision import models, transforms

class SymptomClassifier:
    """
    Pure clinical classification engine. 
    Expects a pre-processed 224x224 grayscale numpy array.
    """
    def __init__(self, model_path, device=None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[*] Initializing Final4 Symptom Classifier on: {self.device}")
        
        # Must perfectly match the alphabetical order from your dataset folders
        self.classes = ['acne', 'clear_face', 'darkspots', 'puffy_eyes', 'wrinkles']
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[!] Critical Error: Classifier weights not found at {model_path}")

        # 1. Load the EfficientNet-B0 Architecture
        self.model = models.efficientnet_b0(weights=None)
        
        # 2. Modify the classification head to match your 5 final classes
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, len(self.classes))
        
        # 3. Inject your trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 4. PyTorch Standardization Pipeline
        # We only handle Tensor conversion and ImageNet normalization here.
        # Resizing and Grayscaling are already done by the upstream pipeline.
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, img_array):
        """
        Receives the clinically normalized numpy array, runs inference,
        and returns a dictionary of class confidences.
        """
        # Transform the numpy array into a normalized PyTorch tensor
        input_tensor = self.preprocess(img_array).unsqueeze(0).to(self.device)
        
        # Neural Network Inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            # Apply Softmax to convert raw math into percentages (0.0 to 1.0)
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            
        probs_array = probabilities.cpu().numpy()
        
        # Package the results cleanly for the orchestrator
        results = {}
        for i, cls_name in enumerate(self.classes):
            results[cls_name] = float(probs_array[i]) * 100
            
        return results